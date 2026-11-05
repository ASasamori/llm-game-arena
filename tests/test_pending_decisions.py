"""Incomplete requests retain evidence without claiming a state transition."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import os
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from game_arena.openspiel.harness import EpisodeStopped, harness
from game_arena.openspiel.datasets import DatasetStore
from game_arena.openspiel.experiments import (
    ExperimentPlan,
    PlanConfig,
    _execute_mock_job,
)
from game_arena.openspiel.model_generation import GenerateResponse, MockModel
from game_arena.openspiel.replay import ReplayError, replay_episode
from game_arena.openspiel.utils import HarnessOptions
from orchestrator import tasks
from orchestrator.models import Base, EpisodeStep, EpisodeTrace, EpisodeTransition, Run


class RecordedPolicy(MockModel):
    def __init__(self, name="recorded", response: str | None = None):
        super().__init__(name)
        self.response = response
        self.calls = 0

    def generate_with_text_input(self, prompt, system_instruction=None, mock=False):
        self.calls += 1
        generated = super().generate_with_text_input(prompt, system_instruction, mock)
        return GenerateResponse(
            self.response if self.response is not None else generated.main_response,
            usage={"input_tokens": 17, "output_tokens": 9},
            metadata={"mock": True, "original": "retained"},
        )


class ThrowingPolicy(MockModel):
    def __init__(self, error: Exception):
        super().__init__("stopping")
        self.error = error
        self.calls = 0

    def generate_with_text_input(self, prompt, system_instruction=None, mock=False):
        self.calls += 1
        raise self.error


def options(models, game="matrix_rps", **changes) -> Any:
    return {
        "type": "PVP",
        "models": models,
        "game": game,
        "num_steps": 20,
        "input": None,
        "seed": 123,
        "failure_policy": "raise",
    } | changes


def interrupted(first: MockModel | None = None) -> dict[str, Any]:
    return dict(
        harness(
            options(
                [
                    first or RecordedPolicy(),
                    ThrowingPolicy(
                        EpisodeStopped("provider_error", "Offline controlled stop")
                    ),
                ],
                failure_policy="random",
            )
        )
    )


class PendingDecisionTests(unittest.TestCase):
    def test_joint_stop_preserves_completed_response_and_unapplied_failure(self):
        first = RecordedPolicy()
        result = interrupted(first)
        self.assertEqual(first.calls, 1)
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["decision_count"], 0)
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["pending_decision_count"], 1)
        self.assertEqual(result["failed_attempt_count"], 1)
        pending = result["pending_transition"]
        self.assertFalse(pending["applied"])
        self.assertEqual(pending["state_hash_before"], result["final_state_hash"])
        self.assertEqual(pending["decision_count"], 1)
        decision = pending["decisions"][0]
        self.assertIn("Final Answer", decision["response"])
        self.assertEqual(decision["usage"], {"input_tokens": 17, "output_tokens": 9})
        self.assertEqual(decision["response_metadata"]["original"], "retained")
        self.assertEqual(decision["selection"]["kind"], "mock")
        failed = pending["failed_attempt"]
        self.assertEqual(failed["player"], 1)
        self.assertIsInstance(failed["prompt"], str)
        self.assertIsNone(failed["action"])
        self.assertEqual(failed["failure"]["stage"], "model_call")
        verification = replay_episode(result)
        self.assertTrue(verification["pending_transition_verified"])
        self.assertEqual(verification["failure_claim_verification"], "recorded_only")

    def test_unapplied_fallback_has_separate_counts_and_still_replays(self):
        result = interrupted(RecordedPolicy(response="Offline invalid action response"))
        self.assertEqual(result["fallback_count"], 0)
        self.assertFalse(result["degraded"])
        self.assertEqual(result["pending_fallback_count"], 1)
        pending = result["pending_transition"]["decisions"][0]
        self.assertEqual(pending["response"], "Offline invalid action response")
        self.assertEqual(pending["selection"]["kind"], "fallback")
        self.assertTrue(replay_episode(result)["verified"])

    def test_strict_failure_keeps_original_exception_and_applied_prefix(self):
        error = RuntimeError("Offline provider exception")
        first = RecordedPolicy()
        with self.assertRaises(RuntimeError) as caught:
            harness(options([first, ThrowingPolicy(error)], game="tic_tac_toe"))
        self.assertIs(caught.exception, error)
        result = getattr(error, "arena_partial_result")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["num_steps"], 1)
        self.assertEqual(result["decision_count"], 1)
        self.assertEqual(result["pending_decision_count"], 0)
        self.assertEqual(result["pending_transition"]["index"], 1)
        self.assertEqual(
            result["steps"][0]["decisions"][0]["usage"]["input_tokens"], 17
        )
        self.assertTrue(replay_episode(dict(result))["verified"])

    def test_strict_parse_failure_retains_raw_invalid_output_and_usage(self):
        with self.assertRaises(ValueError) as caught:
            harness(
                options(
                    [
                        RecordedPolicy("first"),
                        RecordedPolicy("second", "Invalid output preserved exactly"),
                    ]
                )
            )
        result = getattr(caught.exception, "arena_partial_result")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["pending_decision_count"], 1)
        failed = result["pending_transition"]["failed_attempt"]
        self.assertEqual(failed["response"], "Invalid output preserved exactly")
        self.assertEqual(failed["usage"], {"input_tokens": 17, "output_tokens": 9})
        self.assertEqual(failed["response_metadata"]["original"], "retained")
        self.assertEqual(failed["failure"]["stage"], "parse")
        self.assertTrue(replay_episode(dict(result))["verified"])

    def test_pending_evidence_tampering_is_rejected(self):
        original = interrupted()
        for field in (
            "prompt",
            "response",
            "legal_actions",
            "state_hash",
            "applied",
            "count",
            "missing",
        ):
            changed = deepcopy(original)
            pending = changed["pending_transition"]
            if field == "prompt":
                pending["failed_attempt"]["prompt"] += " changed"
            elif field == "response":
                pending["decisions"][0]["response"] = (
                    "Summary: modified\nFinal Answer: 1"
                )
            elif field == "legal_actions":
                pending["failed_attempt"]["legal_actions"] = []
            elif field == "state_hash":
                pending["state_hash_before"] = "changed"
            elif field == "applied":
                pending["applied"] = True
            elif field == "count":
                changed["pending_decision_count"] = 0
            else:
                changed.pop("pending_transition")
            with self.subTest(field=field), self.assertRaises(ReplayError):
                replay_episode(changed)

    def test_legacy_controlled_stop_without_pending_fields_remains_readable(self):
        record = interrupted()
        record["manifest"]["recorded_features"].remove("pending_attempts_v1")
        for key in (
            "pending_transition",
            "pending_decision_count",
            "pending_fallback_count",
            "failed_attempt_count",
        ):
            record.pop(key)
        self.assertTrue(replay_episode(record)["verified"])

    def test_script_errors_retain_requested_input_and_do_not_apply_it(self):
        for script, expected_steps in (([99], 1), ([1], 3)):
            with self.subTest(script=script), self.assertRaises(ValueError) as caught:
                harness(
                    options(
                        [RecordedPolicy()],
                        game="tic_tac_toe",
                        type="INPUT",
                        input=deque(script),
                    )
                )
            result = getattr(caught.exception, "arena_partial_result")
            self.assertEqual(result["num_steps"], expected_steps)
            failed = result["pending_transition"]["failed_attempt"]
            self.assertEqual(failed["player"], 1)
            self.assertIsNone(failed["prompt"])
            if script == [99]:
                self.assertEqual(failed["requested_action"], 99)
            self.assertTrue(replay_episode(dict(result))["verified"])


class PendingWorkerPersistenceTests(unittest.TestCase):
    def test_offline_executor_keeps_failed_prefix_in_its_dataset(self):
        plan = ExperimentPlan(
            PlanConfig(games=("tic_tac_toe",), seed=123, max_steps=20)
        )
        job = next(plan.jobs())
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(plan.manifest)
            with (
                patch(
                    "game_arena.openspiel.model_generation.MockModel",
                    side_effect=[
                        RecordedPolicy("mock"),
                        RecordedPolicy("mock", "Offline invalid output"),
                    ],
                ),
                patch(
                    "game_arena.openspiel.harness.harness",
                    side_effect=lambda supplied: harness(
                        cast(
                            HarnessOptions, dict(supplied) | {"failure_policy": "raise"}
                        )
                    ),
                ),
            ):
                status = _execute_mock_job(job, directory)
            record = store.read_episode(job.episode_id)
        self.assertEqual(status, "FAILED")
        self.assertEqual(record["num_steps"], 1)
        self.assertEqual(
            record["pending_transition"]["failed_attempt"]["response"],
            "Offline invalid output",
        )
        self.assertEqual(record["failure_type"], "ValueError")
        self.assertTrue(replay_episode(record)["verified"])

    def test_worker_persists_failed_prefix_and_duplicate_delivery_reuses_it(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        run_id = str(uuid.uuid4())
        with sessions() as session:
            session.add(
                Run(run_id=run_id, status="PENDING", run_type="PVP", num_episodes=1)
            )
            session.commit()
        first = RecordedPolicy()
        second = RecordedPolicy("invalid", "Original failed provider output")
        settings = {
            "run_id": run_id,
            "episode_num": 0,
            "num_steps": 20,
            "model_names": ["MOCK:first", "MOCK:second"],
            "run_type": "PVP",
            "game": "tic_tac_toe",
            "seed": 123,
            "rotate_seats": False,
            "failure_policy": "raise",
        }
        with (
            patch.object(tasks, "get_session", sessions),
            patch.object(tasks, "_get_model", side_effect=[first, second]) as factory,
        ):
            result = tasks._execute_episode(**settings)["result"]
            reused = tasks._execute_episode(**settings)
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(reused["reused"])
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["num_steps"], 1)
        self.assertEqual(
            result["pending_transition"]["failed_attempt"]["response"],
            "Original failed provider output",
        )
        with sessions() as session:
            trace = session.get(EpisodeTrace, result["episode_id"])
            self.assertIsNotNone(trace)
            assert trace is not None
            record = dict(trace.payload)
            record["steps"] = list(
                session.execute(
                    select(EpisodeTransition.payload)
                    .where(EpisodeTransition.episode_id == result["episode_id"])
                    .order_by(EpisodeTransition.step_index)
                ).scalars()
            )
            self.assertEqual(
                session.execute(select(func.count(EpisodeStep.step_id))).scalar_one(), 1
            )
            self.assertEqual(
                record["pending_transition"]["failed_attempt"]["response"],
                "Original failed provider output",
            )
        self.assertTrue(replay_episode(record)["verified"])


if __name__ == "__main__":
    unittest.main()
