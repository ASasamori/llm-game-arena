"""Jev policies, persistence and capped trials without provider requests."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from game_arena.openspiel import catalog, model_generation
from game_arena.openspiel.budgeted_anthropic import BudgetLedger, JEV_PRICING_SOURCE
from game_arena.openspiel.harness import harness
from game_arena.openspiel.jev import JevModel
from game_arena.openspiel.model_generation import GenerateResponse, LLMModel
from game_arena.openspiel.replay import replay_episode
from game_arena.openspiel.utils import EpisodeResult, HarnessOptions
from orchestrator import models, tasks
from orchestrator.live_trial import run_trial, submit_trial
from orchestrator.reports import build_run_report, render_text_report


class _ApprovingJev(JevModel):
    """Keep the real adapter envelope while using a deterministic offline policy."""

    def __init__(self, model_name: str, **_kwargs: Any):
        super().__init__(model_name=model_name, mock_mode=True)

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        response = super().generate_with_text_input(prompt, system_instruction, True)
        context = json.JSONDecoder().raw_decode(prompt[prompt.rfind("\n{") + 1 :])[0]
        observation = json.loads(context["observation"])
        if observation["phase"] != "vote":
            return response
        legal = {entry["id"]: entry["label"] for entry in context["legal_actions"]}
        action = 1
        assert action in legal
        response.metadata["probabilities"] = {
            key: float(key == f"action_{action}")
            for key in response.metadata["probabilities"]
        }
        response.metadata["chosen_label"] = legal[action]
        response.main_response = (
            "Summary: Adapter mock selected a Jev choice; no explanation was generated.\n"
            f"Final Answer: {action}"
        )
        return response


class JevIntegrationTests(unittest.TestCase):
    def execute(self, competitors: list[LLMModel] | None = None) -> EpisodeResult:
        competitors = competitors or [
            JevModel(
                model_name="jev-1.14.0" if seat % 2 else "jev-1.13.0", mock_mode=True
            )
            for seat in range(6)
        ]
        return harness(
            HarnessOptions(
                type="PVP",
                models=competitors,
                num_steps=1000,
                input=None,
                game="avalon(players=6)",
                seed=123,
                failure_policy="raise",
            )
        )

    def test_provider_validation_and_factory_obey_dynamic_mock_mode(self) -> None:
        self.assertEqual(
            tasks.validate_model_name("JEV:jev-1.14.0"), ("JEV", "jev-1.14.0")
        )
        self.assertEqual(tasks.validate_model_name("jev"), ("JEV", None))
        with (
            patch.object(model_generation, "MODEL_API_MOCK_MODE", True),
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "", "JEV_API_KEY": ""}),
            patch(
                "game_arena.openspiel.jev._post",
                side_effect=AssertionError("network used"),
            ),
        ):
            competitor = tasks._get_model("JEV:jev-1.14.0")
            self.assertIsInstance(competitor, JevModel)
            self.assertTrue(getattr(competitor, "_mock_mode"))
        with self.assertRaises(ValueError):
            tasks.validate_model_name("JEV:")

    def test_six_seat_game_uses_legal_scoped_choices_and_replays_without_network(
        self,
    ) -> None:
        with patch(
            "game_arena.openspiel.jev._post", side_effect=AssertionError("network used")
        ):
            record = self.execute()
        self.assertIn(record["status"], {"SUCCESS", "TRUNCATED"})
        self.assertEqual(record.get("fallback_count"), 0)
        self.assertFalse(record.get("degraded"))
        self.assertTrue(replay_episode(dict(record))["verified"])
        state = catalog.load_game(
            record.get("game_spec", "avalon(players=6)"), seed=123
        ).new_initial_state()
        decisions = 0
        for event in record.get("steps", []):
            self.assertNotIn("state_before", event)
            self.assertNotIn("state_after", event)
            for decision in event["decisions"]:
                decisions += 1
                player = decision["player"]
                self.assertIn(decision["action"], state.legal_actions(player))
                self.assertEqual(
                    decision["observation"], state.observation_string(player)
                )
                metadata = decision["response_metadata"]
                self.assertTrue(metadata["mock"])
                self.assertEqual(
                    metadata["decision_type"],
                    "forced_legal_action"
                    if len(decision["legal_actions"]) == 1
                    else "choice",
                )
                self.assertEqual(metadata["summary_source"], "adapter")
                self.assertIsNone(metadata["provider_response"])
                self.assertEqual(
                    metadata["probabilities"][f"action_{decision['action']}"], 1.0
                )
                context = json.JSONDecoder().raw_decode(
                    decision["prompt"][decision["prompt"].rfind("\n{") + 1 :]
                )[0]
                self.assertEqual(context["observation"], decision["observation"])
                self.assertNotIn("state", context)
                self.assertNotIn("chance_outcomes", context)
                self.assertNotIn("other_players", context)
            state.apply_action(event["actions"][0])
        self.assertGreater(decisions, 0)
        self.assertEqual(len(record.get("returns", [])), 6)

    def test_persisted_report_distinguishes_adapter_envelope_from_typed_provider_output(
        self,
    ) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:")
        self.addCleanup(engine.dispose)
        models.Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        identity = str(uuid.uuid4())
        with sessions() as session:
            session.add(
                models.Run(
                    run_id=identity, run_type="PVP", num_episodes=1, status="PENDING"
                )
            )
            session.commit()

        def transport(request, timeout):
            self.assertEqual(timeout, 20.0)
            payload = json.loads(request.data)
            keys = list(payload["questions"]["action"]["criteria"])
            return {
                "model": payload["model"],
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": keys[0],
                        "confidence": 1.0,
                        "probabilities": {key: float(key == keys[0]) for key in keys},
                    }
                },
                "usage": {"input_tokens": 20, "output_tokens": 10},
            }, "fake-request"

        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(
                Path(directory) / "budget.sqlite",
                budget_usd="0.10",
                pricing_source=JEV_PRICING_SOURCE,
            )
            competitors: list[LLMModel] = [
                JevModel(
                    model_name="jev-1.14.0",
                    api_key="unit-test-key",  # pragma: allowlist secret -- synthetic test credential or sentinel only
                    ledger=ledger,
                    transport=transport,
                    mock_mode=False,
                )
                for _ in range(6)
            ]
            record = self.execute(competitors)
        with patch.object(tasks, "get_session", sessions):
            tasks._persist_result(record, identity, None)
        with sessions() as session:
            report = build_run_report(session, identity, step_limit=200)
        self.assertFalse(report["selected_episode"]["mock"])
        saved = [
            decision
            for step in report["transitions"]["items"]
            for decision in step["decisions"]
        ]
        self.assertTrue(saved)
        self.assertEqual(saved[0]["response_metadata"]["summary_source"], "adapter")
        typed = saved[0]["response_metadata"]["provider_response"]
        self.assertEqual(typed["answers"]["action"]["type"], "choice")
        text = render_text_report(report)
        self.assertIn("Adapter decision summary", text)
        self.assertIn('"answers"', text)
        self.assertIn('"probabilities"', text)
        self.assertNotIn("MOCK OUTPUT", text)

    def test_capped_jev_trial_keeps_three_public_quests_and_reuses_existing_episode(
        self,
    ) -> None:
        engine = create_engine("sqlite+pysqlite:///:memory:")
        self.addCleanup(engine.dispose)
        models.Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(models, "get_session", sessions),
            patch.object(tasks, "get_session", sessions),
            patch("redis.Redis"),
            patch("rq.Queue") as queue,
            patch(
                "game_arena.openspiel.jev.JevModel",
                side_effect=lambda name, **kwargs: _ApprovingJev(name),
            ) as factory,
        ):
            queue.return_value.enqueue.return_value = SimpleNamespace(
                id="jev-offline-job"
            )
            root = Path(directory)
            trial = submit_trial(root, provider="jev")
            configuration = trial["configuration"]
            self.assertEqual(configuration["budget_usd"], "0.10")
            self.assertTrue(
                all(name.startswith("JEV:") for name in configuration["model_names"])
            )
            self.assertEqual(len(configuration["model_names"]), 6)
            factory.side_effect = ValueError("Missing worker credential")
            with self.assertRaisesRegex(ValueError, "Missing worker credential"):
                run_trial(trial["run_id"], str(root))
            with sqlite3.connect(root / "budget.sqlite") as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM episode_claims"
                    ).fetchone()[0],
                    0,
                )
            with sessions() as session:
                run = session.get(models.Run, trial["run_id"])
                assert run is not None
                self.assertEqual(run.status, "PENDING")
            factory.reset_mock()
            factory.side_effect = lambda name, **kwargs: _ApprovingJev(name)
            result = run_trial(trial["run_id"], str(root))
            self.assertEqual(result["status"], "TRUNCATED")
            self.assertEqual(result["stop_reason"], "quest_limit")
            self.assertTrue(result["replay_verified"])
            record = json.loads((root / "episode.json").read_text())
            self.assertEqual(len(record["game_progress"]["quest_results"]), 3)
            self.assertEqual(record["game_progress"]["phase"], "assassinate")
            self.assertTrue(
                all(model["mock"] for model in record["manifest"]["model_configs"])
            )
            self.assertEqual(result["budget"]["accounted_cost_usd"], "0.000000000")
            self.assertTrue(run_trial(trial["run_id"], str(root))["reused"])
            self.assertEqual(factory.call_count, 6)
            with self.assertRaises(ValueError):
                submit_trial(root, provider="jev")
