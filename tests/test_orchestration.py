"""Orchestrator conformance without Redis, Postgres, or provider credentials."""

from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime
from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from game_arena.openspiel import utils
from orchestrator import api_gateway, jobs, ratings, tasks
from orchestrator.models import (
    Base,
    Episode,
    EpisodeStep,
    EpisodeTrace,
    EpisodeTransition,
    Matchup,
    MatchupProgress,
    ModelRating,
    RatingEvent,
    Run,
    RunProgress,
    RunConfiguration,
    RunDispatch,
)


def _result(
    episode_id: str, *, status: str = "SUCCESS", models: list[str] | None = None
) -> utils.EpisodeResult:
    models = models or ["MOCK:a", "MOCK:b", "MOCK:c"]
    now = datetime.now()
    decisions = [
        {
            "player": index,
            "agent": model,
            "observation": "private",
            "prompt": "prompt",
            "response": "response",
            "legal_actions": [0],
            "action": 0,
            "action_string": f"p{index}-action",
            "summary": "choice",
            "fallback_reason": None,
        }
        for index, model in enumerate(models)
    ]
    return cast(
        utils.EpisodeResult,
        {
            "episode_id": episode_id,
            "env": "OPENSPIEL",
            "game": "matching_pennies_3p",
            "arena_type": "PVP",
            "agent": models,
            "player_agents": models,
            "status": status,
            "winner": models[0],
            "num_steps": 2,
            "error": "No errors",
            "summary": ["chance", "choices"],
            "action": ["chance0", "joint action"],
            "start_time": now,
            "end_time": now,
            "seed": 123,
            "returns": [1.0, 0.0, -1.0][: len(models)],
            "terminal": status == "SUCCESS",
            "manifest": {"schema_version": 1, "seed": 123},
            "steps": [
                {
                    "index": 0,
                    "kind": "chance",
                    "players": [-1],
                    "actions": [0],
                    "action_strings": ["chance0"],
                    "decisions": [],
                    "rewards": [0.0] * len(models),
                    "returns": [0.0] * len(models),
                    "state_hash": "hash0",
                },
                {
                    "index": 1,
                    "kind": "simultaneous",
                    "players": list(range(len(models))),
                    "actions": [0] * len(models),
                    "action_strings": [d["action_string"] for d in decisions],
                    "decisions": decisions,
                    "rewards": [1.0, 0.0, -1.0][: len(models)],
                    "returns": [1.0, 0.0, -1.0][: len(models)],
                    "state_hash": "hash1",
                },
            ],
        },
    )


class OrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        @event.listens_for(self.engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.patches = [
            patch.object(module, "get_session", self.sessions)
            for module in (jobs, tasks, api_gateway)
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.engine.dispose)
        for item in self.patches:
            self.addCleanup(item.stop)

    def create_run(self, episodes: int = 1) -> tuple[str, str]:
        run_id, matchup_id = str(uuid.uuid4()), str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Run(
                    run_id=run_id,
                    run_type="PVP",
                    num_episodes=episodes,
                    status="PENDING",
                )
            )
            session.flush()
            session.add(
                Matchup(
                    matchup_id=matchup_id,
                    run_id=run_id,
                    models=["MOCK:a", "MOCK:b", "MOCK:c"],
                    num_episodes=episodes,
                    status="PENDING",
                )
            )
            session.commit()
        return run_id, matchup_id

    def test_multiplayer_trace_is_atomic_and_delivery_idempotent(self) -> None:
        run_id, matchup_id = self.create_run()
        result = _result(str(uuid.uuid4()))
        tasks._persist_result(result, run_id, matchup_id)
        tasks._persist_result(result, run_id, matchup_id)
        with self.sessions() as session:
            self.assertEqual(session.query(Episode).count(), 1)
            self.assertEqual(session.query(EpisodeTrace).count(), 1)
            self.assertEqual(session.query(EpisodeTransition).count(), 2)
            self.assertEqual(session.query(EpisodeStep).count(), 3)
            self.assertEqual(
                {row.player for row in session.query(EpisodeStep)}, {"0", "1", "2"}
            )
            self.assertEqual(session.get(Run, run_id).status, "COMPLETED")
            self.assertEqual(session.get(RunProgress, run_id).completed, 1)
            self.assertEqual(session.get(MatchupProgress, matchup_id).completed, 1)
            self.assertEqual(session.get(Matchup, matchup_id).status, "COMPLETED")
            payload = session.get(EpisodeTrace, result["episode_id"]).payload
            self.assertNotIn("steps", payload)
            self.assertEqual(payload["returns"], [1.0, 0.0, -1.0])

    def test_failed_last_episode_marks_run_failed_in_same_transaction(self) -> None:
        run_id, matchup_id = self.create_run(2)
        tasks._persist_result(_result(str(uuid.uuid4())), run_id, matchup_id)
        with self.sessions() as session:
            self.assertEqual(session.get(Run, run_id).status, "PENDING")
        tasks._persist_result(
            _result(str(uuid.uuid4()), status="TRUNCATED"), run_id, matchup_id
        )
        with self.sessions() as session:
            self.assertEqual(session.get(Run, run_id).status, "FAILED")
            self.assertEqual(session.get(RunProgress, run_id).completed, 2)
            self.assertEqual(session.get(RunProgress, run_id).failed, 1)
            self.assertEqual(session.get(Matchup, matchup_id).status, "FAILED")

    def test_model_initialization_failure_is_persisted(self) -> None:
        run_id, matchup_id = self.create_run()
        identity = str(uuid.uuid4())
        with patch.object(
            tasks, "_get_model", side_effect=ValueError("no API credentials")
        ):
            response = tasks.run_episode_pvp(
                matchup_id, run_id, 0, None, episode_id=identity, seed=99
            )
        self.assertEqual(response["result"]["status"], "FAILED")
        with self.sessions() as session:
            row = session.get(Episode, identity)
            self.assertEqual(row.num_steps, 0)
            self.assertEqual(session.get(EpisodeTrace, identity).payload["seed"], 99)
            self.assertEqual(session.get(Run, run_id).status, "FAILED")

    def test_seat_rotation_and_retry_reuse(self) -> None:
        run_id, matchup_id = self.create_run(2)
        calls: list[list[str]] = []

        def run(options: utils.HarnessOptions) -> utils.EpisodeResult:
            names = [str(model) for model in options["models"]]
            calls.append(names)
            return _result(options.get("episode_id", ""), models=names)

        identity = str(uuid.uuid4())
        with (
            patch.object(tasks, "_get_model", side_effect=lambda name: name),
            patch.object(tasks.harness, "harness", side_effect=run),
        ):
            tasks.run_episode_pvp(matchup_id, run_id, 1, None, episode_id=identity)
            reused = tasks.run_episode_pvp(
                matchup_id, run_id, 1, None, episode_id=identity
            )
        self.assertEqual(calls, [["MOCK:b", "MOCK:c", "MOCK:a"]])
        self.assertTrue(reused["reused"])

    def test_round_robin_n_player_scheduling_and_retry(self) -> None:
        batches: list[list[Any]] = []
        run_id = str(uuid.uuid4())
        kwargs = {
            "run_type": "PVP_RR",
            "num_episodes": 2,
            "num_steps": 20,
            "model_names": ["MOCK:a", "MOCK:b", "MOCK:c", "MOCK:d"],
            "env": "kuhn_poker(players=3)",
            "seed": 71,
            "run_id": run_id,
        }
        with patch.object(
            jobs,
            "_enqueue_batch",
            side_effect=lambda items: batches.append(list(items)),
        ):
            first = jobs.enqueue_run(**kwargs)
            second = jobs.enqueue_run(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 8)
        enqueued = batches[0]
        self.assertEqual(len({item.kwargs["seed"] for item in enqueued}), 8)
        self.assertTrue(all(item.depends_on is None for item in enqueued))
        self.assertTrue(
            all(item.kwargs["game"] == "kuhn_poker(players=3)" for item in enqueued)
        )
        with self.sessions() as session:
            self.assertEqual(session.query(Run).count(), 1)
            self.assertEqual(session.query(Matchup).count(), 4)
            self.assertEqual(session.get(Run, run_id).num_episodes, 8)
            self.assertEqual(session.get(RunConfiguration, run_id).payload["seed"], 71)

    def test_admission_rejects_before_database_and_queue(self) -> None:
        with (
            patch.object(jobs, "get_session") as database,
            patch.object(jobs, "_enqueue_batch") as queue,
        ):
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                jobs.enqueue_run(
                    "PVP", 2, None, ["MOCK:a", "MOCK:b"], env="kuhn_poker(players=3)"
                )
            database.assert_not_called()
            queue.assert_not_called()
        with self.assertRaises(ValidationError):
            api_gateway.RunCreateRequest(
                run_type="PVP",
                model_names=["MOCK:a", "MOCK:b"],
                env="kuhn_poker(players=3)",
            )

    def test_actual_mock_game_runs_and_persists_complete_trace(self) -> None:
        run_id, matchup_id = self.create_run()
        response = tasks.run_episode_pvp(
            matchup_id, run_id, 0, 20, game="matching_pennies_3p", seed=211
        )
        self.assertEqual(response["result"]["status"], "SUCCESS")
        self.assertEqual(len(response["result"]["returns"]), 3)
        with self.sessions() as session:
            self.assertEqual(session.query(EpisodeTransition).count(), 1)
            self.assertEqual(session.query(EpisodeStep).count(), 3)
            payload = session.query(EpisodeTransition).one().payload
            self.assertEqual(payload["kind"], "simultaneous")
            self.assertTrue(
                all(
                    decision["prompt"] and decision["response"]
                    for decision in payload["decisions"]
                )
            )
            self.assertEqual(session.get(Run, run_id).status, "COMPLETED")

    def test_submission_batches_are_bounded_and_retries_skip_existing_jobs(
        self,
    ) -> None:
        sizes: list[int] = []
        with patch.object(
            jobs, "_enqueue_batch", side_effect=lambda items: sizes.append(len(items))
        ):
            jobs.enqueue_run("PVP", 104, 5, ["MOCK:a", "MOCK:b"], env="tic_tac_toe")
        self.assertEqual(sizes, [100, 4])
        queue = MagicMock()
        queue.connection.pipeline.return_value.__enter__.return_value.execute.return_value = [
            b"finished",
            None,
        ]
        items = [SimpleNamespace(job_id=str(uuid.uuid4())) for _ in range(2)]
        with patch.object(jobs, "queue", queue):
            jobs._enqueue_batch(items)
        queue.enqueue_many.assert_called_once_with([items[1]])

    def test_api_rejects_nested_server_file_parameters_before_inspection(self) -> None:
        specs = [
            "nfg_game(filename=/tmp/server-secret)",
            "turn_based_simultaneous_game(game=nfg_game(filename=/tmp/server-secret))",
            "bargaining(instances_file=/tmp/server-secret)",
            "colored_trails(boards_file=/tmp/server-secret)",
        ]
        with (
            patch.object(jobs, "inspect_game") as inspect,
            patch.object(jobs, "get_session") as database,
        ):
            for spec in specs:
                with (
                    self.subTest(spec=spec),
                    self.assertRaisesRegex(ValueError, "File/path"),
                ):
                    jobs.enqueue_run("PVP", 1, 5, ["MOCK:a", "MOCK:b"], env=spec)
            inspect.assert_not_called()
            database.assert_not_called()
        client = TestClient(api_gateway.app)
        with (
            patch.object(api_gateway, "_enforce_rate_limit"),
            patch.object(api_gateway, "API_KEYS", {"local-dev-key"}),
            patch("game_arena.openspiel.catalog.inspect_game") as inspect,
        ):
            response = client.get(
                "/v1/games/inspect",
                params={"game": specs[0]},
                headers={"Authorization": "Bearer local-dev-key"},
            )
            self.assertEqual(response.status_code, 400)
            inspect.assert_not_called()

    def test_mean_field_population_is_validated_before_queue_submission(self) -> None:
        with patch.object(jobs, "get_session") as database:
            with self.assertRaisesRegex(
                ValueError, "mean_field_population must be below"
            ):
                jobs.enqueue_run(
                    "PVP",
                    1,
                    5,
                    ["MOCK"],
                    env="mfg_crowd_modelling",
                    mean_field_distribution="uniform",
                    mean_field_population=1,
                )
            database.assert_not_called()

    def test_elo_uses_returns_and_actual_rotated_seats(self) -> None:
        run_id, matchup_id = self.create_run(2)
        first = _result(str(uuid.uuid4()), models=["ModelA", "ModelB"])
        second = _result(str(uuid.uuid4()), models=["ModelB", "ModelA"])
        for result, returns in ((first, [1.0, -1.0]), (second, [-1.0, 1.0])):
            result["game"] = "connect_four"
            result["returns"] = returns
            result["winner"] = "legacy winner unavailable"
            result.setdefault("manifest", {})["game"] = "connect_four"
            tasks._persist_result(result, run_id, matchup_id)
        with self.sessions() as session:
            episodes = session.query(Episode).filter_by(run_id=run_id).all()
            processed, skipped = ratings._process_episodes(
                session,
                episodes,
                config=ratings.EloConfig(1500, 32, 16, 10),
                skip_existing_events=True,
            )
            self.assertEqual((processed, skipped), (2, 0))
            model = session.query(ModelRating).filter_by(model_name="ModelA").one()
            self.assertEqual(
                (model.wins, model.games_as_x, model.games_as_o), (2, 1, 1)
            )
            self.assertGreater(model.rating, 1500)
            self.assertEqual(
                {event.winner_model for event in session.query(RatingEvent)}, {"ModelA"}
            )

    def test_elo_game_keys_separate_parameter_variants(self) -> None:
        default, _ = ratings._rating_game_identity("connect_four")
        small, _ = ratings._rating_game_identity("hex(board_size=3)")
        large, _ = ratings._rating_game_identity("hex(board_size=5)")
        self.assertEqual(default, "connect_four")
        self.assertNotEqual(small, large)
        self.assertLessEqual(len(small), 50)
        self.assertIn("@", small)

    def test_runtime_mismatch_is_persisted_before_model_creation(self) -> None:
        run_id, matchup_id = self.create_run()
        identity = str(uuid.uuid4())
        with (
            patch.object(tasks, "_get_model") as model,
            self.assertLogs("orchestrator.tasks", level="ERROR"),
        ):
            response = tasks.run_episode_pvp(
                matchup_id,
                run_id,
                0,
                5,
                episode_id=identity,
                metadata={"expected_runtime": {"source_sha256": "different-build"}},
            )
        model.assert_not_called()
        self.assertEqual(response["result"]["status"], "FAILED")
        self.assertIn("runtime does not match", response["result"]["error"])
        with self.sessions() as session:
            trace = session.get(EpisodeTrace, identity)
            self.assertEqual(
                trace.payload["manifest"]["metadata"]["expected_runtime"][
                    "source_sha256"
                ],
                "different-build",
            )

    def test_partial_dispatch_is_recovered_from_durable_configuration(self) -> None:
        run_id = str(uuid.uuid4())
        with patch.object(
            jobs,
            "_enqueue_batch",
            side_effect=[None, ConnectionError("injected Redis outage")],
        ):
            with self.assertRaises(ConnectionError):
                jobs.enqueue_run(
                    "PVP",
                    104,
                    5,
                    ["MOCK:a", "MOCK:b"],
                    env="tic_tac_toe",
                    run_id=run_id,
                    seed=17,
                )
        with self.sessions() as session:
            dispatch = session.get(RunDispatch, run_id)
            self.assertFalse(dispatch.complete)
            self.assertIn("injected Redis outage", dispatch.last_error)
            self.assertEqual(session.get(RunConfiguration, run_id).payload["seed"], 17)
        batches: list[list[Any]] = []
        with (
            patch.object(
                jobs,
                "_enqueue_batch",
                side_effect=lambda items: batches.append(list(items)),
            ),
            patch.object(
                jobs, "runtime_manifest", return_value={"source_sha256": "new-build"}
            ),
        ):
            recovered = jobs.dispatch_pending_runs()
        self.assertEqual(recovered, {"attempted": 1, "completed": 1, "failed": 0})
        self.assertEqual([len(batch) for batch in batches], [100, 4])
        self.assertTrue(
            all(item.kwargs["run_id"] == run_id for batch in batches for item in batch)
        )
        with self.sessions() as session:
            self.assertTrue(session.get(RunDispatch, run_id).complete)
            self.assertIsNone(session.get(RunDispatch, run_id).last_error)
            self.assertEqual(session.query(Run).count(), 1)
            pinned_runtime = session.get(RunConfiguration, run_id).payload["runtime"]
            self.assertTrue(
                all(
                    item.kwargs["metadata"]["expected_runtime"] == pinned_runtime
                    for batch in batches
                    for item in batch
                )
            )
        self.assertEqual(jobs.dispatch_pending_runs()["attempted"], 0)

    def test_pending_dispatch_requeues_only_unsaved_failed_jobs(self) -> None:
        run_id, matchup_id = self.create_run()
        saved_id = str(uuid.uuid4())
        tasks._persist_result(_result(saved_id), run_id, matchup_id)
        failed_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())
        items = [
            SimpleNamespace(job_id=identity)
            for identity in (saved_id, failed_id, new_id)
        ]
        queue = MagicMock()
        queue.connection.pipeline.return_value.__enter__.return_value.execute.return_value = [
            b"failed",
            b"failed",
            None,
        ]
        with (
            patch.object(jobs, "queue", queue),
            patch.object(jobs, "FailedJobRegistry") as registry,
        ):
            jobs._enqueue_batch(items)
        registry.return_value.requeue.assert_called_once_with(failed_id)
        queue.enqueue_many.assert_called_once_with([items[2]])

    def test_submission_failure_response_exposes_pollable_run_id(self) -> None:
        client = TestClient(api_gateway.app)
        with (
            patch.object(api_gateway, "_enforce_rate_limit"),
            patch.object(api_gateway, "API_KEYS", {"local-dev-key"}),
            patch.object(
                api_gateway,
                "_enqueue_with_backoff",
                side_effect=ConnectionError("injected"),
            ) as enqueue,
            self.assertLogs("orchestrator.api_gateway", level="ERROR"),
        ):
            response = client.post(
                "/v1/runs",
                json={
                    "run_type": "PVP",
                    "env": "tic_tac_toe",
                    "model_names": ["MOCK:a", "MOCK:b"],
                },
                headers={"Authorization": "Bearer local-dev-key"},
            )
        self.assertEqual(response.status_code, 503)
        error = response.json()["error"]
        request = enqueue.call_args.args[0]
        self.assertEqual(error["run_id"], request._submission_id)
        self.assertEqual(error["status_url"], f"/v1/runs/{request._submission_id}")

    def test_fallback_assisted_success_is_excluded_from_clean_metrics_and_elo(
        self,
    ) -> None:
        class BrokenModel:
            def __init__(self, name: str):
                self.name = name

            def __str__(self) -> str:
                return self.name

            def generate_with_text_input(
                self,
                prompt: str,
                system_instruction: str | None = None,
                mock: bool = False,
            ):
                raise RuntimeError("injected model failure")

        run_id, matchup_id = self.create_run()
        with self.sessions() as session:
            session.get(Matchup, matchup_id).models = ["MOCK:a", "MOCK:b"]
            session.commit()
        with (
            patch.object(tasks, "_get_model", side_effect=BrokenModel),
            self.assertLogs("game_arena.openspiel.harness", level="WARNING"),
        ):
            response = tasks.run_episode_pvp(
                matchup_id, run_id, 0, 10, game="matrix_rps", seed=77
            )
        result = response["result"]
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result.get("degraded"))
        self.assertEqual(result.get("fallback_count"), 2)
        with self.sessions() as session:
            episodes = session.query(Episode).filter_by(run_id=run_id).all()
            self.assertEqual(
                ratings._process_episodes(
                    session,
                    episodes,
                    config=ratings.EloConfig(1500, 32, 16, 10),
                    skip_existing_events=True,
                ),
                (0, 1),
            )
            self.assertEqual(session.query(ModelRating).count(), 0)
        with patch.object(api_gateway, "_enforce_rate_limit"):
            metrics = api_gateway.get_run_metrics(run_id, api_key="local-dev-key")
        self.assertEqual(
            metrics["episode_quality_counts"], {"fallback_assisted_terminal": 1}
        )
        for index, row in enumerate(metrics["player_metrics"]):
            self.assertIsNone(row["mean_return"])
            self.assertEqual(row["clean_terminal_episodes"], 0)
            self.assertEqual(row["fallback_assisted_terminal_episodes"], 1)
            self.assertEqual(
                row["fallback_assisted_mean_return"], result["returns"][index]
            )

    def test_trace_pagination_and_metrics_use_real_player_ids(self) -> None:
        run_id, matchup_id = self.create_run()
        result = _result(str(uuid.uuid4()))
        tasks._persist_result(result, run_id, matchup_id)
        client = TestClient(api_gateway.app)
        with (
            patch.object(api_gateway, "_enforce_rate_limit"),
            patch.object(api_gateway, "API_KEYS", {"local-dev-key"}),
        ):
            trace = client.get(
                f"/v1/episodes/{result['episode_id']}/trace?limit=1&offset=1",
                headers={"Authorization": "Bearer local-dev-key"},
            )
            self.assertEqual(trace.status_code, 200)
            self.assertEqual(trace.json()["total"], 2)
            self.assertEqual(trace.json()["transitions"][0]["players"], [0, 1, 2])
            metrics = client.get(
                f"/v1/runs/{run_id}/metrics",
                headers={"Authorization": "Bearer local-dev-key"},
            )
            self.assertEqual(metrics.status_code, 200)
            self.assertEqual(metrics.json()["decisions"], 3)
            self.assertEqual(
                [row["mean_return"] for row in metrics.json()["player_metrics"]],
                [1.0, 0.0, -1.0],
            )


if __name__ == "__main__":
    unittest.main()
