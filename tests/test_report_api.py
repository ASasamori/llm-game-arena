"""Run-report API integration using real mock episodes and isolated SQLite."""

from __future__ import annotations

import os
import unittest
from typing import Any
from urllib.parse import parse_qs, urlsplit
import uuid
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import GenerateResponse, LLMModel, MockModel
from game_arena.openspiel.utils import EpisodeResult, HarnessOptions
from orchestrator import api_gateway, tasks
from orchestrator.models import (
    Base,
    EpisodeTrace,
    EpisodeTransition,
    Run,
    RunConfiguration,
)


class _RecordedModel(MockModel):
    """Supply recognizable recorded text without constructing an API client."""

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        response = super().generate_with_text_input(prompt, system_instruction, mock)
        answer = response.main_response.rsplit("Final Answer:", 1)[-1].strip()
        response.main_response = (
            f"Summary: Recorded decision from {self._model_name}.\n"
            f"Final Answer: {answer}"
        )
        return response


class ReportApiTests(unittest.TestCase):
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
            patch.object(api_gateway, "get_session", self.sessions),
            patch.object(tasks, "get_session", self.sessions),
            patch.object(api_gateway, "_enforce_rate_limit"),
            patch.object(api_gateway, "API_KEYS", {"report-test-key"}),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.engine.dispose)
        for item in self.patches:
            self.addCleanup(item.stop)
        self.client = TestClient(api_gateway.app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer report-test-key"}

    def create_run(self, episodes: int = 1) -> str:
        identity = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Run(
                    run_id=identity,
                    run_type="PVP",
                    num_episodes=episodes,
                    status="PENDING",
                )
            )
            session.flush()
            session.add(
                RunConfiguration(
                    run_id=identity,
                    payload={
                        "game": "tic_tac_toe",
                        "run_type": "PVP",
                        "model_names": ["MOCK:alpha", "MOCK:beta"],
                        "seed": 71,
                        "rotate_seats": True,
                        "unrecognized_private_configuration": "CONFIGURATION_SECRET",
                    },
                )
            )
            session.commit()
        return identity

    def persist_episode(self, run_id: str, label: str = "alpha") -> EpisodeResult:
        models: list[LLMModel] = [_RecordedModel(label), _RecordedModel("beta")]
        result = harness(
            HarnessOptions(
                type="PVP",
                models=models,
                num_steps=20,
                input=None,
                game="tic_tac_toe",
                seed=71,
                failure_policy="raise",
            )
        )
        tasks._persist_result(result, run_id, None)
        return result

    def report(self, run_id: str, **params: Any):
        return self.client.get(
            f"/v1/runs/{run_id}/report", headers=self.headers, params=params
        )

    def test_public_shell_does_not_expose_run_data_or_key(self) -> None:
        run_id = self.create_run()
        self.persist_episode(run_id, label="PRIVATE_MODEL_EVIDENCE")
        with patch.object(
            api_gateway,
            "get_session",
            side_effect=AssertionError("HTML queried run data"),
        ):
            for path in ("/reports", f"/runs/{run_id}"):
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(
                        response.headers["content-type"].startswith("text/html")
                    )
                    self.assertNotIn("PRIVATE_MODEL_EVIDENCE", response.text)
                    self.assertNotIn("report-test-key", response.text)
                    self.assertNotIn("CONFIGURATION_SECRET", response.text)

    def test_report_data_and_text_require_authentication(self) -> None:
        run_id = self.create_run()
        for params in ({}, {"format": "text"}):
            with self.subTest(params=params):
                response = self.client.get(f"/v1/runs/{run_id}/report", params=params)
                self.assertEqual(response.status_code, 401)

    def test_actual_mock_trace_has_json_and_ascii_reports(self) -> None:
        run_id = self.create_run()
        episode = self.persist_episode(run_id)
        response = self.report(run_id)
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["run"]["run_id"], run_id)
        self.assertEqual(report["episodes"]["total"], 1)
        self.assertEqual(
            report["selected_episode"]["episode_id"], episode["episode_id"]
        )
        self.assertTrue(report["selected_episode"]["mock"])
        self.assertIs(report["selected_episode"]["terminal"], True)
        self.assertIs(report["selected_episode"]["degraded"], False)
        self.assertIs(report["run"]["configuration"]["rotate_seats"], True)
        self.assertEqual(report["transitions"]["total"], episode["num_steps"])
        self.assertNotIn("CONFIGURATION_SECRET", response.text)
        first = report["transitions"]["items"][0]
        self.assertEqual(first["index"], 0)
        self.assertIn("xox", report["transitions"]["items"][-1]["state_after"].lower())
        self.assertIn(
            "Recorded decision from alpha.", first["decisions"][0]["response"]
        )

        text = self.report(run_id, format="text", width=100)
        self.assertEqual(text.status_code, 200, text.text)
        self.assertTrue(text.headers["content-type"].startswith("text/plain"))
        self.assertTrue(text.text.isascii())
        self.assertIn(run_id, text.text)
        self.assertIn(episode["episode_id"], text.text)
        self.assertIn("SUCCESS", text.text)
        self.assertIn(episode["winner"], text.text)
        self.assertIn("Recorded decision from alpha.", text.text)
        self.assertIn("Final Answer: 0", text.text)
        self.assertIn("xox", text.text.lower())
        self.assertIn("oxo", text.text.lower())
        self.assertNotIn("CONFIGURATION_SECRET", text.text)
        self.assertNotIn("Play this OpenSpiel game", text.text)
        with_prompts = self.report(run_id, format="text", include_prompt="true")
        self.assertEqual(with_prompts.status_code, 200)
        self.assertIn("Play this OpenSpiel game", with_prompts.text)

    def test_episode_and_transition_pagination_are_independent(self) -> None:
        run_id = self.create_run(2)
        self.persist_episode(run_id, "first")
        self.persist_episode(run_id, "second")
        full = self.report(run_id, episode_limit=100).json()
        ids = [item["episode_id"] for item in full["episodes"]["items"]]
        page = self.report(run_id, episode_limit=1, episode_offset=1).json()
        self.assertEqual(page["episodes"]["total"], 2)
        self.assertEqual(page["episodes"]["limit"], 1)
        self.assertEqual(page["episodes"]["offset"], 1)
        self.assertEqual(
            [item["episode_id"] for item in page["episodes"]["items"]], ids[1:2]
        )
        transitions = self.report(
            run_id, episode_id=ids[0], step_limit=2, step_offset=1
        ).json()
        self.assertEqual(transitions["selected_episode"]["episode_id"], ids[0])
        self.assertEqual(transitions["transitions"]["offset"], 1)
        self.assertEqual(transitions["transitions"]["limit"], 2)
        self.assertEqual(
            [item["index"] for item in transitions["transitions"]["items"]], [1, 2]
        )
        empty = self.report(run_id, episode_id=ids[0], step_offset=10000).json()
        self.assertEqual(empty["transitions"]["items"], [])
        self.assertGreater(empty["transitions"]["total"], 0)

    def test_unknown_run_and_cross_run_episode_do_not_expose_steps(self) -> None:
        first_run = self.create_run()
        self.persist_episode(first_run)
        other_run = self.create_run()
        other_episode = self.persist_episode(other_run, "CROSS_RUN_SECRET")
        unknown = self.report(str(uuid.uuid4()))
        self.assertEqual(unknown.status_code, 404)
        wrong_run = self.report(first_run, episode_id=other_episode["episode_id"])
        self.assertEqual(wrong_run.status_code, 404)
        self.assertNotIn("CROSS_RUN_SECRET", wrong_run.text)

    def test_invalid_limits_formats_and_identifiers_return_client_errors(self) -> None:
        run_id = self.create_run()
        for params in (
            {"episode_limit": 0},
            {"episode_limit": 101},
            {"episode_offset": -1},
            {"step_limit": 0},
            {"step_limit": 201},
            {"step_offset": -1},
            {"format": "html"},
            {"format": "text", "width": 59},
            {"format": "text", "width": 161},
            {"episode_id": "not-a-uuid"},
        ):
            with self.subTest(params=params):
                response = self.report(run_id, **params)
                self.assertIn(response.status_code, {400, 422}, response.text)
        self.assertEqual(self.report("not-a-uuid").status_code, 400)

    def test_failed_and_legacy_episodes_have_no_invented_transitions(self) -> None:
        failed_run = self.create_run()
        failed = tasks._failed_episode_result(
            run_type="PVP",
            game="tic_tac_toe",
            models=["MOCK:alpha", "MOCK:beta"],
            error="Injected mock failure before any move.",
            episode_id=str(uuid.uuid4()),
            seed=71,
            metadata={},
        )
        tasks._persist_result(failed, failed_run, None)
        response = self.report(failed_run)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["selected_episode"]["status"], "FAILED")
        self.assertEqual(response.json()["transitions"]["items"], [])
        text = self.report(failed_run, format="text")
        self.assertIn("Injected mock failure before any move.", text.text)

        legacy_run = self.create_run()
        legacy = self.persist_episode(legacy_run)
        with self.sessions() as session:
            session.query(EpisodeTransition).filter_by(
                episode_id=legacy["episode_id"]
            ).delete()
            session.query(EpisodeTrace).filter_by(
                episode_id=legacy["episode_id"]
            ).delete()
            session.commit()
        response = self.report(legacy_run)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["transitions"]["items"], [])
        notices = str(response.json()["notices"]).lower()
        self.assertTrue(
            any(
                word in notices
                for word in ("legacy", "missing", "unavailable", "no saved")
            ),
            notices,
        )

    def test_creation_and_status_expose_absolute_report_links_without_keys(
        self,
    ) -> None:
        run_id = self.create_run()
        with (
            patch.object(
                api_gateway,
                "_enqueue_with_backoff",
                return_value=(run_id, ["mock-job"]),
            ),
            patch.object(
                api_gateway,
                "_trigger_worker_if_configured",
                return_value=(None, None, None),
            ),
        ):
            created = self.client.post(
                "/v1/runs",
                headers=self.headers,
                json={
                    "run_type": "PVP",
                    "env": "tic_tac_toe",
                    "model_names": ["MOCK:a", "MOCK:b"],
                    "num_episodes": 1,
                    "num_steps": 20,
                },
            )
        self.assertEqual(created.status_code, 202, created.text)
        status = self.client.get(f"/v1/runs/{run_id}", headers=self.headers)
        self.assertEqual(status.status_code, 200, status.text)
        for payload in (created.json(), status.json()):
            for field, path in (
                ("report_url", f"/runs/{run_id}"),
                ("text_report_url", f"/v1/runs/{run_id}/report"),
                ("report_data_url", f"/v1/runs/{run_id}/report"),
            ):
                with self.subTest(field=field):
                    url = urlsplit(payload[field])
                    self.assertEqual(url.scheme, "http")
                    self.assertEqual(url.netloc, "testserver")
                    self.assertEqual(url.path, path)
                    self.assertNotIn("report-test-key", payload[field])
            self.assertEqual(
                parse_qs(urlsplit(payload["text_report_url"]).query).get("format"),
                ["text"],
            )


if __name__ == "__main__":
    unittest.main()
