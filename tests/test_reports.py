"""Report pagination and truthful presentation from real, local mock traces."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
import unittest
from typing import Any
import uuid
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from game_arena.openspiel import harness, utils
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.harness import EpisodeStopped
from orchestrator import tasks
from orchestrator.models import Base, Episode, EpisodeTransition, Run, RunConfiguration
from orchestrator.reports import (
    ReportNotFound,
    ReportRequestError,
    build_run_report,
    render_text_report,
    terminal_safe,
)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.task_session = patch.object(tasks, "get_session", self.sessions)
        self.task_session.start()
        self.addCleanup(self.task_session.stop)
        self.addCleanup(self.engine.dispose)

    def create_run(self, episodes: int = 1, configuration: dict | None = None) -> str:
        identity = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Run(
                    run_id=identity,
                    status="PENDING",
                    run_type="PVP",
                    num_episodes=episodes,
                    created_at=datetime(2026, 11, 4, tzinfo=UTC),
                )
            )
            session.flush()
            if configuration is not None:
                session.add(RunConfiguration(run_id=identity, payload=configuration))
            session.commit()
        return identity

    def mock_result(
        self,
        *,
        game: str = "tic_tac_toe",
        steps: int | None = 3,
        players: int = 2,
        start: int = 0,
    ) -> utils.EpisodeResult:
        options = utils.HarnessOptions(
            type="PVP",
            models=[MockModel(name=f"player-{player}") for player in range(players)],
            num_steps=steps,
            input=None,
            game=game,
            seed=19,
        )
        if game.startswith("mfg_"):
            options["mean_field_distribution"] = "uniform"
        result = harness.harness(options)
        result["start_time"] = datetime(2026, 11, 4, tzinfo=UTC) + timedelta(
            seconds=start
        )
        result["end_time"] = result["start_time"] + timedelta(seconds=1)
        return result

    def save(self, run_id: str, result: utils.EpisodeResult) -> None:
        tasks._persist_result(result, run_id, None)

    def test_actual_mock_joint_decisions_and_provenance_are_preserved(self) -> None:
        run_id = self.create_run()
        record = self.mock_result(game="matrix_rps", steps=10)
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id)
        selected = report["selected_episode"]
        self.assertEqual(selected["episode_id"], record["episode_id"])
        self.assertEqual(selected["returns"], record.get("returns"))
        self.assertTrue(selected["terminal"])
        self.assertTrue(selected["mock"])
        self.assertEqual(
            selected["provenance"]["initial_state_hash"],
            record.get("manifest", {})["initial_state_hash"],
        )
        self.assertEqual(
            report["transitions"]["items"], json.loads(json.dumps(record.get("steps")))
        )
        self.assertEqual(report["run"]["status_counts"], {"SUCCESS": 1})
        self.assertTrue(report["run"]["created_at"].endswith("+00:00"))
        text = render_text_report(report)
        self.assertIn("simultaneous", text)
        self.assertIn("MOCK OUTPUT", text)
        self.assertIn("Recorded mock policy summary", text)
        self.assertIn("Deterministic offline legal action.", text)
        self.assertIn("Raw recorded response", text)
        self.assertIn("Chosen action ID", text)
        self.assertIn("Recorded utilities / returns", text)
        self.assertIn("Recorded player ID: 0", text)
        self.assertIn("Recorded player ID: 1", text)

    def test_native_team_winners_preserve_explicit_seats_and_utilities(self) -> None:
        run_id = self.create_run()
        record = self.mock_result(game="avalon(players=5)", steps=2000, players=5)
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id)
        episode = report["selected_episode"]
        values = record.get("returns", [])
        self.assertEqual(len(values), 5)
        self.assertEqual(episode["provenance"]["environment_backend"], "native")
        self.assertEqual(
            episode["winner_players"],
            [p for p, value in enumerate(values) if value > 0],
        )
        self.assertGreater(len(episode["winner_players"]), 1)
        self.assertEqual(episode["returns"], values)
        self.assertIn("Winner Players", render_text_report(report))

    def test_pages_are_stable_and_do_not_load_legacy_action_arrays(self) -> None:
        run_id = self.create_run(3)
        records = [self.mock_result(start=value) for value in (2, 0, 1)]
        for record in records:
            self.save(run_id, record)
        queries: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith("SELECT"):
                queries.append(statement)

        event.listen(self.engine, "before_cursor_execute", capture)
        self.addCleanup(event.remove, self.engine, "before_cursor_execute", capture)
        with self.sessions() as session:
            report = build_run_report(
                session,
                run_id,
                episode_limit=1,
                episode_offset=1,
                step_limit=1,
                step_offset=1,
            )
        self.assertEqual(report["episodes"]["total"], 3)
        self.assertEqual(
            [episode["episode_id"] for episode in report["episodes"]["items"]],
            [records[2]["episode_id"]],
        )
        self.assertEqual(
            report["selected_episode"]["episode_id"], records[2]["episode_id"]
        )
        self.assertEqual(report["transitions"]["total"], 3)
        self.assertEqual(report["transitions"]["items"][0]["index"], 1)
        self.assertEqual(len(report["transitions"]["items"]), 1)
        self.assertFalse(
            any(
                "episodes.action" in query or "episodes.summary" in query
                for query in queries
            )
        )
        text = render_text_report(report)
        self.assertIn("Next episodes: episode_offset=2", text)
        self.assertIn("step_offset=2", text)
        self.assertNotIn(records[0]["episode_id"], text)

    def test_evaluator_board_is_separate_from_the_recorded_player_view(self) -> None:
        run_id = self.create_run()
        record = self.mock_result(steps=100)
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id)
        events = report["transitions"]["items"]
        self.assertEqual(events[0]["state_before"], "...\n...\n...")
        self.assertEqual(events[0]["decisions"][0]["observation"], "")
        text = render_text_report(report)
        self.assertIn("Evaluator state before", text)
        self.assertIn("Evaluator state after", text)
        self.assertIn("xox", text.lower())
        self.assertIn("Observation (recorded before action)", text)

    def test_navigation_index_is_independent_of_page_and_omits_private_evidence(
        self,
    ) -> None:
        run_id = self.create_run()
        record = self.mock_result(game="avalon", steps=100, players=5)
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id, step_offset=12, step_limit=1)
        index = report["transition_index"]
        self.assertEqual(len(index["items"]), len(record.get("steps", [])))
        self.assertEqual(index["total"], report["transitions"]["total"])
        self.assertFalse(index["truncated"])
        self.assertEqual(report["transitions"]["items"][0]["index"], 12)
        self.assertEqual(index["items"][0]["index"], 0)
        self.assertEqual(index["items"][0]["label"], "Chance event")
        phases = {item["phase"] for item in index["items"]}
        self.assertIn("proposal", phases)
        self.assertIn("vote", phases)
        self.assertNotIn("reject_team", json.dumps(index))
        self.assertNotIn("deal_role", json.dumps(index))
        self.assertNotIn("observation", json.dumps(index))
        self.assertNotIn("prompt", json.dumps(index))
        self.assertNotIn("response", json.dumps(index))

    def test_openspiel_review_and_decision_mechanisms_survive_report_pages(
        self,
    ) -> None:
        for game, players in (
            ("connect_four", 2),
            ("kuhn_poker", 2),
            ("matching_pennies_3p", 3),
            ("mfg_crowd_modelling(horizon=2)", 1),
        ):
            with self.subTest(game=game):
                run_id = self.create_run()
                record = self.mock_result(game=game, players=players, steps=10)
                self.save(run_id, record)
                with self.sessions() as session:
                    report = build_run_report(session, run_id, step_limit=1)
                index = report["transition_index"]["items"]
                self.assertEqual(
                    index[0]["node_kind"], record.get("steps", [])[0]["kind"]
                )
                event = report["transitions"]["items"][0]
                self.assertEqual(
                    event, json.loads(json.dumps(record.get("steps", [])[0]))
                )
                self.assertIn("review_state_before", event)
                self.assertIn("review_state_after", event)
                for decision in event["decisions"]:
                    self.assertEqual(decision["selection"]["kind"], "mock")
                text = render_text_report(report)
                self.assertIn("Game state before", text)
                self.assertFalse(
                    any(
                        "snapshots were not recorded" in notice
                        for notice in report["notices"]
                    )
                )

    def test_large_index_stays_bounded_but_numeric_jump_reaches_final_step(
        self,
    ) -> None:
        run_id = self.create_run()
        record = self.mock_result(steps=1)
        self.save(run_id, record)
        episode_id = record["episode_id"]
        with self.sessions() as session:
            session.add_all(
                EpisodeTransition(
                    episode_id=episode_id,
                    step_index=index,
                    payload={
                        "index": index,
                        "kind": "decision",
                        "action_strings": ["move"],
                    },
                )
                for index in range(1, 5002)
            )
            session.commit()
            report = build_run_report(session, run_id, step_offset=5001, step_limit=1)
        self.assertEqual(len(report["transition_index"]["items"]), 5000)
        self.assertTrue(report["transition_index"]["truncated"])
        self.assertEqual(report["transition_index"]["total"], 5002)
        self.assertEqual(report["transitions"]["items"][0]["index"], 5001)

    def test_unapplied_joint_attempts_are_preserved_and_paginated_separately(
        self,
    ) -> None:
        class StopModel(MockModel):
            def generate_with_text_input(
                self, prompt, system_instruction=None, mock=False
            ):
                raise EpisodeStopped("provider_error", "Offline test stop")

        run_id = self.create_run()
        options: Any = {
            "type": "PVP",
            "models": [MockModel("first"), StopModel("second")],
            "game": "matrix_rps",
            "num_steps": 1,
            "input": None,
            "seed": 31,
            "failure_policy": "raise",
            "permitted_stop_reasons": ["provider_error"],
        }
        record = harness.harness(options)
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id, pending_limit=1)
            next_page = build_run_report(
                session, run_id, pending_limit=1, pending_offset=1
            )
        self.assertEqual(report["transitions"]["total"], 0)
        pending = report["pending_transition"]
        self.assertIs(pending["applied"], False)
        self.assertEqual(pending["decisions"]["total"], 1)
        stored_pending = record.get("pending_transition")
        assert stored_pending is not None
        self.assertEqual(
            pending["decisions"]["items"][0]["response"],
            stored_pending["decisions"][0]["response"],
        )
        self.assertEqual(pending["failed_attempt"]["player"], 1)
        self.assertEqual(next_page["pending_transition"]["decisions"]["items"], [])
        text = render_text_report(report)
        self.assertIn("INTERRUPTED ATTEMPTS - NOT APPLIED", text)
        self.assertIn("Deterministic offline legal action.", text)
        self.assertIn("Offline test stop", text)
        self.assertNotIn("Play this OpenSpiel game", text)
        self.assertIn(
            "Play this OpenSpiel game", render_text_report(report, include_prompt=True)
        )

    def test_catan_index_includes_public_turn_numbers(self) -> None:
        run_id = self.create_run()
        record = self.mock_result(
            game="catan(players=3,max_turns=3)", players=3, steps=200
        )
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id, step_offset=35)
        items = report["transition_index"]["items"]
        self.assertEqual({item["turn"] for item in items}, {0, 1, 2, 3})
        self.assertIn("Public board before:", render_text_report(report))
        self.assertFalse(
            any("snapshots were not recorded" in notice for notice in report["notices"])
        )

    def test_explicit_selection_can_be_outside_page_but_not_outside_run(self) -> None:
        run_id = self.create_run(2)
        first, second = self.mock_result(start=0), self.mock_result(start=1)
        self.save(run_id, first)
        self.save(run_id, second)
        other_run = self.create_run()
        other = self.mock_result(game="matrix_rps")
        self.save(other_run, other)
        with self.sessions() as session:
            selected = build_run_report(
                session, run_id, episode_id=second["episode_id"], episode_limit=1
            )
            self.assertEqual(
                selected["episodes"]["items"][0]["episode_id"], first["episode_id"]
            )
            self.assertEqual(
                selected["selected_episode"]["episode_id"], second["episode_id"]
            )
            with self.assertRaises(ReportNotFound):
                build_run_report(session, run_id, episode_id=other["episode_id"])
            with self.assertRaises(ReportNotFound):
                build_run_report(session, str(uuid.uuid4()))

    def test_terminal_text_is_ascii_and_drops_control_sequences(self) -> None:
        secret = "never-export-provider-key"
        run_id = self.create_run(
            configuration={
                "game": "matrix_rps",
                "seed": 19,
                "api_key": secret,
                "metadata": {"secret": secret},
                "runtime": {"source_sha256": "recorded", "api_key": secret},
            }
        )
        record = self.mock_result(game="matrix_rps", steps=10)
        decision = record.get("steps", [])[0]["decisions"][0]
        malicious = "\x1b[31mRED\x1b[0m\n\x1b]52;c;CLIPBOARD_PAYLOAD\x07safe\n\x9d8;;https://malicious.invalid\x9clink\x9d8;;\x9c\n<script>alert('literal text')</script>\x00\x08\r\x85 snowman \u2603"
        decision["response"] = malicious
        decision["summary"] = "Recorded summary \x1b[2J do not erase"
        decision["prompt"] = "PROMPT_ONLY_MARKER"
        record.get("manifest", {})["model_configs"][0]["api_key"] = secret
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id)
        self.assertEqual(
            report["transitions"]["items"][0]["decisions"][0]["response"], malicious
        )
        self.assertNotIn(secret, json.dumps(report))
        text = render_text_report(report, width=72)
        self.assertTrue(text.isascii())
        self.assertTrue(all(char == "\n" or 32 <= ord(char) <= 126 for char in text))
        self.assertTrue(all(len(line) <= 72 for line in text.splitlines()))
        self.assertNotIn("CLIPBOARD_PAYLOAD", text)
        self.assertNotIn("malicious.invalid", text)
        self.assertIn("RED", text)
        self.assertIn("\\u2603", text)
        self.assertIn("<script>", text)  # Literal plain text, never HTML rendering.
        self.assertNotIn("PROMPT_ONLY_MARKER", text)
        self.assertIn(
            "PROMPT_ONLY_MARKER", render_text_report(report, include_prompt=True)
        )
        self.assertEqual(terminal_safe("safe\x1b]52;c;unfinished"), "safe")

    def test_legacy_failed_episode_has_no_invented_player_or_reasoning(self) -> None:
        run_id = self.create_run()
        identity = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Episode(
                    episode_id=identity,
                    run_id=run_id,
                    matchup_id=None,
                    worker_name=None,
                    env="OPENSPIEL",
                    agent=["legacy-model"],
                    type="RAND",
                    game="chess",
                    status="FAILED",
                    winner="UNDETERMINED/TIE",
                    num_steps=2,
                    error="Legacy execution failed",
                    summary=["legacy-summary-not-safe-to-attribute"],
                    action=["e4", "d5"],
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                )
            )
            session.commit()
            report = build_run_report(session, run_id)
        selected = report["selected_episode"]
        self.assertFalse(selected["trace_available"])
        self.assertIsNone(selected["player_agents"])
        self.assertIsNone(selected["returns"])
        self.assertIsNone(selected["mock"])
        self.assertEqual(report["transitions"]["items"], [])
        text = render_text_report(report)
        self.assertIn("Legacy episode", text)
        self.assertIn("Legacy execution failed", text)
        self.assertNotIn("legacy-summary-not-safe-to-attribute", text)
        self.assertNotIn("Recorded player ID: 0", text)

    def test_chance_and_mean_field_nodes_keep_their_recorded_data(self) -> None:
        for game, players, expected in (
            ("kuhn_poker", 2, "chance"),
            ("mfg_crowd_modelling(horizon=2)", 1, "mean_field"),
        ):
            with self.subTest(game=game):
                run_id = self.create_run()
                record = self.mock_result(game=game, steps=100, players=players)
                self.save(run_id, record)
                with self.sessions() as session:
                    report = build_run_report(session, run_id)
                self.assertIn(
                    expected, {item["kind"] for item in report["transitions"]["items"]}
                )
                self.assertEqual(
                    report["transitions"]["items"],
                    json.loads(json.dumps(record.get("steps"))),
                )
                text = render_text_report(report)
                self.assertIn(expected, text)
                if expected == "mean_field":
                    self.assertIn("does not establish an equilibrium", text)
                    self.assertIn("Distribution Support", text)
                else:
                    self.assertIn("Chance Outcomes", text)
                    self.assertTrue(
                        all(
                            "state_before" not in item and "state_after" not in item
                            for item in report["transitions"]["items"]
                        )
                    )

    def test_truncated_and_fallback_assisted_results_are_labeled_honestly(self) -> None:
        run_id = self.create_run()
        record = self.mock_result()
        record["fallback_count"] = 1
        record["degraded"] = True
        record.get("steps", [])[0]["decisions"][0]["fallback_reason"] = {
            "category": "model_api",
            "code": "model_call_failed",
            "message": "Injected provider failure",
        }
        self.save(run_id, record)
        with self.sessions() as session:
            report = build_run_report(session, run_id)
        text = render_text_report(report)
        self.assertIn("TRUNCATED", text)
        self.assertIn(
            "intermediate returns are not a final score", " ".join(text.split())
        )
        self.assertIn("Fallback-assisted", text)
        self.assertIn("Recorded fallback summary", text)
        self.assertIn("Injected provider failure", " ".join(text.split()))

    def test_pending_empty_run_and_invalid_paging(self) -> None:
        run_id = self.create_run()
        with self.sessions() as session:
            report = build_run_report(session, run_id)
            self.assertEqual(report["run"]["episodes_saved"], 0)
            self.assertIsNone(report["selected_episode"])
            self.assertEqual(report["transitions"]["total"], 0)
            invalid_options: tuple[dict[str, Any], ...] = (
                {"episode_limit": 0},
                {"episode_limit": 101},
                {"step_limit": 201},
                {"step_offset": -1},
                {"episode_offset": True},
                {"episode_id": ""},
            )
            for options in invalid_options:
                with (
                    self.subTest(options=options),
                    self.assertRaises(ReportRequestError),
                ):
                    build_run_report(session, run_id, **options)
        self.assertIn("No episodes", render_text_report(report))
        with self.assertRaises(ReportRequestError):
            render_text_report(report, width=0)


if __name__ == "__main__":
    unittest.main()
