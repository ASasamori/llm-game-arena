"""The README demo uses real, replayable mock transitions and an isolated DB."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts.seed_demo import PROVIDER_KEYS, seed_demo


class DemoSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "demo"
        # Any accidental provider constructor or outbound socket is a failure,
        # rather than a paid request hidden inside documentation generation.
        ambient = {name: "ambient-sensitive-value" for name in PROVIDER_KEYS}
        ambient.update(
            DATABASE_URL="postgresql://untouched-host/private",
            MODEL_API_MOCK_MODE="false",
            WORKER_TRIGGER_MODE="cloud_run_job",
            PYTHON_DOTENV_DISABLED="0",
        )
        with patch.dict(os.environ, ambient):
            before = dict(os.environ)
            with patch(
                "socket.socket.connect",
                side_effect=AssertionError("Demo attempted network I/O"),
            ):
                cls.manifest = seed_demo(cls.root)
            cls.ambient_restored = dict(os.environ) == before

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def results(self) -> list[dict[str, Any]]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from orchestrator.models import EpisodeTrace, EpisodeTransition

        engine = create_engine(self.manifest["database_url"])
        try:
            with Session(engine) as session:
                records = []
                for demo in self.manifest["runs"]:
                    trace = session.get(EpisodeTrace, demo["episode_id"])
                    assert trace is not None
                    record = dict(trace.payload)
                    record["steps"] = [
                        row.payload
                        for row in session.query(EpisodeTransition)
                        .filter_by(episode_id=demo["episode_id"])
                        .order_by(EpisodeTransition.step_index)
                    ]
                    records.append(record)
                return records
        finally:
            engine.dispose()

    def test_all_three_games_replay_and_are_explicitly_mock(self) -> None:
        from game_arena.openspiel.replay import replay_episode

        self.assertEqual(
            {entry["name"] for entry in self.manifest["runs"]},
            {"connect_four", "avalon", "catan"},
        )
        self.assertTrue(self.manifest["mock_only"])
        self.assertEqual(self.manifest["provider_requests"], 0)
        self.assertEqual(
            json.loads((self.root / "manifest.json").read_text()), self.manifest
        )
        for record in self.results():
            with self.subTest(game=record["game"]):
                self.assertTrue(replay_episode(record)["verified"])
                self.assertEqual(record["fallback_count"], 0)
                self.assertTrue(
                    all(agent.startswith("MOCK:") for agent in record["player_agents"])
                )
                self.assertTrue(
                    all(
                        config["mock"] for config in record["manifest"]["model_configs"]
                    )
                )
                for step in record["steps"]:
                    for decision in step["decisions"]:
                        self.assertEqual(decision["selection"]["kind"], "mock")
                        self.assertIn("Mock script:", decision["response"])
                        self.assertEqual(
                            decision["response_metadata"]["provider_requests"], 0
                        )
                        self.assertIn(decision["action"], decision["legal_actions"])

    def test_highlights_are_real_rules_outcomes_not_edited_board_states(self) -> None:
        records = {record["game"].split("(")[0]: record for record in self.results()}
        connect = records["connect_four"]
        self.assertTrue(connect["terminal"])
        self.assertEqual(connect["returns"], [1.0, -1.0])
        self.assertEqual(
            [step["actions"][0] for step in connect["steps"]], [3, 2, 3, 2, 3, 2, 3]
        )
        avalon = records["avalon"]
        self.assertEqual(avalon["stop_reason"], "quest_limit")
        self.assertEqual(len(avalon["game_progress"]["quest_results"]), 3)
        self.assertFalse(avalon["terminal"])
        catan = records["catan"]
        self.assertTrue(catan["terminal"])
        board = catan["steps"][-1]["public_state_after"]
        self.assertEqual(len(board["board"]["tiles"]), 19)
        catan_demo = next(
            entry for entry in self.manifest["runs"] if entry["name"] == "catan"
        )
        self.assertTrue(catan_demo["trade_offer_steps"])
        for index in catan_demo["trade_offer_steps"]:
            self.assertIn("Send trade offer", catan["steps"][index]["action_strings"])

    def test_existing_data_is_never_overwritten(self) -> None:
        before = (self.root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new or empty directory"):
            seed_demo(self.root)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.txt"
            target.write_text("retain this file")
            with self.assertRaises(ValueError):
                seed_demo(Path(directory))
            self.assertEqual(target.read_text(), "retain this file")

    def test_symlink_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "link"
            destination = Path(directory) / "target"
            destination.mkdir()
            link.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                seed_demo(link)
            self.assertEqual(list(destination.iterdir()), [])

    def test_generation_ignores_and_restores_ambient_credentials_and_database(
        self,
    ) -> None:
        self.assertTrue(self.ambient_restored)
        self.assertEqual(
            self.manifest["database_url"],
            "sqlite+pysqlite:///" + str(self.root.resolve() / "arena.sqlite"),
        )


if __name__ == "__main__":
    unittest.main()
