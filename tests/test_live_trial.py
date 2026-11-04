"""The paid-trial path is exercised with policy stubs and no provider calls."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from rq import Queue as RealQueue
from game_arena.openspiel.model_generation import GenerateResponse, MockModel
from orchestrator import models, tasks
from orchestrator.live_trial import DEFAULT_SEATS, run_trial, submit_trial
from orchestrator.reports import build_run_report, render_text_report


class _ApprovingModel(MockModel):
    def generate_with_text_input(self, prompt, system_instruction=None, mock=False):
        payload = json.JSONDecoder().raw_decode(prompt[prompt.rfind("\n{") + 1 :])[0]
        observation = json.loads(payload["observation"])
        action = (
            1 if observation["phase"] == "vote" else payload["legal_actions"][0]["id"]
        )
        return GenerateResponse(
            f"Summary: Offline approval policy.\nFinal Answer: {action}"
        )


class LiveTrialTests(unittest.TestCase):
    def test_one_queued_trial_preserves_three_quests_and_never_repeats_paid_path(self):
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
                "game_arena.openspiel.budgeted_anthropic.BudgetedClaudeModel",
                side_effect=lambda name, ledger, max_tokens: _ApprovingModel(name),
            ) as factory,
        ):
            queue.return_value.enqueue.return_value = SimpleNamespace(
                id="offline-test-job"
            )
            root = Path(directory)
            trial = submit_trial(root)
            self.assertEqual(trial["configuration"]["budget_usd"], "0.90")
            self.assertEqual(len(DEFAULT_SEATS), 6)
            self.assertEqual(DEFAULT_SEATS.count("claude-haiku-5-5"), 2)
            self.assertEqual(len(set(DEFAULT_SEATS)), 5)
            result = run_trial(trial["run_id"], str(root))
            self.assertEqual(result["status"], "TRUNCATED")
            self.assertEqual(result["stop_reason"], "quest_limit")
            self.assertTrue(result["replay_verified"])
            episode = json.loads((root / "episode.json").read_text())
            self.assertEqual(len(episode["game_progress"]["quest_results"]), 3)
            self.assertEqual(episode["game_progress"]["phase"], "assassinate")
            self.assertFalse(episode["terminal"])
            with sessions() as session:
                self.assertEqual(
                    session.get(models.Run, trial["run_id"]).status, "STOPPED"
                )
                report = build_run_report(session, trial["run_id"])
            selected = report["selected_episode"]
            self.assertEqual(selected["stop_reason"], "quest_limit")
            self.assertEqual(len(selected["game_progress"]["quest_results"]), 3)
            self.assertIn("quest_limit", render_text_report(report))
            self.assertTrue(run_trial(trial["run_id"], str(root))["reused"])
            self.assertEqual(factory.call_count, 6)
            self.assertEqual(queue.return_value.enqueue.call_count, 1)
            queued = queue.return_value.enqueue.call_args.args
            self.assertEqual(queued[0], "orchestrator.live_trial.run_trial")
            self.assertEqual(
                RealQueue.parse_args(*queued)[0], "orchestrator.live_trial.run_trial"
            )
            with self.assertRaises(ValueError):
                submit_trial(root)
