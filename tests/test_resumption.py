"""Validated model prefixes continue without repeat provider requests."""

from copy import deepcopy
import json
from typing import Any
import unittest

from game_arena.openspiel.harness import EpisodeStopped, harness
from game_arena.openspiel.model_generation import GenerateResponse, LLMModel, MockModel
from game_arena.openspiel.replay import replay_episode
from game_arena.openspiel.resumption import ResumeError, resume_episode
from game_arena.openspiel.utils import HarnessOptions


class _Policy(MockModel):
    def __init__(self, seat: int, counter: list[int], allowance: int = 10000):
        super().__init__(f"seat-{seat}")
        self.counter, self.allowance, self.calls = counter, allowance, 0

    def generate_with_text_input(self, prompt, system_instruction=None, mock=False):
        if self.counter[0] >= self.allowance:
            raise EpisodeStopped("provider_error", "Offline controlled provider stop")
        self.counter[0] += 1
        self.calls += 1
        payload = json.JSONDecoder().raw_decode(prompt[prompt.rfind("\n{") + 1 :])[0]
        phase = json.loads(payload["observation"])["phase"]
        if phase == "assassinate":
            raise AssertionError(
                "Resumed episode called Assassin beyond its quest limit"
            )
        action = 1 if phase == "vote" else payload["legal_actions"][0]["id"]
        return GenerateResponse(
            f"Summary: Offline test policy.\nFinal Answer: {action}",
            usage={"input_tokens": 12, "output_tokens": 8},
            metadata={"mock": True, "test": "original"},
        )


def _models(allowance=10000):
    counter = [0]
    return [_Policy(p, counter, allowance) for p in range(6)]


def _prefix(allowance=15) -> dict[str, Any]:
    competitors: list[LLMModel] = list(_models(allowance))
    return dict(
        harness(
            HarnessOptions(
                type="PVP",
                models=competitors,
                game="avalon(players=6,discussion_rounds=1)",
                num_steps=1000,
                input=None,
                seed=123,
                failure_policy="raise",
                stop_after_quests=3,
            )
        )
    )


class ResumptionTests(unittest.TestCase):
    def test_resume_preserves_every_prior_turn_and_calls_only_for_new_choices(self):
        prefix = _prefix()
        self.assertEqual(prefix["decision_count"], 15)
        self.assertEqual(prefix["stop_reason"], "provider_error")
        original = deepcopy(prefix)
        live = _models()
        result: dict[str, Any] = dict(resume_episode(prefix, list(live)))
        self.assertEqual(prefix, original)
        self.assertEqual(result["steps"][: len(prefix["steps"])], prefix["steps"])
        self.assertEqual(sum(m.calls for m in live), result["decision_count"] - 15)
        self.assertEqual(result["episode_id"], prefix["episode_id"])
        self.assertEqual(result["start_time"], prefix["start_time"])
        self.assertEqual(result["stop_reason"], "quest_limit")
        self.assertEqual(len(result["game_progress"]["quest_results"]), 3)
        self.assertEqual(result["game_progress"]["phase"], "assassinate")
        self.assertTrue(replay_episode(result)["verified"])
        self.assertEqual(
            result["manifest"]["resume_history"][-1]["replayed_provider_requests"], 0
        )

    def test_json_saved_prefix_retains_original_start_and_latency_records(self):
        prefix = _prefix()
        saved = json.loads(json.dumps(prefix, default=lambda v: v.isoformat()))
        result = resume_episode(saved, list(_models()))
        self.assertEqual(result["start_time"].isoformat(), saved["start_time"])
        self.assertEqual(result.get("steps", [])[: len(saved["steps"])], saved["steps"])

    def test_tampered_prefix_or_models_rejected_before_provider_calls(self):
        for kind in ("prompt", "hash", "identity", "version", "mock", "count"):
            with self.subTest(kind=kind):
                prefix = _prefix()
                live = _models()
                if kind == "prompt":
                    next(e for e in prefix["steps"] if e["decisions"])["decisions"][0][
                        "prompt"
                    ] += " changed"
                elif kind == "hash":
                    prefix["steps"][0]["state_hash"] = "changed"
                elif kind == "identity":
                    live[0]._model_name = "other"
                elif kind == "version":
                    prefix["manifest"]["model_configs"][0]["model_name"] = "other"
                elif kind == "mock":
                    live[0]._mock_mode = False
                else:
                    live.pop()
                with self.assertRaises(ValueError):
                    resume_episode(prefix, list(live))
                self.assertEqual(sum(m.calls for m in live), 0)

    def test_completed_or_quest_limited_episode_cannot_resume(self):
        models: list[LLMModel] = list(_models())
        done = dict(
            harness(
                HarnessOptions(
                    type="PVP",
                    models=models,
                    game="avalon(players=6,discussion_rounds=0)",
                    num_steps=1000,
                    input=None,
                    seed=123,
                    failure_policy="raise",
                    stop_after_quests=3,
                )
            )
        )
        live = _models()
        with self.assertRaises(ResumeError):
            resume_episode(done, list(live))
        self.assertEqual(sum(m.calls for m in live), 0)

    def test_new_provider_error_preserves_the_entire_original_prefix(self):
        prefix = _prefix()
        live = _models(0)
        result = resume_episode(prefix, list(live))
        self.assertEqual(result.get("steps"), prefix["steps"])
        self.assertEqual(result.get("decision_count"), 15)
        self.assertEqual(result.get("stop_reason"), "provider_error")
        self.assertEqual(sum(m.calls for m in live), 0)
        self.assertTrue(replay_episode(dict(result))["verified"])
