"""Controlled trial stops retain evidence without extra calls or fabricated wins."""

from __future__ import annotations

from copy import deepcopy
import json
import random
from typing import Any
import unittest

from game_arena.environments.avalon import EVIL_ROLES
from game_arena.environments.registry import load_game
from game_arena.openspiel.harness import EpisodeStopped, harness
from game_arena.openspiel.model_generation import GenerateResponse, MockModel
from game_arena.openspiel.provenance import derive_seed
from game_arena.openspiel.replay import ReplayError, replay_episode
from game_arena.openspiel.utils import HarnessOptions


def options(game: str, models: list[Any], **changes: Any) -> HarnessOptions:
    result: Any = {
        "type": "PVP",
        "game": game,
        "models": models,
        "num_steps": 1000,
        "input": None,
        "seed": 23,
    }
    result.update(changes)
    return result


def prompt_payload(prompt: str) -> dict[str, Any]:
    return json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])[0]


def final_state(record: dict[str, Any]) -> Any:
    state = load_game(record["game_spec"]).new_initial_state()
    for event in record["steps"]:
        state.apply_action(event["actions"][0])
    return state


class QuestPolicy(MockModel):
    def __init__(self, seat: int, *, fail: bool = False, reject: bool = False):
        super().__init__(f"seat-{seat}")
        self.fail, self.reject = fail, reject
        self.calls: list[str] = []

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        payload = prompt_payload(prompt)
        observation = json.loads(payload["observation"])
        phase = observation["phase"]
        self.calls.append(phase)
        if phase == "assassinate":
            raise AssertionError("Quest-limited trial called the Assassin")
        action = payload["legal_actions"][0]["id"]
        if phase == "vote":
            action = 0 if self.reject else 1
        elif phase == "quest":
            action = int(self.fail and observation["role"] in EVIL_ROLES)
        return GenerateResponse(
            f"Summary: Scripted environment test.\nFinal Answer: {action}",
            metadata={"mock": True},
        )


class BudgetPolicy(MockModel):
    def __init__(self, counter: list[int], allowance: int, code: str):
        super().__init__("budget-test")
        self.counter, self.allowance, self.code = counter, allowance, code

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        if self.counter[0] >= self.allowance:
            raise EpisodeStopped(
                self.code, "Test resource boundary reached before request"
            )
        self.counter[0] += 1
        return super().generate_with_text_input(prompt, system_instruction, mock)


class EpisodeStopTests(unittest.TestCase):
    def test_quest_limits_stop_at_exact_public_boundary_before_assassination(
        self,
    ) -> None:
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                models = [QuestPolicy(seat) for seat in range(6)]
                record: dict[str, Any] = dict(
                    harness(
                        options(
                            "avalon(players=6,discussion_rounds=0)",
                            models,
                            stop_after_quests=limit,
                        )
                    )
                )
                state = final_state(record)
                self.assertEqual(len(state.quest_results), limit)
                self.assertTrue(
                    all(result["success"] for result in state.quest_results)
                )
                self.assertEqual(record["stop_reason"], "quest_limit")
                self.assertEqual(record["status"], "TRUNCATED")
                self.assertFalse(record["terminal"])
                self.assertEqual(record["returns"], [0.0] * 6)
                self.assertEqual(record["winner_players"], [])
                self.assertFalse(any("assassinate" in model.calls for model in models))
                if limit == 3:
                    self.assertEqual(state.phase, "assassinate")
                self.assertTrue(replay_episode(record)["verified"])

    def test_final_quest_results_survive_stop_and_progress_tampering_is_rejected(
        self,
    ) -> None:
        record: dict[str, Any] = dict(
            harness(
                options(
                    "avalon(players=6,discussion_rounds=0)",
                    [QuestPolicy(p) for p in range(6)],
                    stop_after_quests=3,
                )
            )
        )
        progress = record["game_progress"]
        self.assertEqual(
            set(progress),
            {"phase", "quest_results", "leader", "consecutive_rejections"},
        )
        self.assertEqual(progress["phase"], "assassinate")
        self.assertEqual(len(progress["quest_results"]), 3)
        self.assertEqual(progress["quest_results"][-1]["quest"], 3)
        self.assertTrue(progress["quest_results"][-1]["success"])
        for corruption in ("result", "missing_quest", "missing_progress"):
            with self.subTest(corruption=corruption):
                changed = deepcopy(record)
                if corruption == "result":
                    changed["game_progress"]["quest_results"][-1]["success"] = False
                elif corruption == "missing_quest":
                    changed["game_progress"]["quest_results"].pop()
                else:
                    changed.pop("game_progress")
                with self.assertRaisesRegex(ReplayError, "public game progress"):
                    replay_episode(changed)

    def test_natural_three_failure_ending_is_preserved(self) -> None:
        spec = "avalon(players=6,discussion_rounds=0)"
        seed = None
        for candidate in range(100):
            state: Any = load_game(spec).new_initial_state()
            rng = random.Random(derive_seed(candidate, "chance"))
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                state.apply_action(
                    rng.choices(
                        [a for a, _ in outcomes], weights=[p for _, p in outcomes]
                    )[0]
                )
            if any(role in EVIL_ROLES for role in state.roles[:2]):
                seed = candidate
                break
        self.assertIsNotNone(seed)
        record: dict[str, Any] = dict(
            harness(
                options(
                    spec,
                    [QuestPolicy(p, fail=True) for p in range(6)],
                    seed=seed,
                    stop_after_quests=3,
                )
            )
        )
        state = final_state(record)
        self.assertEqual(len(state.quest_results), 3)
        self.assertTrue(all(not result["success"] for result in state.quest_results))
        self.assertEqual(state.end_reason, "three_failed_quests")
        self.assertEqual(record["status"], "SUCCESS")
        self.assertTrue(record["terminal"])
        self.assertIsNone(record["stop_reason"])
        self.assertTrue(replay_episode(record)["verified"])

    def test_five_rejections_can_end_before_requested_quest_count(self) -> None:
        record: dict[str, Any] = dict(
            harness(
                options(
                    "avalon(players=6,discussion_rounds=0)",
                    [QuestPolicy(p, reject=True) for p in range(6)],
                    stop_after_quests=3,
                )
            )
        )
        state = final_state(record)
        self.assertEqual(len(state.quest_results), 0)
        self.assertEqual(record["status"], "SUCCESS")
        self.assertTrue(record["terminal"])
        self.assertIsNone(record["stop_reason"])
        self.assertTrue(replay_episode(record)["verified"])

    def test_controlled_stops_preserve_previous_decisions_without_fallback(
        self,
    ) -> None:
        for code in ("budget_exhausted", "provider_error", "call_limit"):
            with self.subTest(code=code):
                counter = [0]
                models = [BudgetPolicy(counter, 3, code) for _ in range(2)]
                record: dict[str, Any] = dict(
                    harness(options("tic_tac_toe", models, failure_policy="random"))
                )
                self.assertEqual(counter, [3])
                self.assertEqual(record["num_steps"], 3)
                self.assertEqual(record["decision_count"], 3)
                self.assertEqual(record["fallback_count"], 0)
                self.assertFalse(record["degraded"])
                self.assertEqual(record["stop_reason"], code)
                self.assertEqual(record["status"], "TRUNCATED")
                self.assertFalse(record["terminal"])
                self.assertTrue(replay_episode(record)["verified"])

    def test_stop_during_joint_choice_does_not_apply_partial_joint_transition(
        self,
    ) -> None:
        counter = [0]
        models = [BudgetPolicy(counter, 1, "budget_exhausted") for _ in range(2)]
        record: dict[str, Any] = dict(harness(options("matrix_rps", models)))
        self.assertEqual(counter, [1])
        self.assertEqual(record["steps"], [])
        self.assertEqual(record["stop_reason"], "budget_exhausted")
        self.assertFalse(record["terminal"])
        self.assertTrue(replay_episode(record)["verified"])

    def test_unpermitted_controlled_stop_does_not_turn_into_random_play(self) -> None:
        counter = [0]
        with self.assertRaises(EpisodeStopped):
            harness(
                options(
                    "tic_tac_toe",
                    [BudgetPolicy(counter, 0, "call_limit")] * 2,
                    permitted_stop_reasons=[],
                )
            )
        self.assertEqual(counter, [0])

    def test_replay_refuses_false_or_unpermitted_stop_reasons(self) -> None:
        counter = [0]
        record: dict[str, Any] = dict(
            harness(
                options(
                    "tic_tac_toe", [BudgetPolicy(counter, 2, "budget_exhausted")] * 2
                )
            )
        )
        for change in ("unknown", "quest_limit", None):
            with self.subTest(reason=change):
                altered = deepcopy(record)
                altered["stop_reason"] = change
                if change is None:
                    altered["stop_message"] = None
                with self.assertRaises(ReplayError):
                    replay_episode(altered)
        altered = deepcopy(record)
        altered["manifest"]["permitted_stop_reasons"] = []
        with self.assertRaises(ReplayError):
            replay_episode(altered)

    def test_replay_rejects_transition_after_quest_boundary(self) -> None:
        record: dict[str, Any] = dict(
            harness(
                options(
                    "avalon(players=6,discussion_rounds=0)",
                    [QuestPolicy(p) for p in range(6)],
                    stop_after_quests=3,
                )
            )
        )
        added = deepcopy(record["steps"][0])
        added["index"] = len(record["steps"])
        record["steps"].append(added)
        with self.assertRaisesRegex(ReplayError, "quest limit"):
            replay_episode(record)

    def test_invalid_limits_and_stop_codes_fail_before_model_calls(self) -> None:
        for value in (False, 0, 4, 1.0, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                harness(
                    options(
                        "avalon(players=6)", [MockModel()] * 6, stop_after_quests=value
                    )
                )
        with self.assertRaisesRegex(ValueError, "only to native Avalon"):
            harness(options("tic_tac_toe", [MockModel()] * 2, stop_after_quests=3))
        with self.assertRaises(ValueError):
            EpisodeStopped("made_up", "Invalid")
        with self.assertRaises(ValueError):
            EpisodeStopped("budget_exhausted", "")
        with self.assertRaises(ValueError):
            harness(
                options(
                    "tic_tac_toe", [MockModel()] * 2, permitted_stop_reasons=["made_up"]
                )
            )

    def test_legacy_step_limit_traces_still_require_their_full_limit(self) -> None:
        record: dict[str, Any] = dict(
            harness(options("tic_tac_toe", [MockModel()] * 2, num_steps=3))
        )
        for key in ("stop_reason", "stop_message"):
            record.pop(key, None)
        for key in ("stop_after_quests", "permitted_stop_reasons"):
            record["manifest"].pop(key, None)
        self.assertTrue(replay_episode(record)["verified"])
        record["manifest"]["max_steps"] += 1
        with self.assertRaisesRegex(ReplayError, "before its configured limit"):
            replay_episode(record)
