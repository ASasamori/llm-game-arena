"""Behavioral tests for privacy, game dynamics, reproducibility and trace replay."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import random
from typing import Any
import unittest
from unittest.mock import patch

from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import GenerateResponse, MockModel
from game_arena.openspiel.parsers import parse_legal_action
from game_arena.openspiel.replay import ReplayError, replay_episode
from game_arena.openspiel.utils import HarnessOptions


def options(game: str, players: int = 2, **changes: Any) -> HarnessOptions:
    result: Any = {
        "type": "PVP",
        "models": [MockModel(str(p)) for p in range(players)],
        "num_steps": 100,
        "input": None,
        "game": game,
        "seed": 71,
    }
    result.update(changes)
    return result


def execute(config: HarnessOptions) -> dict[str, Any]:
    result: dict[str, Any] = dict(harness(config))
    return result


def prompt_data(prompt: str) -> dict[str, Any]:
    return json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])[0]


class EngineTests(unittest.TestCase):
    def test_sequential_chance_simultaneous_single_and_n_player_games_replay(
        self,
    ) -> None:
        for game, players in [
            ("tic_tac_toe", 2),
            ("kuhn_poker(players=3)", 3),
            ("matrix_rps", 2),
            ("matching_pennies_3p", 3),
            ("catch", 1),
            ("turn_based_simultaneous_game(game=matrix_rps())", 2),
        ]:
            with self.subTest(game=game):
                record = execute(options(game, players))
                self.assertEqual(record["status"], "SUCCESS")
                self.assertEqual(record["fallback_count"], 0)
                self.assertEqual(len(record["returns"]), players)
                self.assertTrue(replay_episode(dict(record))["verified"])

    def test_private_deals_stay_out_of_player_prompts(self) -> None:
        record = execute(options("kuhn_poker"))
        self.assertEqual(record["steps"][0]["kind"], "chance")
        for event in record["steps"]:
            for decision in event["decisions"]:
                payload = prompt_data(decision["prompt"])
                self.assertEqual(payload["observation"], decision["observation"])
                self.assertEqual(
                    payload["observation_type"], "information_state_string"
                )
                self.assertNotIn("move_history", payload)
                self.assertNotIn("state", payload)
                self.assertNotIn("chance_outcomes", payload)
                self.assertTrue(
                    all(
                        set(action) == {"action", "label"}
                        for action in payload["own_action_history"]
                    )
                )

    def test_simultaneous_choices_use_the_same_unmodified_state(self) -> None:
        from game_arena.openspiel.catalog import load_game

        initial = load_game("matching_pennies_3p").new_initial_state()
        record = execute(options("matching_pennies_3p", 3))
        event = record["steps"][0]
        self.assertEqual(event["kind"], "simultaneous")
        self.assertEqual(len(event["decisions"]), 3)
        for decision in event["decisions"]:
            payload = prompt_data(decision["prompt"])
            self.assertEqual(payload["own_action_history"], [])
            self.assertEqual(payload["player"], decision["player"])
            self.assertEqual(
                decision["observation"],
                initial.information_state_string(decision["player"]),
            )

    def test_seed_does_not_depend_on_global_rng_or_another_episode(self) -> None:
        config = options("kuhn_poker(players=3)", 3, type="RAND", models=[MockModel()])
        first = execute(config)
        random.seed(999)
        execute(options("tic_tac_toe", seed=999))
        second = execute(config)
        self.assertEqual(first["action"], second["action"])
        self.assertEqual(first["returns"], second["returns"])
        self.assertEqual(first["final_state_hash"], second["final_state_hash"])

    def test_sampled_native_rng_replays_from_canonical_parameters(self) -> None:
        first = execute(options("negotiation"))
        second = execute(options("negotiation"))
        self.assertIn("rng_seed=", first["game_spec"])
        self.assertEqual(first["final_state_hash"], second["final_state_hash"])
        self.assertTrue(replay_episode(dict(first))["verified"])

    def test_input_script_is_not_consumed_by_execution_or_retry(self) -> None:
        script = deque([6, 6])
        config = options(
            "connect_four",
            type="INPUT",
            models=[MockModel()],
            num_steps=4,
            input=script,
        )
        first, second = execute(config), execute(config)
        self.assertEqual(script, deque([6, 6]))
        self.assertEqual(first["action"], second["action"])
        self.assertTrue(replay_episode(dict(first))["verified"])

    def test_fallback_is_explicit_and_strict_mode_raises(self) -> None:
        class BrokenModel(MockModel):
            def generate_with_text_input(
                self,
                prompt: str,
                system_instruction: str | None = None,
                mock: bool = False,
            ) -> GenerateResponse:
                return GenerateResponse("Final Answer: 99999")

        config = options(
            "tic_tac_toe", models=[BrokenModel(), MockModel()], num_steps=1
        )
        record = execute(config)
        self.assertEqual(record["fallback_count"], 1)
        self.assertTrue(record["degraded"])
        self.assertIsNotNone(record["steps"][0]["decisions"][0]["fallback_reason"])
        self.assertTrue(replay_episode(dict(record))["verified"])
        config["failure_policy"] = "raise"
        with self.assertRaisesRegex(ValueError, "Illegal action"):
            execute(config)

    def test_truncated_utility_is_not_a_terminal_win(self) -> None:
        record = execute(options("tic_tac_toe", num_steps=1))
        self.assertEqual(record["status"], "TRUNCATED")
        self.assertFalse(record["terminal"])
        self.assertEqual(record["winner"], "UNDETERMINED/TIE")
        self.assertEqual(record["winner_players"], [])

    def test_mean_field_requires_an_explicit_population_assumption(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit mean_field_distribution"):
            execute(options("mfg_crowd_modelling", 1))
        record = execute(
            options("mfg_crowd_modelling", 1, mean_field_distribution="uniform")
        )
        self.assertEqual(record["status"], "SUCCESS")
        self.assertTrue(any(event["kind"] == "mean_field" for event in record["steps"]))
        self.assertTrue(replay_episode(dict(record))["verified"])

    def test_replay_rejects_changed_action_reward_prompt_and_missing_tail(self) -> None:
        original = dict(execute(options("tic_tac_toe")))
        for corruption in ("action", "reward", "prompt", "tail"):
            with self.subTest(corruption=corruption):
                record = deepcopy(original)
                if corruption == "action":
                    record["steps"][0]["actions"][0] = 99999
                elif corruption == "reward":
                    record["steps"][0]["rewards"][0] = 99999
                elif corruption == "prompt":
                    record["steps"][0]["decisions"][0]["prompt"] += "injected"
                else:
                    record["steps"].pop()
                with self.assertRaises(ReplayError):
                    replay_episode(record)

    def test_generic_action_parser_preserves_arbitrary_labels(self) -> None:
        labels = {5: "call_raises (2)", 7: "pass turn"}
        parsed = parse_legal_action(
            "Summary: a choice\nFinal Answer: call_raises (2)", labels
        )
        self.assertEqual((parsed.action, parsed.summary), (5, "a choice"))
        self.assertEqual(
            parse_legal_action('{"action":7,"summary":"hold"}', labels).action, 7
        )
        with self.assertRaises(ValueError):
            parse_legal_action('{"action":true}', labels)

    def test_seed_option_wins_over_invalid_process_default(self) -> None:
        with patch.dict("os.environ", {"HARNESS_RANDOM_SEED": "bad-value"}):
            self.assertEqual(execute(options("tic_tac_toe"))["seed"], 71)


if __name__ == "__main__":
    unittest.main()
