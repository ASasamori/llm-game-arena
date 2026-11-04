"""Representative game families exercise the generic state machine with mocks."""

from pathlib import Path
import subprocess
import tempfile

import pyspiel
import unittest
from unittest.mock import patch

from game_arena.openspiel import conformance, model_generation


class GameConformanceTests(unittest.TestCase):
    def test_representative_families_repeat_without_provider_calls(self) -> None:
        cases = [
            ("tic_tac_toe", 12, {"decision"}, 2),
            ("matrix_rps", 1, {"simultaneous"}, 2),
            ("matching_pennies_3p", 2, {"simultaneous"}, 3),
            ("kuhn_poker(players=3)", 16, {"chance", "decision"}, 3),
            (
                "mfg_crowd_modelling(horizon=2)",
                16,
                {"chance", "decision", "mean_field"},
                1,
            ),
            ("negotiation", 4, {"chance", "decision"}, 2),
            ("python_tic_tac_toe", 12, {"decision"}, 2),
        ]
        with (
            patch.object(
                model_generation,
                "GPTModel",
                side_effect=AssertionError("provider constructed"),
            ),
            patch.object(
                model_generation,
                "GeminiModel",
                side_effect=AssertionError("provider constructed"),
            ),
            patch.object(
                model_generation,
                "ClaudeModel",
                side_effect=AssertionError("provider constructed"),
            ),
        ):
            for spec, max_steps, kinds, players in cases:
                with self.subTest(spec=spec):
                    result = conformance._check_episode(spec, max_steps, seed=71)
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["num_players"], players)
                    self.assertTrue(result["repeat_verified"])
                    self.assertTrue(result["replay_verified"])
                    self.assertTrue(kinds.issubset(set(result["kinds"])))

    def test_parameterized_constructor_names_and_external_definitions_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nfg = Path(directory) / "fixture.nfg"
            nfg.write_text(pyspiel.game_to_nfg_string(pyspiel.load_game("matrix_rps")))
            efg = Path(directory) / "fixture.efg"
            efg.write_text(pyspiel.get_kuhn_poker_efg_data())
            cases = [
                f"nfg_game(filename={nfg})",
                f"efg_game(filename={efg})",
                "normal_form_extensive_game(game=matrix_rps())",
                "restricted_nash_response(game=kuhn_poker(),fixed_player=0,p=0.5)",
                "repeated_poker(universal_poker_game_string=universal_poker(),max_num_hands=2,reset_stacks=false,rotate_dealer=true)",
            ]
            for spec in cases:
                with self.subTest(spec=spec):
                    result = conformance._check_episode(spec, max_steps=32, seed=19)
                    self.assertEqual(result["status"], "passed")
                    self.assertTrue(result["replay_verified"])

    def test_isolated_worker_returns_bounded_prefix(self) -> None:
        result = conformance.check_game("chess", max_steps=1, seed=2)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["terminal"])
        self.assertEqual(result["verification_scope"], "bounded_prefix")
        self.assertEqual(result["num_steps"], 1)

    def test_timeout_is_reported_without_stopping_other_checks(self) -> None:
        with patch.object(
            subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 0.1)
        ):
            result = conformance.check_game("tic_tac_toe", timeout=0.1)
        self.assertEqual(result["status"], "timeout")

    def test_invalid_limits_are_rejected(self) -> None:
        for kwargs in [{"max_steps": 0}, {"timeout": -1}, {"seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                conformance.check_game("tic_tac_toe", **kwargs)
        with self.assertRaises(ValueError):
            conformance.run_conformance(["tic_tac_toe"], workers=0)

    def test_explicit_unsupported_game_is_skipped(self) -> None:
        result = conformance.check_game("chat_game")
        self.assertEqual(result["status"], "skipped")
        self.assertIn("external", result["reason"])


if __name__ == "__main__":
    unittest.main()
