"""Registry discovery, parameter safety, and native RNG reproducibility."""

import unittest
from unittest.mock import patch

import pyspiel

from game_arena.openspiel import catalog


class CatalogTests(unittest.TestCase):
    def test_registry_discovery_does_not_instantiate_games(self) -> None:
        catalog.register_python_games()
        with patch.object(
            pyspiel, "load_game", side_effect=AssertionError("must not load")
        ):
            rows = catalog.list_games()
        entries = {row["short_name"]: row for row in rows}
        self.assertIn("tic_tac_toe", entries)
        self.assertIn("python_kuhn_poker", entries)
        self.assertFalse(entries["tic_tac_toe"]["verified"])
        self.assertIn(
            "explicit_game_parameters", entries["repeated_game"]["requirements"]
        )

    def test_aliases_and_parameterized_names(self) -> None:
        self.assertEqual(catalog.resolve_game_spec("CONNECT_FOUR_V1"), "connect_four")
        info = catalog.inspect_game("kuhn_poker(players=3)")
        self.assertEqual(info["num_players"], 3)
        self.assertEqual(info["canonical_game"], "kuhn_poker(players=3)")
        self.assertEqual(info["information"], "IMPERFECT_INFORMATION")
        self.assertTrue(info["supported"])

    def test_unknown_malformed_and_missing_parameters_rejected(self) -> None:
        for spec in ["", "made_up_game", "tic_tac_toe(foo", "repeated_game"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                catalog.inspect_game(spec)

    def test_invalid_constructor_is_contained(self) -> None:
        with self.assertRaises(catalog.UnsupportedGameError):
            catalog.inspect_game("kuhn_poker(players=999)")
        self.assertEqual(catalog.inspect_game("tic_tac_toe")["num_players"], 2)

    def test_sampled_rng_seed_and_canonical_reload(self) -> None:
        canonical = catalog.inspect_game("negotiation", seed=117)["canonical_game"]
        self.assertIn("rng_seed=117", canonical)
        first = catalog.load_game(canonical, seed=999).new_initial_state()
        second = catalog.load_game(canonical, seed=1).new_initial_state()
        for _ in range(5):
            if first.is_terminal():
                break
            self.assertEqual(first.to_string(), second.to_string())
            if first.is_chance_node():
                action = first.chance_outcomes()[0][0]
            else:
                action = first.legal_actions()[0]
            first.apply_action(action)
            second.apply_action(action)
        self.assertEqual(first.to_string(), second.to_string())

    def test_nested_native_rng_is_seeded(self) -> None:
        game = catalog.load_game("misere(game=tarok())", seed=42)
        self.assertIn("rng_seed=42", game.to_string())
        self.assertEqual(
            game.to_string(), catalog.load_game(game.to_string(), seed=9).to_string()
        )

    def test_mean_field_is_conditional_and_inspectable(self) -> None:
        info = catalog.inspect_game("mfg_crowd_modelling")
        self.assertTrue(info["supported"])
        self.assertEqual(info["num_players"], 1)
        self.assertIn("mean_field_distribution", info["requirements"])

    def test_external_chat_environment_is_never_loaded(self) -> None:
        with patch.object(
            pyspiel, "load_game", side_effect=AssertionError("external game loaded")
        ):
            info = catalog._inspect_in_process("chat_game", 0)
            with self.assertRaises(catalog.UnsupportedGameError):
                catalog.load_game("chat_game")
        self.assertFalse(info["supported"])
        self.assertIn("external", info["unsupported_reasons"][0])

    def test_unsafe_imperfect_information_game_has_clear_reason(self) -> None:
        # Metadata differs across OpenSpiel versions; only builds without an
        # observation implementation should be refused.
        info = catalog.inspect_game("cribbage")
        if not any(info["observations"].values()):
            self.assertFalse(info["supported"])
            with self.assertRaises(catalog.UnsupportedGameError):
                catalog.load_game("cribbage")

    def test_cached_inspection_returns_independent_results(self) -> None:
        first = catalog.inspect_game("tic_tac_toe")
        first["requirements"].append("mutated")
        self.assertNotIn("mutated", catalog.inspect_game("tic_tac_toe")["requirements"])


if __name__ == "__main__":
    unittest.main()
