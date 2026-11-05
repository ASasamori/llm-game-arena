"""Public displays retain engine evidence and never inspect secret states."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
import unittest

import pyspiel

from game_arena.openspiel.presentation import describe_state


class _DisplayOverride:
    def __init__(self, state: Any, text: str):
        self.state, self.text = state, text

    def __getattr__(self, name: str) -> Any:
        return getattr(self.state, name)

    def __str__(self) -> str:
        return self.text


class _HiddenGuard(_DisplayOverride):
    def __str__(self) -> str:
        raise AssertionError("A spectator tried to read hidden evaluator state")

    def observation_string(self, player: int) -> str:
        raise AssertionError("A spectator tried to read a private observation")


class PresentationTests(unittest.TestCase):
    def test_common_board_families_preserve_exact_text_before_and_after_moves(
        self,
    ) -> None:
        cases = {
            "tic_tac_toe": (3, 3),
            "mnk(m=5,n=4,k=3)": (4, 5),
            "connect_four(rows=5,columns=8)": (5, 8),
            "catch(rows=6,columns=4)": (6, 4),
            "chess": (8, 8),
            "antichess": (8, 8),
            "go(board_size=5)": (5, 5),
            "cursor_go(board_size=5)": (5, 5),
            "hex(num_rows=3,num_cols=5)": (3, 5),
            "othello": (8, 8),
            "checkers": (8, 8),
            "breakthrough(rows=5,columns=5)": (5, 5),
            "clobber(rows=4,columns=7)": (4, 7),
            "lines_of_action": (8, 8),
            "amazons": (10, 10),
            "pentago": (6, 6),
            "2048": (4, 4),
            "ultimate_tic_tac_toe": (9, 9),
            "pathfinding": (3, 7),
            "cliff_walking": (4, 8),
            "markov_soccer": (4, 5),
            "stones_and_gems": (12, 20),
        }
        for spec, shape in cases.items():
            with self.subTest(spec=spec):
                state = pyspiel.load_game(spec).new_initial_state()
                for _ in range(4):
                    original = state.serialize()
                    view = describe_state(state)
                    assert view is not None
                    self.assertTrue(view["visibility"]["public"])
                    self.assertEqual(view["board"]["text"], str(state))
                    if spec == "stones_and_gems" and state.is_chance_node():
                        self.assertEqual(view["board"]["format"], "text")
                        state.apply_action(state.legal_actions()[0])
                        continue
                    self.assertEqual(view["board"]["format"], "grid")
                    self.assertEqual(len(view["board"]["rows"]), shape[0])
                    self.assertTrue(
                        all(len(row) == shape[1] for row in view["board"]["rows"])
                    )
                    self.assertEqual(state.serialize(), original)
                    self.assertEqual(json.loads(json.dumps(view)), view)
                    self.assertEqual(len(view["board"]["row_labels"]), shape[0])
                    self.assertEqual(len(view["board"]["column_labels"]), shape[1])
                    if state.is_terminal() or state.is_simultaneous_node():
                        break
                    state.apply_action(state.legal_actions()[0])

    def test_coordinates_and_ownership_follow_each_engines_notation(self) -> None:
        chess = describe_state(pyspiel.load_game("chess").new_initial_state())
        assert chess is not None
        self.assertEqual(chess["node"]["current_player"], 1)
        self.assertEqual(chess["board"]["row_labels"], list("87654321"))
        self.assertEqual(chess["board"]["rows"][0], list("rnbqkbnr"))
        self.assertIn("Player 1", chess["board"]["legend"]["P"])
        self.assertIn("Player 0", chess["board"]["legend"]["p"])

        go = describe_state(pyspiel.load_game("go(board_size=9)").new_initial_state())
        assert go is not None
        self.assertEqual(go["board"]["column_labels"], list("ABCDEFGHJ"))
        self.assertEqual(go["board"]["row_labels"], list("987654321"))
        self.assertEqual(go["board"]["empty_symbols"], ["+"])

        # A leading king digit must not be swallowed by the rank label.
        checkers = pyspiel.load_game("checkers").new_initial_state()
        text = str(checkers).replace("8.+.+.+.+", "88+.+.+.+")
        view = describe_state(_DisplayOverride(checkers, text))
        assert view is not None
        self.assertEqual(view["board"]["format"], "grid")
        self.assertEqual(view["board"]["row_labels"][0], "8")
        self.assertEqual(view["board"]["rows"][0][0], "8")
        self.assertEqual(view["board"]["legend"]["8"], "Player 0 king")

        clobber = describe_state(pyspiel.load_game("clobber").new_initial_state())
        assert clobber is not None
        self.assertEqual(clobber["board"]["legend"]["o"], "Player 0")

    def test_hidden_states_never_read_full_state_even_after_terminal(self) -> None:
        for spec in (
            "kuhn_poker",
            "leduc_poker",
            "dark_hex(board_size=3)",
            "phantom_ttt",
        ):
            with self.subTest(spec=spec):
                state = pyspiel.load_game(spec).new_initial_state()
                for _ in range(300):
                    view = describe_state(_HiddenGuard(state, "SECRET"))
                    assert view is not None
                    self.assertFalse(view["visibility"]["public"])
                    self.assertEqual(
                        view["board"],
                        {
                            "format": "unavailable",
                            "source": "unavailable",
                            "text": None,
                        },
                    )
                    self.assertNotIn("SECRET", json.dumps(view))
                    if state.is_terminal():
                        break
                    state.apply_action(state.legal_actions()[0])
                self.assertTrue(state.is_terminal())

    def test_unknown_layouts_and_large_boards_retain_every_character(self) -> None:
        for spec, text in (
            ("tic_tac_toe", "x..\n.o.\n?..\nnew annotation <tag>\x1b[31m"),
            ("chess", "future chess state layout: ♞ & castling=custom"),
            ("hex(board_size=3)", ". . .\n . . .\n  ? . ."),
        ):
            with self.subTest(spec=spec):
                view = describe_state(
                    _DisplayOverride(pyspiel.load_game(spec).new_initial_state(), text)
                )
                assert view is not None
                self.assertEqual(view["board"]["format"], "text")
                self.assertEqual(view["board"]["text"], text)
        state = pyspiel.load_game("mnk(m=65,n=65,k=5)").new_initial_state()
        view = describe_state(state)
        assert view is not None
        self.assertEqual(view["board"]["format"], "text")
        self.assertEqual(view["board"]["text"], str(state))

    def test_public_nonboard_games_and_simultaneous_actors_have_explicit_nodes(
        self,
    ) -> None:
        for spec in (
            "nim",
            "pig",
            "mancala",
            "backgammon",
            "matrix_rps",
            "matching_pennies_3p",
        ):
            with self.subTest(spec=spec):
                state = pyspiel.load_game(spec).new_initial_state()
                view = describe_state(state)
                assert view is not None
                self.assertEqual(view["board"]["format"], "text")
                self.assertEqual(view["board"]["text"], str(state))
                if state.is_simultaneous_node():
                    self.assertEqual(view["node"]["kind"], "simultaneous")
                    self.assertEqual(
                        view["node"]["active_players"],
                        list(range(state.get_game().num_players())),
                    )
                    state.apply_actions(
                        [
                            state.legal_actions(p)[0]
                            for p in range(state.get_game().num_players())
                        ]
                    )
                    terminal = describe_state(state)
                    assert terminal is not None
                    self.assertTrue(terminal["node"]["terminal"])
                    self.assertEqual(terminal["node"]["kind"], "terminal")
                    self.assertEqual(terminal["node"]["active_players"], [])

    def test_mean_field_nodes_and_native_opt_out_are_not_player_decisions(self) -> None:
        state = pyspiel.load_game("mfg_crowd_modelling").new_initial_state()
        for _ in range(10):
            view = describe_state(state)
            assert view is not None
            if state.is_mean_field_node():
                self.assertEqual(view["node"]["kind"], "mean_field")
                self.assertEqual(view["node"]["active_players"], [])
                break
            if state.is_chance_node():
                self.assertEqual(view["node"]["kind"], "chance")
                self.assertEqual(view["node"]["active_players"], [])
            state.apply_action(state.legal_actions()[0])
        else:
            self.fail("Mean field fixture never reached distribution update")
        native = SimpleNamespace(
            get_game=lambda: SimpleNamespace(environment_backend="native")
        )
        self.assertIsNone(describe_state(native))


if __name__ == "__main__":
    unittest.main()
