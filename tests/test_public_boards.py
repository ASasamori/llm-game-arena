"""Spectator boards expose resolved facts and remain verifiable audit evidence."""

from __future__ import annotations

from copy import deepcopy
import unittest
from typing import Any

from game_arena.environments.avalon import AvalonGame
from game_arena.environments.coup import CoupGame
from game_arena.environments.werewolf import WerewolfGame
from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.replay import ReplayError, replay_episode


def deal(state: Any) -> None:
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])


class PublicBoardPrivacyTests(unittest.TestCase):
    def test_avalon_pending_secret_votes_are_identical_until_public_resolution(
        self,
    ) -> None:
        state = AvalonGame({"discussion_rounds": 0}).new_initial_state()
        deal(state)
        state.apply_action(state.legal_actions()[0])
        self.assertEqual(state.phase, "vote")
        approval = state.clone()
        rejection = state.clone()
        approval.apply_action(1)
        rejection.apply_action(0)
        self.assertEqual(approval.public_state(), rejection.public_state())
        self.assertIsNone(approval.public_state()["last_team_vote"])
        for _ in range(4):
            approval.apply_action(1)
        self.assertEqual(approval.phase, "quest")
        board = approval.public_state()
        self.assertEqual(board["last_team_vote"]["votes"], [True] * 5)
        self.assertNotIn("votes", board)
        self.assertNotIn("quest_votes", board)
        self.assertNotIn("roles", board)
        self.assertNotIn("role", board["players"][0])

        # An evil quest card and a success card produce the same public board
        # until the entire quest resolves; only the aggregate may then differ.
        first = approval.proposal[0]
        assassin = approval.roles.index("assassin")
        approval.roles[first], approval.roles[assassin] = (
            approval.roles[assassin],
            approval.roles[first],
        )
        failed_card = approval.clone()
        failed_card.apply_action(1)
        approval.apply_action(0)
        self.assertEqual(approval.public_state(), failed_card.public_state())
        approval.apply_action(0)
        failed_card.apply_action(0)
        self.assertTrue(approval.public_state()["quest_results"][0]["success"])
        self.assertFalse(failed_card.public_state()["quest_results"][0]["success"])
        self.assertEqual(failed_card.public_state()["quest_results"][0]["fails"], 1)

    def test_public_boards_are_snapshots_and_cannot_mutate_game_state(self) -> None:
        state = AvalonGame({"discussion_rounds": 0}).new_initial_state()
        deal(state)
        state.apply_action(0)
        board = state.public_state()
        previous = deepcopy(board)
        state.apply_action(1)
        self.assertEqual(board, previous)
        board["proposed_team"].append(99)
        board["public_history"][0]["team"].append(98)
        self.assertNotIn(99, state.proposal)
        self.assertNotIn(98, state.public_history[0]["team"])

    def test_coup_hidden_hands_exchange_and_court_never_enter_public_board(
        self,
    ) -> None:
        state = CoupGame().new_initial_state()
        deal(state)
        original = state.public_state()
        state.hands = [["contessa", "captain"] for _ in state.hands]
        state.court = [0, 1, 2, 3, 4]
        state.exchange_pool = ["duke", "assassin"]
        state.private_events[0].append({"event": "private", "role": "duke"})
        self.assertEqual(state.public_state(), original)
        self.assertEqual(original["players"][0]["influence_count"], 2)
        for field in ("hands", "court", "exchange_pool", "own_influence"):
            self.assertNotIn(field, original)

    def test_board_history_stays_bounded_without_deleting_audit_history(self) -> None:
        for game, field in (
            (AvalonGame(), "public_history"),
            (CoupGame(), "history"),
            (WerewolfGame(), "public_transcript"),
        ):
            state = game.new_initial_state()
            history = [{"kind": "discussion", "number": index} for index in range(75)]
            setattr(state, field, deepcopy(history))
            board = state.public_state()
            assert board is not None
            self.assertEqual(board["public_history"], history[-32:])
            self.assertEqual(board["public_history_total"], 75)
            self.assertEqual(getattr(state, field), history)

    def test_werewolf_night_hides_actor_order_and_reveals_only_at_terminal(
        self,
    ) -> None:
        state = WerewolfGame().new_initial_state()
        deal(state)
        self.assertEqual(state.phase, "night")
        original = state.public_state()
        self.assertIsNone(original["current_player"])
        state.cards = list(reversed(state.cards))
        state.original_roles = list(reversed(state.original_roles))
        state.private_findings[0].append({"role": "seer"})
        self.assertEqual(state.public_state(), original)
        for field in ("final_roles", "center_cards", "votes", "private_findings"):
            self.assertNotIn(field, original)
        state.phase = "terminal"
        state.dead = [1]
        state.winners = [0, 2]
        result = state.public_state()
        self.assertEqual(result["final_roles"], state.cards[:3])
        self.assertEqual(result["players"][1]["role"], state.cards[1])
        self.assertTrue(result["players"][1]["dead"])


class PublicBoardReplayTests(unittest.TestCase):
    def test_mock_runs_record_boards_replay_and_reject_tampered_boards(self) -> None:
        for game, players in (("avalon", 5), ("coup", 3), ("one_night_werewolf", 3)):
            with self.subTest(game=game):
                options: Any = {
                    "type": "PVP",
                    "game": game,
                    "models": [MockModel(str(seat)) for seat in range(players)],
                    "num_steps": 2000,
                    "input": None,
                    "seed": 31,
                    "failure_policy": "raise",
                }
                record: dict[str, Any] = dict(harness(options))
                self.assertTrue(replay_episode(record)["verified"])
                self.assertTrue(record["terminal"])
                for index, event in enumerate(record["steps"]):
                    self.assertEqual(event["public_state_before"]["game"], game)
                    self.assertEqual(event["public_state_after"]["game"], game)
                    if index:
                        self.assertEqual(
                            event["public_state_before"],
                            record["steps"][index - 1]["public_state_after"],
                        )
                for field in ("public_state_before", "public_state_after"):
                    corrupt = deepcopy(record)
                    corrupt["steps"][0][field]["phase"] = "invented"
                    with self.assertRaisesRegex(ReplayError, "public board mismatch"):
                        replay_episode(corrupt)
                # Old traces remain valid without newly optional public boards.
                legacy = deepcopy(record)
                for event in legacy["steps"]:
                    event.pop("public_state_before")
                    event.pop("public_state_after")
                self.assertTrue(replay_episode(legacy)["verified"])


if __name__ == "__main__":
    unittest.main()
