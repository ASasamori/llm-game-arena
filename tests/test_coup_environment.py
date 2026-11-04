"""Exercise Coup rules directly, with no harness, model, provider or network."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import random
from typing import Any, cast
import unittest

from game_arena.environments.coup import (
    ASSASSINATE,
    BLOCK_AMBASSADOR,
    BLOCK_CAPTAIN,
    BLOCK_CONTESSA,
    BLOCK_DUKE,
    CHALLENGE,
    CONCEDE,
    COUP,
    EXCHANGE,
    FOREIGN_AID,
    INCOME,
    LOSE,
    PASS,
    PROVE,
    RETURN,
    ROLES,
    STEAL,
    TAX,
    CoupGame,
    CoupState,
)


def dealt(hands: list[list[str]], **parameters: Any) -> CoupState:
    state = CoupGame({"players": len(hands), **parameters}).new_initial_state()
    for hand in hands:
        if len(hand) != 2:
            raise ValueError("A scripted initial hand must have two cards")
        for role in hand:
            state.apply_action(ROLES.index(role))
    return state


def pass_window(state: CoupState) -> None:
    while state.phase in {"action_challenge", "block", "block_challenge"}:
        state.apply_action(PASS)


def draw(state: CoupState, role: str | None = None) -> None:
    state.apply_action(ROLES.index(role) if role else state.legal_actions()[0])


def card_inventory(state: CoupState) -> Counter[str]:
    cards: Counter[str] = Counter(
        {role: state.court[i] for i, role in enumerate(ROLES)}
    )
    for player, hand in enumerate(state.hands):
        if player == state.turn_player and state.phase in {
            "exchange_draw",
            "exchange_return",
        }:
            cards.update(state.exchange_pool)
        else:
            cards.update(hand)
        cards.update(state.revealed[player])
    return cards


class CoupEnvironmentTests(unittest.TestCase):
    def test_parameters_and_standard_two_player_setup(self) -> None:
        for parameters in [
            {"players": 1},
            {"players": 7},
            {"players": True},
            {"players": 3.0},
            {"players": "3"},
            {"starting_player": -1},
            {"starting_player": 3},
            {"max_turns": 0},
            {"max_turns": True},
            {"max_turns": 10001},
            {"roles": "duke"},
            {"reformation": True},
        ]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                CoupGame(parameters)
        game = CoupGame({"players": 2, "starting_player": 1, "max_turns": 30})
        state = game.new_initial_state()
        self.assertEqual(state.coins, [2, 1])
        self.assertEqual(
            game.parameters, {"players": 2, "starting_player": 1, "max_turns": 30}
        )
        self.assertEqual(game.num_players(), 2)
        self.assertGreater(game.max_history_length(), 30)
        self.assertEqual(CoupGame().new_initial_state().coins, [2, 2, 2])

    def test_explicit_weighted_dealing_and_illegal_chance_is_atomic(self) -> None:
        state = CoupGame().new_initial_state()
        self.assertTrue(state.is_chance_node())
        self.assertEqual(state.chance_outcomes(), [(i, 0.2) for i in range(5)])
        state.apply_action(0)
        self.assertEqual(dict(state.chance_outcomes())[0], 2 / 14)
        state.apply_action(0)
        state.apply_action(0)
        self.assertNotIn(0, state.legal_actions())
        before = state.serialize()
        with self.assertRaises(ValueError):
            state.apply_action(0)
        self.assertEqual(state.serialize(), before)

    def test_private_observations_do_not_expose_other_hands_or_court(self) -> None:
        first = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        second = dealt(
            [["duke", "contessa"], ["ambassador", "duke"], ["assassin", "captain"]]
        )
        self.assertEqual(first.observation_string(0), second.observation_string(0))
        view = json.loads(first.observation_string(0))
        self.assertEqual(view["own_influence"], ["duke", "contessa"])
        self.assertNotIn("hands", view)
        self.assertNotIn("court", view)
        self.assertNotIn("private_events", view)
        self.assertEqual(view["influence_counts"], [2, 2, 2])
        with self.assertRaises(ValueError):
            first.observation_string(-1)
        with self.assertRaises(ValueError):
            first.observation_string(3)

    def test_action_choices_allow_bluffs_independent_of_hidden_role(self) -> None:
        first = dealt([["duke", "duke"], ["assassin", "contessa"]])
        second = dealt([["captain", "ambassador"], ["assassin", "contessa"]])
        self.assertEqual(first.legal_actions(), second.legal_actions())
        self.assertIn(TAX, second.legal_actions())
        self.assertIn(EXCHANGE, first.legal_actions())
        self.assertEqual(first.legal_actions(1), [])

    def test_income_is_immediate_and_rotates_starting_player(self) -> None:
        state = dealt([["duke", "contessa"], ["assassin", "captain"]])
        state.apply_action(INCOME)
        self.assertEqual(state.coins, [2, 2])
        self.assertEqual(state.phase, "turn")
        self.assertEqual(state.current_player(), 1)
        self.assertEqual(state.completed_turns, 1)

    def test_foreign_aid_cannot_be_challenged_and_can_be_bluff_blocked(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.apply_action(FOREIGN_AID)
        self.assertEqual(state.phase, "block")
        self.assertEqual(state.current_player(), 1)
        self.assertNotIn(CHALLENGE, state.legal_actions())
        self.assertIn(BLOCK_DUKE, state.legal_actions())
        state.apply_action(BLOCK_DUKE)
        self.assertEqual(state.phase, "block_challenge")
        pass_window(state)
        self.assertEqual(state.coins[0], 2)
        self.assertEqual(state.current_player(), 1)

    def test_foreign_aid_without_blocks_gains_two(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.apply_action(FOREIGN_AID)
        pass_window(state)
        self.assertEqual(state.coins[0], 4)

    def test_false_tax_challenge_cancels_action_and_defender_chooses_loss(self) -> None:
        state = dealt(
            [["captain", "contessa"], ["assassin", "duke"], ["ambassador", "duke"]]
        )
        state.apply_action(TAX)
        state.apply_action(CHALLENGE)
        self.assertEqual(state.current_player(), 0)
        self.assertEqual(state.legal_actions(), [CONCEDE])
        state.apply_action(CONCEDE)
        state.apply_action(LOSE + 1)
        self.assertEqual(state.revealed[0], ["contessa"])
        self.assertEqual(state.coins[0], 2)
        self.assertEqual(state.current_player(), 1)

    def test_true_tax_replaces_proved_card_and_challenger_loses(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.apply_action(TAX)
        state.apply_action(CHALLENGE)
        self.assertIn(CONCEDE, state.legal_actions())
        self.assertIn(PROVE, state.legal_actions())
        state.apply_action(PROVE)
        self.assertTrue(state.is_chance_node())
        self.assertEqual(
            json.loads(state.observation_string(1))["influence_counts"], [2, 2, 2]
        )
        draw(state, "captain")
        self.assertEqual(state.current_player(), 1)
        state.apply_action(LOSE)
        self.assertEqual(state.hands[0], ["contessa", "captain"])
        self.assertEqual(len(state.hands[1]), 1)
        self.assertEqual(state.coins[0], 5)
        self.assertEqual(card_inventory(state), Counter({r: 3 for r in ROLES}))

    def test_can_concede_truthful_claim_voluntarily(self) -> None:
        state = dealt([["duke", "contessa"], ["assassin", "captain"]])
        state.apply_action(TAX)
        state.apply_action(CHALLENGE)
        state.apply_action(CONCEDE)
        state.apply_action(LOSE)
        self.assertEqual(state.revealed[0], ["duke"])
        self.assertEqual(state.coins[0], 1)

    def test_assassination_successful_challenge_refunds_cost(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.coins[0] = 5
        state.apply_action(ASSASSINATE + 1)
        self.assertEqual(state.coins[0], 2)
        state.apply_action(CHALLENGE)
        state.apply_action(CONCEDE)
        state.apply_action(LOSE)
        self.assertEqual(state.coins[0], 5)
        self.assertEqual(len(state.hands[1]), 2)

    def test_assassination_unchallenged_block_keeps_cost_spent(self) -> None:
        state = dealt(
            [["assassin", "duke"], ["contessa", "captain"], ["ambassador", "duke"]]
        )
        state.coins[0] = 5
        state.apply_action(ASSASSINATE + 1)
        state.apply_action(PASS)
        state.apply_action(PASS)
        self.assertEqual(state.phase, "block")
        self.assertEqual(state.current_player(), 1)
        state.apply_action(BLOCK_CONTESSA)
        pass_window(state)
        self.assertEqual(state.coins[0], 2)
        self.assertEqual(len(state.hands[1]), 2)

    def test_failed_assassin_challenge_can_cost_both_influences(self) -> None:
        state = dealt(
            [["assassin", "duke"], ["contessa", "captain"], ["ambassador", "duke"]]
        )
        state.coins[0] = 3
        state.apply_action(ASSASSINATE + 1)
        state.apply_action(CHALLENGE)
        state.apply_action(PROVE)
        draw(state, "assassin")
        state.apply_action(LOSE)  # Target loses Contessa from failed challenge.
        self.assertEqual(state.phase, "block")
        state.apply_action(PASS)
        self.assertEqual(state.phase, "lose")
        state.apply_action(LOSE)
        self.assertEqual(state.hands[1], [])
        self.assertEqual(len(state.revealed[1]), 2)
        self.assertEqual(state.current_player(), 2)

    def test_false_contessa_block_can_cost_both_influences(self) -> None:
        state = dealt(
            [["assassin", "duke"], ["ambassador", "captain"], ["contessa", "duke"]]
        )
        state.coins[0] = 3
        state.apply_action(ASSASSINATE + 1)
        state.apply_action(PASS)
        state.apply_action(PASS)
        state.apply_action(BLOCK_CONTESSA)
        self.assertEqual(
            state.current_player(), 2
        )  # Third parties can challenge blocks.
        state.apply_action(CHALLENGE)
        state.apply_action(CONCEDE)
        state.apply_action(LOSE)
        self.assertEqual(state.phase, "lose")
        state.apply_action(LOSE)
        self.assertEqual(state.hands[1], [])
        self.assertEqual(state.coins[0], 0)

    def test_truthful_block_replacement_hides_new_card_from_opponents(self) -> None:
        state = dealt(
            [["assassin", "duke"], ["contessa", "captain"], ["ambassador", "duke"]]
        )
        state.coins[0] = 3
        state.apply_action(ASSASSINATE + 1)
        state.apply_action(PASS)
        state.apply_action(PASS)
        state.apply_action(BLOCK_CONTESSA)
        state.apply_action(CHALLENGE)
        state.apply_action(PROVE)
        alternative = cast(CoupState, state.clone())
        draw(state, "assassin")
        draw(alternative, "ambassador")
        self.assertEqual(state.observation_string(0), alternative.observation_string(0))
        self.assertNotEqual(
            state.observation_string(1), alternative.observation_string(1)
        )
        state.apply_action(LOSE)
        self.assertEqual(len(state.hands[1]), 2)
        self.assertEqual(len(state.hands[2]), 1)
        self.assertEqual(state.coins[0], 0)

    def test_coup_is_unblockable_and_ten_coins_force_it(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.coins[0] = 10
        self.assertEqual(state.legal_actions(), [COUP + 1, COUP + 2])
        before = state.serialize()
        with self.assertRaises(ValueError):
            state.apply_action(INCOME)
        self.assertEqual(state.serialize(), before)
        state.apply_action(COUP + 2)
        self.assertEqual(state.phase, "lose")
        self.assertEqual(state.current_player(), 2)
        self.assertEqual(state.coins[0], 3)
        state.apply_action(LOSE + 1)
        self.assertEqual(state.revealed[2], ["duke"])

    def test_steal_takes_only_available_coins_and_has_both_blocks(self) -> None:
        for block in (BLOCK_CAPTAIN, BLOCK_AMBASSADOR):
            state = dealt(
                [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
            )
            state.coins[1] = 1
            state.apply_action(STEAL + 1)
            state.apply_action(PASS)
            state.apply_action(PASS)
            self.assertIn(block, state.legal_actions())
            state.apply_action(PASS)
            self.assertEqual(state.coins[:2], [3, 0])

    def test_exchange_draw_and_return_are_private_and_preserve_hand_size(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.apply_action(EXCHANGE)
        pass_window(state)
        self.assertEqual(state.phase, "exchange_draw")
        draw(state, "captain")
        alternate = cast(CoupState, state.clone())
        draw(state, "assassin")
        draw(alternate, "ambassador")
        self.assertEqual(state.observation_string(1), alternate.observation_string(1))
        self.assertNotEqual(
            state.observation_string(0), alternate.observation_string(0)
        )
        self.assertEqual(len(state.exchange_pool), 4)
        self.assertEqual(len(state.legal_actions()), 4)
        state.apply_action(RETURN)
        state.apply_action(RETURN)
        self.assertEqual(state.hands[0], ["captain", "assassin"])
        self.assertEqual(card_inventory(state), Counter({r: 3 for r in ROLES}))
        self.assertEqual(state.current_player(), 1)

    def test_exchange_with_one_influence_keeps_only_one(self) -> None:
        state = dealt(
            [["duke", "contessa"], ["assassin", "captain"], ["ambassador", "duke"]]
        )
        state.revealed[0].append(state.hands[0].pop())
        state.apply_action(EXCHANGE)
        pass_window(state)
        draw(state)
        draw(state)
        self.assertEqual(len(state.exchange_pool), 3)
        state.apply_action(RETURN)
        state.apply_action(RETURN)
        self.assertEqual(len(state.hands[0]), 1)
        self.assertEqual(card_inventory(state), Counter({r: 3 for r in ROLES}))

    def test_elimination_returns_coins_skips_turn_and_last_survivor_wins(self) -> None:
        state = dealt([["duke", "contessa"], ["assassin", "captain"]])
        state.revealed[1].append(state.hands[1].pop())
        state.coins[0] = 7
        state.apply_action(COUP + 1)
        state.apply_action(LOSE)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.coins[1], 0)
        self.assertEqual(state.returns(), [1.0, -1.0])
        self.assertEqual(state.rewards(), state.returns())
        self.assertEqual(state.finish_reason, "last_survivor")
        self.assertEqual(state.legal_actions(), [])

    def test_turn_cap_is_a_draw_not_a_survivor_win(self) -> None:
        state = dealt([["duke", "contessa"], ["assassin", "captain"]], max_turns=1)
        state.apply_action(INCOME)
        self.assertTrue(state.is_terminal())
        self.assertIsNone(state.winner)
        self.assertEqual(state.finish_reason, "turn_limit_draw")
        self.assertEqual(state.returns(), [0.0, 0.0])

    def test_clone_is_independent_and_invalid_inputs_never_mutate(self) -> None:
        state = dealt([["duke", "contessa"], ["assassin", "captain"]])
        clone = cast(CoupState, state.clone())
        original = state.serialize()
        clone.apply_action(INCOME)
        self.assertEqual(state.serialize(), original)
        self.assertNotEqual(state.serialize(), clone.serialize())
        for action in (True, -1, 999, "0", COUP + 1):
            with self.subTest(action=action), self.assertRaises(ValueError):
                state.apply_action(action)  # type: ignore[arg-type]
            self.assertEqual(state.serialize(), original)
        self.assertEqual(
            json.loads(state.serialize()), json.loads(deepcopy(state).serialize())
        )

    def test_model_free_random_games_preserve_cards_and_terminate(self) -> None:
        for players in range(2, 7):
            for seed in range(12):
                with self.subTest(players=players, seed=seed):
                    game = CoupGame(
                        {
                            "players": players,
                            "max_turns": 30,
                            "starting_player": seed % players,
                        }
                    )
                    state = game.new_initial_state()
                    rng = random.Random(seed)
                    actions: list[int] = []
                    while not state.is_terminal():
                        legal = state.legal_actions()
                        self.assertTrue(legal)
                        if state.is_chance_node():
                            outcomes = state.chance_outcomes()
                            self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)
                            action = rng.choices(
                                [a for a, _ in outcomes],
                                weights=[p for _, p in outcomes],
                            )[0]
                        else:
                            action = rng.choice(legal)
                        before = state.serialize()
                        with self.assertRaises(ValueError):
                            state.apply_action(99999)
                        self.assertEqual(state.serialize(), before)
                        state.apply_action(action)
                        actions.append(action)
                        self.assertEqual(
                            card_inventory(state), Counter({r: 3 for r in ROLES})
                        )
                        self.assertTrue(all(coins >= 0 for coins in state.coins))
                        self.assertLessEqual(len(actions), game.max_history_length())
                    replayed = game.new_initial_state()
                    for action in actions:
                        replayed.apply_action(action)
                    self.assertEqual(replayed.serialize(), state.serialize())
                    self.assertEqual(len(state.returns()), players)

    def test_first_legal_baseline_reaches_actual_winner(self) -> None:
        for players in range(2, 7):
            state = CoupGame({"players": players}).new_initial_state()
            for _ in range(state.game.max_history_length()):
                if state.is_terminal():
                    break
                state.apply_action(state.legal_actions()[0])
            self.assertTrue(state.is_terminal())
            self.assertEqual(state.finish_reason, "last_survivor")
            self.assertEqual(state.returns().count(1.0), 1)


if __name__ == "__main__":
    unittest.main()
