"""Rule/parameter/privacy checks that never import a model or OpenSpiel."""

from __future__ import annotations

from collections import Counter
import json
import random
from typing import Any
import unittest

from game_arena.environments.base import CHANCE, TERMINAL
from game_arena.environments.werewolf import ROLE_IDS, WerewolfGame, WerewolfState


def dealt(players: list[str], center: list[str], **parameters: Any) -> WerewolfState:
    game = WerewolfGame(
        {"players": len(players), "roles": players + center, "discussion_rounds": 0}
        | parameters
    )
    state = game.new_initial_state()
    for role in players + center:
        state.apply_action(ROLE_IDS[role])
    return state


def action(state: WerewolfState, label: str) -> None:
    choices = [
        a
        for a in state.legal_actions()
        if state.action_to_string(state.current_player(), a) == label
    ]
    if len(choices) != 1:
        raise AssertionError(f"Missing/ambiguous action {label}: {state._actions()}")
    state.apply_action(choices[0])


def skip_night(state: WerewolfState) -> None:
    while state.phase == "night":
        state.apply_action(state.legal_actions()[0])


def vote(state: WerewolfState, targets: list[int]) -> None:
    if state.phase != "vote":
        raise AssertionError(f"Expected voting, got {state.phase}")
    for target in targets:
        state.apply_action(1000 + target)


class WerewolfEnvironmentTests(unittest.TestCase):
    def test_default_parameters_and_every_player_count_finish_without_ai(self) -> None:
        for players in range(3, 11):
            for rounds in range(4):
                for seed in range(12):
                    with self.subTest(players=players, rounds=rounds, seed=seed):
                        game = WerewolfGame(
                            {"players": players, "discussion_rounds": rounds}
                        )
                        state = game.new_initial_state()
                        rng = random.Random(seed)
                        transitions = 0
                        while not state.is_terminal():
                            legal = state.legal_actions()
                            self.assertTrue(legal)
                            self.assertEqual(legal, sorted(set(legal)))
                            self.assertEqual(len(state.returns()), players)
                            self.assertEqual(state.returns(), [0.0] * players)
                            if state.is_chance_node():
                                outcomes = state.chance_outcomes()
                                self.assertAlmostEqual(sum(p for _, p in outcomes), 1)
                                chosen = rng.choices(
                                    [a for a, _ in outcomes],
                                    weights=[p for _, p in outcomes],
                                )[0]
                            else:
                                chosen = rng.choice(legal)
                            state.apply_action(chosen)
                            transitions += 1
                            self.assertLessEqual(transitions, game.max_history_length())
                        self.assertEqual(state.current_player(), TERMINAL)
                        self.assertEqual(state.legal_actions(), [])
                        self.assertEqual(state.chance_outcomes(), [])
                        self.assertEqual(Counter(state.cards), Counter(game.role_deck))
                        self.assertTrue(
                            all(value in (-1.0, 1.0) for value in state.returns())
                        )

    def test_all_parameter_ranges_types_and_unknown_roles_are_validated(self) -> None:
        for parameters in [
            {"players": 2},
            {"players": 11},
            {"players": True},
            {"players": "3"},
            {"discussion_rounds": -1},
            {"discussion_rounds": 4},
            {"discussion_rounds": False},
            {"lone_wolf": 1},
            {"lone_wolf": "false"},
            {"unknown": 1},
            {"roles": "werewolf"},
            {"roles": ["villager"] * 6},
            {"roles": ["werewolf"] * 6},
            {"roles": ["seer", "seer", "robber", "werewolf", "werewolf", "villager"]},
            {"roles": "werewolf|werewolf|seer|robber|villager|doppelganger"},
            {"roles": "werewolf|werewolf|seer|robber|villager|mason"},
            {"roles": "werewolf|werewolf|seer|tanner|villager|insomniac"},
            {"roles": ["werewolf", "werewolf", "seer", "robber", "villager", 1]},
            {"roles": None},
            {"roles": 42},
        ]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                WerewolfGame(parameters)
        game = WerewolfGame(
            {"roles": "werewolf|werewolf|seer|robber|troublemaker|villager"}
        )
        self.assertEqual(WerewolfGame(game.parameters).parameters, game.parameters)

    def test_chance_deals_are_weighted_without_replacement(self) -> None:
        state = WerewolfGame().new_initial_state()
        self.assertEqual(state.current_player(), CHANCE)
        self.assertAlmostEqual(
            dict(state.chance_outcomes())[ROLE_IDS["werewolf"]], 2 / 6
        )
        state.apply_action(ROLE_IDS["werewolf"])
        self.assertAlmostEqual(
            dict(state.chance_outcomes())[ROLE_IDS["werewolf"]], 1 / 5
        )
        state.apply_action(ROLE_IDS["werewolf"])
        self.assertNotIn(ROLE_IDS["werewolf"], state.legal_actions())

    def test_two_wolves_learn_each_other_but_not_the_minion(self) -> None:
        state = dealt(
            ["werewolf", "werewolf", "minion"], ["seer", "robber", "villager"]
        )
        state.apply_action(0)
        state.apply_action(0)
        state.apply_action(0)
        self.assertEqual(
            state.private_findings[0], [{"kind": "werewolf_players", "players": [1]}]
        )
        self.assertEqual(
            state.private_findings[1], [{"kind": "werewolf_players", "players": [0]}]
        )
        self.assertEqual(
            state.private_findings[2], [{"kind": "werewolf_players", "players": [0, 1]}]
        )
        self.assertEqual(state.phase, "vote")

    def test_lone_wolf_inspection_is_optional_and_parameter_controlled(self) -> None:
        for enabled in (True, False):
            state = dealt(
                ["werewolf", "seer", "robber"],
                ["werewolf", "villager", "troublemaker"],
                lone_wolf=enabled,
            )
            self.assertEqual(
                state.legal_actions(), [0, 400, 401, 402] if enabled else [0]
            )
            if enabled:
                state.apply_action(401)
                self.assertEqual(
                    state.private_findings[0][-1],
                    {"kind": "center_card", "index": 1, "role": "villager"},
                )
            else:
                state.apply_action(0)

    def test_masons_learn_only_other_original_masons(self) -> None:
        state = dealt(["mason", "mason", "seer"], ["werewolf", "werewolf", "villager"])
        state.apply_action(0)
        state.apply_action(0)
        self.assertEqual(
            state.private_findings[0], [{"kind": "mason_players", "players": [1]}]
        )
        self.assertEqual(
            state.private_findings[1], [{"kind": "mason_players", "players": [0]}]
        )
        state = dealt(["mason", "robber", "seer"], ["mason", "werewolf", "villager"])
        state.apply_action(0)
        self.assertEqual(
            state.private_findings[0], [{"kind": "mason_players", "players": []}]
        )

    def test_seer_sees_one_player_or_exactly_two_chosen_center_cards(self) -> None:
        for choice, indices in [(200, [0, 1]), (201, [0, 2]), (202, [1, 2])]:
            state = dealt(
                ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
            )
            original = list(state.cards)
            state.apply_action(choice)
            self.assertEqual(state.cards, original)
            self.assertEqual(
                [card["index"] for card in state.private_findings[0][0]["cards"]],
                indices,
            )
        state = dealt(
            ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
        )
        state.apply_action(101)
        self.assertEqual(
            state.private_findings[0],
            [{"kind": "player_card", "player": 1, "role": "robber"}],
        )

    def test_original_roles_control_night_but_final_swapped_roles_control_wins(
        self,
    ) -> None:
        state = dealt(
            ["robber", "werewolf", "troublemaker", "insomniac", "seer"],
            ["werewolf", "villager", "villager"],
        )
        self.assertEqual(state.current_player(), 1)
        state.apply_action(0)
        self.assertEqual(state.current_player(), 4)
        action(state, "inspect player 0")
        self.assertEqual(state.current_player(), 0)
        action(state, "rob player 1")
        self.assertEqual(state.cards[:2], ["werewolf", "robber"])
        self.assertEqual(state.current_player(), 2)
        action(state, "swap players 0 and 3")
        self.assertEqual(state.current_player(), 3)
        state.apply_action(0)
        self.assertEqual(
            state.cards[:5], ["insomniac", "robber", "troublemaker", "werewolf", "seer"]
        )
        self.assertEqual(
            state.private_findings[0],
            [{"kind": "robbed", "player": 1, "role": "werewolf"}],
        )
        self.assertEqual(
            state.private_findings[3],
            [{"kind": "own_current_card", "role": "werewolf"}],
        )
        self.assertEqual(
            json.loads(state.observation_string(0))["initial_role"], "robber"
        )
        vote(state, [3, 3, 3, 0, 3])
        self.assertEqual(state.winners, [0, 1, 2, 4])

    def test_drunk_must_exchange_but_never_learns_new_role(self) -> None:
        state = dealt(["drunk", "robber", "seer"], ["werewolf", "werewolf", "villager"])
        state.apply_action(0)
        state.apply_action(0)
        self.assertEqual(state.current_player(), 0)
        self.assertEqual(state.legal_actions(), [400, 401, 402])
        state.apply_action(400)
        self.assertEqual(state.cards[0], "werewolf")
        observation = json.loads(state.observation_string(0))
        self.assertEqual(observation["initial_role"], "drunk")
        self.assertEqual(
            observation["private_findings"],
            [{"kind": "exchanged_with_center", "index": 0}],
        )
        self.assertNotIn("final_roles", observation)
        self.assertNotIn("center_cards", observation)

    def test_private_night_actions_and_votes_do_not_leak(self) -> None:
        state = dealt(
            ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
        )
        before = state.observation_string(2)
        state.apply_action(101)
        self.assertEqual(state.observation_string(2), before)
        state.apply_action(0)
        observation = json.loads(state.observation_string(2))
        self.assertEqual(observation["private_findings"], [])
        self.assertNotIn("night_queue", observation)
        self.assertNotIn("original_roles", observation)
        self.assertNotIn("cards", observation)
        before = state.observation_string(1)
        state.apply_action(1002)
        self.assertEqual(state.observation_string(1), before)
        self.assertEqual(json.loads(state.observation_string(0))["own_vote"], 2)
        self.assertNotIn("votes", json.loads(str(state)))

    def test_public_claims_can_be_false_and_are_shared_in_order(self) -> None:
        state = dealt(
            ["werewolf", "robber", "villager"],
            ["werewolf", "seer", "troublemaker"],
            discussion_rounds=2,
        )
        skip_night(state)
        self.assertEqual(state.phase, "discussion")
        action(state, "claim initial role seer")
        self.assertEqual(
            json.loads(state.observation_string(2))["transcript"][0]["statement"],
            "claim initial role seer",
        )
        action(state, "accuse player 0 of being a werewolf")
        for _ in range(4):
            state.apply_action(0)
        self.assertEqual(state.phase, "vote")
        self.assertEqual(len(state.public_transcript), 6)
        self.assertEqual(
            [entry["player"] for entry in state.public_transcript], [0, 1, 2, 0, 1, 2]
        )

    def test_no_wolf_no_deaths_gives_all_villagers_a_win(self) -> None:
        state = dealt(
            ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
        )
        skip_night(state)
        vote(state, [1, 2, 0])
        self.assertEqual(state.dead, [])
        self.assertEqual(state.winners, [0, 1, 2])
        self.assertEqual(state.returns(), [1, 1, 1])

    def test_no_wolf_wrong_execution_has_no_village_winners(self) -> None:
        state = dealt(
            ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
        )
        skip_night(state)
        vote(state, [1, 0, 0])
        self.assertEqual(state.dead, [0])
        self.assertEqual(state.winners, [])

    def test_tanner_death_prevents_werewolf_win(self) -> None:
        state = dealt(
            ["werewolf", "tanner", "villager"], ["seer", "robber", "troublemaker"]
        )
        skip_night(state)
        vote(state, [1, 0, 1])
        self.assertEqual(state.dead, [1])
        self.assertEqual(state.winners, [1])

    def test_tied_wolf_and_tanner_execution_shares_village_and_tanner_win(self) -> None:
        state = dealt(
            ["werewolf", "tanner", "robber", "villager"],
            ["werewolf", "seer", "troublemaker"],
        )
        skip_night(state)
        vote(state, [1, 0, 0, 1])
        self.assertEqual(state.dead, [0, 1])
        self.assertEqual(state.winners, [1, 2, 3])

    def test_killed_hunter_also_kills_its_vote_target(self) -> None:
        state = dealt(
            ["hunter", "werewolf", "villager"], ["seer", "robber", "troublemaker"]
        )
        skip_night(state)
        vote(state, [1, 0, 0])
        self.assertEqual(state.dead, [0, 1])
        self.assertEqual(state.winners, [0, 2])
        final = json.loads(state.observation_string(2))
        self.assertEqual(final["final_roles"], ["hunter", "werewolf", "villager"])
        self.assertEqual(final["votes"], [1, 0, 0])

    def test_minion_can_die_while_protecting_living_wolf(self) -> None:
        state = dealt(
            ["werewolf", "minion", "villager"], ["seer", "robber", "troublemaker"]
        )
        skip_night(state)
        vote(state, [1, 0, 1])
        self.assertEqual(state.dead, [1])
        self.assertEqual(state.winners, [0, 1])

    def test_no_wolf_minion_uses_explicit_english_rulebook_condition(self) -> None:
        state = dealt(
            ["minion", "villager", "villager", "hunter"],
            ["werewolf", "werewolf", "seer"],
        )
        skip_night(state)
        vote(state, [1, 0, 0, 1])
        self.assertEqual(state.dead, [0, 1])
        self.assertEqual(state.winners, [0])
        state = dealt(
            ["minion", "villager", "hunter"], ["werewolf", "werewolf", "seer"]
        )
        skip_night(state)
        vote(state, [1, 0, 0])
        self.assertEqual(state.dead, [0])
        self.assertEqual(state.winners, [])

    def test_no_wolf_tanner_win_also_overrides_minion(self) -> None:
        state = dealt(
            ["minion", "tanner", "villager", "hunter"], ["werewolf", "werewolf", "seer"]
        )
        skip_night(state)
        vote(state, [1, 0, 0, 1])
        self.assertEqual(state.dead, [0, 1])
        self.assertEqual(state.winners, [1])

    def test_illegal_actions_never_mutate_and_clones_do_not_share_state(self) -> None:
        state = WerewolfGame().new_initial_state()
        for invalid in (-1, 10000, True, "0", None):
            before = state.serialize()
            with self.assertRaises(ValueError):
                state.apply_action(invalid)  # type: ignore[arg-type]
            self.assertEqual(state.serialize(), before)
        clone = state.clone()
        clone.apply_action(clone.legal_actions()[0])
        self.assertNotEqual(clone.serialize(), state.serialize())
        self.assertEqual(state.cards, [])
        state = dealt(
            ["seer", "robber", "villager"], ["werewolf", "werewolf", "troublemaker"]
        )
        before = state.serialize()
        with self.assertRaises(ValueError):
            state.apply_action(100)  # seer cannot inspect self
        self.assertEqual(state.serialize(), before)
        skip_night(state)
        self.assertNotIn(1000, state.legal_actions())
        for invalid_player in (-1, 3, True):
            with self.assertRaises(ValueError):
                state.observation_string(invalid_player)
        self.assertEqual(state.legal_actions(1), [])


if __name__ == "__main__":
    unittest.main()
