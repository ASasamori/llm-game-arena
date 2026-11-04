"""Avalon rule tests exercise only the environment, without models or services."""

from __future__ import annotations

from collections import Counter
import json
import random
import unittest

from game_arena.environments.avalon import (
    EVIL_COUNTS,
    EVIL_ROLES,
    ROLES,
    TEAM_SIZES,
    AvalonGame,
    AvalonState,
)
from game_arena.environments.base import CHANCE, TERMINAL


def dealt_state(roles: list[str] | None = None, leader: int = 0) -> AvalonState:
    roles = roles or ["merlin", "servant", "servant", "assassin", "minion"]
    game = AvalonGame(
        {"players": len(roles), "roles": "|".join(roles), "discussion_rounds": 0}
    )
    state = game.new_initial_state()
    for role in roles:
        state.apply_action(ROLES.index(role))
    state.apply_action(leader)
    return state


def propose(state: AvalonState, team: list[int]) -> None:
    expected = "propose_team:" + ",".join(map(str, sorted(team)))
    action = next(
        a
        for a in state.legal_actions()
        if state.action_to_string(state.current_player(), a) == expected
    )
    state.apply_action(action)


def approve(state: AvalonState, votes: list[int] | None = None) -> None:
    for vote in votes or [1] * state.game.players:
        state.apply_action(vote)


def quest(
    state: AvalonState, team: list[int], fail_seats: tuple[int, ...] = ()
) -> None:
    propose(state, team)
    approve(state)
    while state.phase == "quest":
        state.apply_action(int(state.current_player() in fail_seats))


class AvalonEnvironmentTests(unittest.TestCase):
    def test_defaults_and_every_supported_player_count(self) -> None:
        for players in range(5, 11):
            game = AvalonGame({"players": players})
            self.assertEqual(game.num_players(), players)
            self.assertEqual(len(game.role_deck), players)
            self.assertEqual(
                sum(role in EVIL_ROLES for role in game.role_deck), EVIL_COUNTS[players]
            )
            self.assertEqual(game.role_deck.count("merlin"), 1)
            self.assertEqual(game.role_deck.count("assassin"), 1)
            self.assertEqual(AvalonGame(game.parameters).parameters, game.parameters)
            self.assertIn(str(TEAM_SIZES[players]), game.rules)

    def test_invalid_parameters_fail_before_state_creation(self) -> None:
        cases = [
            {"players": 4},
            {"players": 11},
            {"players": True},
            {"players": 5.0},
            {"discussion_rounds": -1},
            {"discussion_rounds": 4},
            {"discussion_rounds": True},
            {"unknown": 1},
            {"roles": []},
            {"roles": "merlin|servant"},
            {"roles": "merlin|servant|servant|assassin|dragon"},
            {"roles": "merlin|merlin|servant|assassin|minion"},
            {"roles": "servant|servant|servant|assassin|minion"},
            {"roles": "merlin|servant|assassin|minion|morgana"},
            {"roles": "merlin|servant|servant|minion|minion"},
            {
                "players": 7,
                "roles": "merlin|percival|percival|servant|assassin|minion|minion",
            },
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AvalonGame(values)

    def test_explicit_chance_is_weighted_without_replacement(self) -> None:
        state = AvalonGame().new_initial_state()
        probabilities = dict(state.chance_outcomes())
        self.assertAlmostEqual(probabilities[ROLES.index("servant")], 2 / 5)
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        state.apply_action(ROLES.index("merlin"))
        self.assertNotIn(ROLES.index("merlin"), state.legal_actions())
        state.apply_action(ROLES.index("assassin"))
        state.apply_action(ROLES.index("minion"))
        state.apply_action(ROLES.index("servant"))
        state.apply_action(ROLES.index("servant"))
        self.assertEqual(state.phase, "leader")
        self.assertEqual(state.current_player(), CHANCE)
        self.assertEqual(state.chance_outcomes(), [(p, 0.2) for p in range(5)])
        state.apply_action(4)
        self.assertEqual(state.current_player(), 4)

    def test_special_role_knowledge_original_2012_rules(self) -> None:
        roles = [
            "merlin",
            "percival",
            "servant",
            "servant",
            "servant",
            "servant",
            "assassin",
            "morgana",
            "mordred",
            "oberon",
        ]
        state = dealt_state(roles)
        views = [json.loads(state.observation_string(p)) for p in range(10)]
        self.assertEqual(views[0]["knowledge"], {"known_evil": [6, 7, 9]})
        self.assertEqual(views[1]["knowledge"], {"merlin_candidates": [0, 7]})
        self.assertEqual(views[6]["knowledge"], {"evil_allies": [7, 8]})
        self.assertEqual(views[8]["knowledge"], {"evil_allies": [6, 7]})
        self.assertEqual(views[9]["knowledge"], {})
        self.assertEqual(views[2]["knowledge"], {})
        self.assertEqual(views[6]["role"], "assassin")
        self.assertNotIn("revealed_roles", views[6])
        self.assertNotIn("remaining_roles", views[6])

    def test_secret_approval_votes_are_revealed_only_together(self) -> None:
        a = dealt_state()
        propose(a, [0, 1])
        b = a.clone()
        a.apply_action(0)
        b.apply_action(1)
        self.assertEqual(a.observation_string(1), b.observation_string(1))
        self.assertNotEqual(a.observation_string(0), b.observation_string(0))
        for _ in range(4):
            a.apply_action(1)
            b.apply_action(1)
        result = json.loads(a.observation_string(2))["public_history"][-1]
        self.assertEqual(result["kind"], "team_vote")
        self.assertEqual(result["votes"], [False, True, True, True, True])
        self.assertTrue(result["approved"])

    def test_strict_majority_and_five_rejections(self) -> None:
        state = dealt_state(
            ["merlin", "servant", "servant", "servant", "assassin", "minion"]
        )
        for attempt in range(5):
            self.assertEqual(state.leader, attempt)
            propose(state, [0, 1])
            approve(state, [1, 1, 1, 0, 0, 0])
            if attempt < 4:
                self.assertFalse(state.is_terminal())
                self.assertEqual(state.rejections, attempt + 1)
        self.assertEqual(state.current_player(), TERMINAL)
        self.assertEqual(state.winner, "evil")
        self.assertEqual(state.end_reason, "five_rejected_teams")
        self.assertEqual(state.returns(), [-1, -1, -1, -1, 1, 1])

    def test_approval_resets_rejections_and_leader_rotates_after_quest(self) -> None:
        state = dealt_state()
        propose(state, [0, 1])
        approve(state, [0] * 5)
        self.assertEqual(state.rejections, 1)
        self.assertEqual(state.leader, 1)
        quest(state, [0, 1])
        self.assertEqual(state.rejections, 0)
        self.assertEqual(state.leader, 2)
        self.assertEqual(state.phase, "proposal")

    def test_good_cannot_sabotage_and_quest_votes_are_anonymous(self) -> None:
        a = dealt_state()
        propose(a, [0, 3])
        approve(a)
        self.assertEqual(a.current_player(), 0)
        self.assertEqual(a.legal_actions(), [0])
        before = a.serialize()
        with self.assertRaises(ValueError):
            a.apply_action(1)
        self.assertEqual(before, a.serialize())
        a.apply_action(0)
        self.assertEqual(a.legal_actions(), [0, 1])
        a.apply_action(1)
        view = json.loads(a.observation_string(2))
        result = view["quest_results"][0]
        self.assertFalse(result["success"])
        self.assertEqual(result["fails"], 1)
        self.assertNotIn("quest_votes", view)
        self.assertNotIn("cards", result)

    def test_unresolved_quest_cards_do_not_leak_to_other_seats(self) -> None:
        a = dealt_state()
        propose(a, [3, 4])
        approve(a)
        b = a.clone()
        a.apply_action(0)
        b.apply_action(1)
        self.assertEqual(a.observation_string(4), b.observation_string(4))
        self.assertNotEqual(a.observation_string(3), b.observation_string(3))

    def test_three_failures_end_immediately(self) -> None:
        state = dealt_state()
        quest(state, [0, 3], (3,))
        quest(state, [0, 1, 3], (3,))
        quest(state, [0, 3], (3,))
        self.assertEqual(state.winner, "evil")
        self.assertEqual(state.end_reason, "three_failed_quests")
        self.assertEqual(state.phase, "terminal")

    def test_three_successes_require_assassination_before_good_wins(self) -> None:
        state = dealt_state()
        quest(state, [0, 1])
        quest(state, [0, 1, 2])
        quest(state, [1, 2])
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.returns(), [0.0] * 5)
        self.assertEqual(state.phase, "assassinate")
        self.assertEqual(state.current_player(), 3)
        self.assertEqual(state.legal_actions(), [0, 1, 2, 4])
        other = state.clone()
        state.apply_action(0)
        other.apply_action(1)
        self.assertEqual(state.winner, "evil")
        self.assertEqual(state.end_reason, "merlin_assassinated")
        self.assertEqual(other.winner, "good")
        self.assertEqual(other.returns(), [1, 1, 1, -1, -1])
        self.assertEqual(
            json.loads(other.observation_string(4))["revealed_roles"], other.roles
        )

    def test_quest_four_requires_two_fails_only_at_seven_or_more_players(self) -> None:
        for players in range(5, 11):
            game = AvalonGame({"players": players, "discussion_rounds": 0})
            roles = list(game.role_deck)
            state = dealt_state(roles)
            good = [p for p, role in enumerate(roles) if role not in EVIL_ROLES]
            evil = [p for p, role in enumerate(roles) if role in EVIL_ROLES]
            # Reach quest four with two success and one fail; no early terminal.
            for index in range(3):
                team = sorted(([evil[0]] + good)[: state.team_size])
                quest(state, team, (evil[0],) if index == 0 else ())
            original = state.clone()
            team = sorted(([evil[0]] + good)[: state.team_size])
            quest(state, team, (evil[0],))
            self.assertEqual(
                state.quest_results[-1]["fails_required"], 2 if players >= 7 else 1
            )
            self.assertEqual(state.quest_results[-1]["success"], players >= 7)
            if players >= 7:
                team = sorted((evil[:2] + good)[: original.team_size])
                quest(original, team, tuple(evil[:2]))
                self.assertFalse(original.quest_results[-1]["success"])

    def test_structured_discussion_is_public_bounded_and_permits_bluffs(self) -> None:
        game = AvalonGame({"discussion_rounds": 2})
        state = game.new_initial_state()
        for role in game.role_deck:
            state.apply_action(ROLES.index(role))
        state.apply_action(2)
        state.apply_action(0)
        self.assertEqual(state.phase, "discussion")
        for index in range(10):
            self.assertEqual(state.current_player(), (2 + index) % 5)
            # Every role claim is legal, irrespective of the seat's true role.
            self.assertIn(3 + ROLES.index("merlin"), state.legal_actions())
            state.apply_action(3 + ROLES.index("merlin"))
        self.assertEqual(state.phase, "vote")
        transcript = json.loads(state.observation_string(0))["public_history"]
        self.assertEqual(sum(event["kind"] == "discussion" for event in transcript), 10)
        self.assertEqual(transcript[-1]["statement"], "claim_role:merlin")

    def test_clone_and_illegal_actions_preserve_original_state(self) -> None:
        state = dealt_state()
        before = state.serialize()
        clone = state.clone()
        clone.apply_action(0)
        self.assertEqual(state.serialize(), before)
        self.assertNotEqual(clone.serialize(), before)
        for action in (-1, 999, True, 0.5, "0"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                state.apply_action(action)  # type: ignore[arg-type]
            self.assertEqual(state.serialize(), before)
        self.assertEqual(state.legal_actions(1), [])
        for player in (-1, 5, True):
            with self.assertRaises(ValueError):
                state.observation_string(player)

    def test_terminal_states_have_no_actions_and_no_further_mutation(self) -> None:
        state = dealt_state()
        for _ in range(5):
            propose(state, [0, 1])
            approve(state, [0] * 5)
        before = state.serialize()
        self.assertEqual(state.legal_actions(), [])
        self.assertEqual(state.chance_outcomes(), [])
        with self.assertRaises(ValueError):
            state.apply_action(0)
        self.assertEqual(before, state.serialize())

    def test_random_environment_games_terminate_and_replay_without_ai(self) -> None:
        for players in range(5, 11):
            for seed in range(10):
                rng = random.Random(seed)
                game = AvalonGame({"players": players, "discussion_rounds": seed % 4})
                state = game.new_initial_state()
                actions = []
                while not state.is_terminal():
                    self.assertLess(len(actions), game.max_history_length())
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        self.assertAlmostEqual(sum(p for _, p in outcomes), 1)
                        action = rng.choices(
                            [a for a, _ in outcomes], [p for _, p in outcomes]
                        )[0]
                    else:
                        action = rng.choice(state.legal_actions())
                        self.assertTrue(
                            state.action_to_string(state.current_player(), action)
                        )
                        json.loads(state.observation_string(state.current_player()))
                    actions.append(action)
                    state.apply_action(action)
                self.assertEqual(Counter(state.roles), Counter(game.role_deck))
                self.assertEqual(len(state.returns()), players)
                self.assertTrue(all(value in {-1.0, 1.0} for value in state.returns()))
                replay = game.new_initial_state()
                for action in actions:
                    replay.apply_action(action)
                self.assertEqual(replay.serialize(), state.serialize())
                self.assertEqual(replay.returns(), state.returns())


if __name__ == "__main__":
    unittest.main()
