"""CATAN rules are exercised without models, keys, network or arena workers."""

from __future__ import annotations

from copy import deepcopy
import json
import random
import unittest

from game_arena.environments.catan import (
    ACCEPT,
    AXIAL,
    BUY_DEVELOPMENT,
    CANCEL,
    CITY,
    COAST,
    COUNTER,
    DEVELOPMENTS,
    EDGES,
    END,
    GIVE_ADD,
    MARITIME,
    NEIGHBORS,
    NUMBER_TOKENS,
    PARTNER,
    PLAY_DEVELOPMENT,
    RECEIVE_ADD,
    REJECT,
    RESOURCE,
    RESOURCES,
    ROAD,
    ROBBER,
    ROLL,
    SEND,
    SETTLEMENT,
    TILE_VERTICES,
    TRADE,
    VERTICES,
    VICTIM,
    CatanGame,
    CatanState,
)


def prepared(players: int = 3, **parameters: int) -> CatanState:
    state = CatanGame(
        {"players": players, "starting_player": 0, **parameters}
    ).new_initial_state()
    while state.phase in {
        "terrain",
        "starting_player",
        "setup_settlement",
        "setup_road",
    }:
        state.apply_action(state.legal_actions()[0])
    return state


def grant(state: CatanState, player: int, quantities: list[int]) -> None:
    """Move fixture cards from the bank, keeping the 95-card invariant."""
    for resource, quantity in enumerate(quantities):
        state.hands[player][resource] += quantity
        state.bank[resource] -= quantity


def main_state() -> CatanState:
    state = prepared()
    state.apply_action(ROLL)
    state.apply_action(2)
    return state


def offer(state: CatanState, partner: int = 1, give: int = 0, receive: int = 1) -> None:
    state.apply_action(TRADE)
    state.apply_action(PARTNER + partner)
    state.apply_action(GIVE_ADD + give)
    state.apply_action(RECEIVE_ADD + receive)
    state.apply_action(SEND)


class CatanEnvironmentTests(unittest.TestCase):
    def assert_inventory(self, state: CatanState) -> None:
        for resource in range(5):
            self.assertEqual(
                state.bank[resource] + sum(hand[resource] for hand in state.hands), 19
            )
            self.assertGreaterEqual(state.bank[resource], 0)
            self.assertTrue(all(hand[resource] >= 0 for hand in state.hands))
        for card, total in enumerate((14, 2, 2, 2, 5)):
            self.assertEqual(
                state.development_supply[card]
                + sum(hand[card] for hand in state.development)
                + state.development_discard[card],
                total,
            )

    def test_strict_parameters_and_geometry(self) -> None:
        for values in (
            {"players": 2},
            {"players": 5},
            {"players": True},
            {"max_turns": 0},
            {"starting_player": 4},
            {"max_actions_per_turn": 1},
            {"seafarers": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CatanGame(values)
        self.assertEqual(
            (len(AXIAL), len(VERTICES), len(EDGES), len(COAST)), (19, 54, 72, 30)
        )
        self.assertEqual(len(set(EDGES)), 72)
        self.assertTrue(all(len(set(tile)) == 6 for tile in TILE_VERTICES))
        state = CatanGame().new_initial_state()
        self.assertEqual(sum(port is not None for port in state.ports), 18)
        self.assertEqual(sum(port == "any" for port in state.ports), 8)

    def test_explicit_terrain_starting_player_and_spiral(self) -> None:
        state = CatanGame().new_initial_state()
        self.assertEqual(state.chance_outcomes()[0], (0, 3 / 19))
        for _ in range(19):
            state.apply_action(state.legal_actions()[0])
        self.assertEqual(state.phase, "starting_player")
        self.assertEqual(state.chance_outcomes(), [(p, 0.25) for p in range(4)])
        self.assertEqual(sorted(state.numbers), sorted([0, *NUMBER_TOKENS]))
        self.assertEqual(state.numbers[state.robber], 0)
        state.apply_action(2)
        self.assertEqual(state.setup_order, [2, 3, 0, 1, 1, 0, 3, 2])
        # Every possible desert position preserves nonadjacent red tokens.
        for desert in range(19):
            check = CatanGame({"starting_player": 0}).new_initial_state()
            terrain = [0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 3
            terrain.insert(desert, 5)
            for tile in terrain:
                check.apply_action(tile)
            red = [i for i, number in enumerate(check.numbers) if number in {6, 8}]
            self.assertTrue(
                all(
                    len(set(TILE_VERTICES[a]) & set(TILE_VERTICES[b])) < 2
                    for a in red
                    for b in red
                    if a != b
                )
            )

    def test_snake_placement_distance_second_resources_and_initial_road(self) -> None:
        state = CatanGame({"players": 3, "starting_player": 0}).new_initial_state()
        while state.phase == "terrain":
            state.apply_action(state.legal_actions()[0])
        actors: list[int] = []
        for placement in range(6):
            player = state.current_player()
            actors.append(player)
            before = sum(state.hands[player])
            action = state.legal_actions()[0]
            vertex = action - SETTLEMENT
            self.assertTrue(all(state.building_owner[n] < 0 for n in NEIGHBORS[vertex]))
            state.apply_action(action)
            delta = sum(state.hands[player]) - before
            expected = (
                sum(
                    state.terrain[t] < 5
                    for t in range(19)
                    if vertex in TILE_VERTICES[t]
                )
                if placement >= 3
                else 0
            )
            self.assertEqual(delta, expected)
            self.assertTrue(
                all(vertex in EDGES[a - ROAD] for a in state.legal_actions())
            )
            state.apply_action(state.legal_actions()[0])
        self.assertEqual(actors, [0, 1, 2, 2, 1, 0])
        self.assertEqual(
            (state.phase, state.turn_player, state.turn_number), ("pre_roll", 0, 1)
        )
        self.assert_inventory(state)

    def test_maritime_cost_ports_and_bank_conservation(self) -> None:
        state = main_state()
        state.building_owner = [-1] * 54
        state.building_level = [0] * 54
        state.hands = [[0] * 5 for _ in range(3)]
        state.bank = [19] * 5
        grant(state, 0, [4, 0, 0, 0, 0])
        self.assertIn(MARITIME + 1, state.legal_actions())
        state.apply_action(MARITIME + 1)
        self.assertEqual(state.hands[0], [0, 1, 0, 0, 0])
        any_port = state.ports.index("any")
        state.building_owner[any_port], state.building_level[any_port] = 0, 1
        grant(state, 0, [3, 0, 0, 0, 0])
        self.assertEqual(state._ratio(0, 0), 3)
        state.apply_action(MARITIME + 2)
        brick_port = state.ports.index("brick")
        state.building_owner[brick_port], state.building_level[brick_port] = 0, 1
        grant(state, 0, [2, 0, 0, 0, 0])
        self.assertEqual(state._ratio(0, 0), 2)
        state.apply_action(MARITIME + 3)
        self.assertEqual(state.hands[0], [0, 1, 1, 1, 0])
        self.assert_inventory(state)

    def test_trade_requires_consent_private_draft_and_disjoint_bundles(self) -> None:
        state = main_state()
        grant(state, 0, [2, 0, 0, 0, 0])
        grant(state, 1, [0, 2, 0, 0, 0])
        before = deepcopy(state.hands)
        state.apply_action(TRADE)
        state.apply_action(PARTNER + 1)
        state.apply_action(GIVE_ADD)
        self.assertNotIn(RECEIVE_ADD, state.legal_actions())
        self.assertIsNone(state.public_state()["trade"])
        self.assertIsNone(json.loads(state.observation_string(1))["trade"])
        self.assertEqual(
            json.loads(state.observation_string(0))["trade"]["give"]["brick"], 1
        )
        state.apply_action(GIVE_ADD)
        state.apply_action(RECEIVE_ADD + 1)
        state.apply_action(SEND)
        self.assertEqual(state.current_player(), 1)
        self.assertEqual(state.hands, before)
        self.assertEqual(state.public_state()["trade"]["give"]["brick"], 2)
        state.apply_action(ACCEPT)
        self.assertEqual(state.hands[0][0], before[0][0] - 2)
        self.assertEqual(state.hands[0][1], before[0][1] + 1)
        self.assertEqual(state.hands[1][0], before[1][0] + 2)
        self.assertEqual((state.current_player(), state.phase), (0, "main"))
        self.assert_inventory(state)

    def test_trade_unaffordable_reject_counteroffer_and_invalid_atomicity(self) -> None:
        state = main_state()
        grant(state, 0, [2, 0, 0, 0, 0])
        # Requesting a resource does not reveal whether the other player has it.
        state.hands[1][1], state.bank[1] = 0, 19 - state.hands[0][1] - state.hands[2][1]
        offer(state)
        self.assertNotIn(ACCEPT, state.legal_actions())
        before = state.serialize()
        with self.assertRaises(ValueError):
            state.apply_action(ACCEPT)
        self.assertEqual(state.serialize(), before)
        state.apply_action(REJECT)
        grant(state, 1, [0, 1, 0, 0, 0])
        offer(state)
        state.apply_action(COUNTER)
        self.assertEqual(state.current_player(), 1)
        state.apply_action(GIVE_ADD + 1)
        state.apply_action(RECEIVE_ADD)
        state.apply_action(SEND)
        self.assertEqual(state.current_player(), 0)
        self.assertEqual(state.public_state()["trade"]["recipient"], 0)
        state.apply_action(ACCEPT)
        self.assertEqual((state.current_player(), state.phase), (0, "main"))
        state.apply_action(TRADE)
        state.apply_action(CANCEL)
        self.assertEqual(state.phase, "main")
        self.assert_inventory(state)

    def test_resource_production_city_robber_and_shortage(self) -> None:
        state = main_state()
        state.building_owner, state.building_level = [-1] * 54, [0] * 54
        tile = next(i for i, resource in enumerate(state.terrain) if resource == 0)
        state.numbers = [0] * 19
        state.numbers[tile] = 6
        state.robber = next(i for i in range(19) if i != tile)
        a, b = TILE_VERTICES[tile][0], TILE_VERTICES[tile][2]
        state.building_owner[a], state.building_level[a] = 0, 2
        before = state.hands[0][0]
        state._produce(6)
        self.assertEqual(state.hands[0][0], before + 2)
        state.robber = tile
        state._produce(6)
        self.assertEqual(state.hands[0][0], before + 2)
        state.robber = -1
        state.bank[0] = 1
        state._produce(6)
        self.assertEqual(state.hands[0][0], before + 3)
        state.bank[0] = 1
        state.building_owner[b], state.building_level[b] = 1, 1
        hands = deepcopy(state.hands)
        state._produce(6)
        self.assertEqual(state.hands, hands)
        self.assertEqual(state.bank[0], 1)

    def test_seven_discards_once_then_weighted_steal(self) -> None:
        state = prepared()
        state.hands = [[0] * 5 for _ in range(3)]
        state.bank = [19] * 5
        grant(state, 0, [8, 0, 0, 0, 0])
        grant(state, 1, [0, 9, 0, 0, 0])
        state.apply_action(ROLL)
        self.assertAlmostEqual(dict(state.chance_outcomes())[7], 1 / 6)
        state.apply_action(7)
        for _ in range(4):
            self.assertEqual(state.current_player(), 0)
            state.apply_action(RESOURCE)
        for _ in range(4):
            self.assertEqual(state.current_player(), 1)
            state.apply_action(RESOURCE + 1)
        self.assertEqual(
            (state.phase, list(map(sum, state.hands))), ("robber", [4, 5, 0])
        )
        target = next(tile for tile in range(19) if tile != state.robber)
        state.building_owner, state.building_level = [-1] * 54, [0] * 54
        v = TILE_VERTICES[target][0]
        state.building_owner[v], state.building_level[v] = 1, 1
        state.apply_action(ROBBER + target)
        self.assertEqual(state.legal_actions(), [VICTIM + 1])
        state.apply_action(VICTIM + 1)
        self.assertEqual(state.chance_outcomes(), [(1, 1.0)])
        state.apply_action(1)
        self.assertEqual(state.phase, "main")
        self.assertEqual(state.hands[0][1], 1)
        self.assertNotIn("lumber", state.public_state()["last_event"])
        self.assert_inventory(state)

    def test_development_purchase_timing_and_victory_point_exception(self) -> None:
        state = main_state()
        grant(state, 0, [0, 0, 2, 2, 2])
        state.apply_action(BUY_DEVELOPMENT)
        self.assertEqual(dict(state.chance_outcomes())[0], 14 / 25)
        self.assertEqual(state.action_to_string(-1, 0), "Draw hidden development card")
        state.apply_action(0)
        self.assertNotIn(PLAY_DEVELOPMENT, state.legal_actions())
        state.new_development[0] = [0] * 5
        self.assertIn(PLAY_DEVELOPMENT, state.legal_actions())
        state.phase = "pre_roll"
        state.apply_action(PLAY_DEVELOPMENT)
        self.assertEqual(state.phase, "robber")
        self.assertEqual(state.knights[0], 1)
        self.assertEqual(state.discard_remaining, [0, 0, 0])
        state.apply_action(
            next(
                a
                for a in state.legal_actions()
                if not any(
                    state.building_owner[v] not in {-1, 0}
                    and sum(state.hands[state.building_owner[v]])
                    for v in TILE_VERTICES[a - ROBBER]
                )
            )
        )
        self.assertEqual(state.phase, "pre_roll")
        self.assertNotIn(PLAY_DEVELOPMENT, state.legal_actions())
        self.assert_inventory(state)
        victory = main_state()
        victory.building_owner = [0] * 6 + [-1] * 48
        victory.building_level = [2, 2, 2, 1, 1, 1] + [0] * 48
        grant(victory, 0, [0, 0, 1, 1, 1])
        victory.apply_action(BUY_DEVELOPMENT)
        # Fixture already has nine points; a just-bought VP card may win now.
        victory.apply_action(4)
        self.assertEqual((victory.winner, victory.phase), (0, "terminal"))
        self.assert_inventory(victory)

    def test_knight_wins_immediately_and_next_turn_victory_is_recorded_once(
        self,
    ) -> None:
        state = main_state()
        state.building_owner = [0] * 4 + [-1] * 50
        state.building_level = [2] * 4 + [0] * 50
        state.knights[0] = 2
        state.development[0][0] = 1
        state.development_supply[0] -= 3
        state.development_discard[0] = 2
        state.apply_action(PLAY_DEVELOPMENT)
        self.assertEqual(
            (state.phase, state.winner, state.largest_army_owner), ("terminal", 0, 0)
        )
        self.assertEqual(
            sum("wins with" in message for message in state.public_history), 1
        )
        self.assert_inventory(state)
        state = main_state()
        state.building_owner = [1] * 6 + [-1] * 48
        state.building_level = [2, 2, 2, 1, 1, 1] + [0] * 48
        state.development[1][4] = 1
        state.development_supply[4] -= 1
        self.assertEqual(state.winner, None)
        state.apply_action(END)
        self.assertEqual((state.phase, state.winner), ("terminal", 1))
        self.assertEqual(
            sum("wins with" in message for message in state.public_history), 1
        )

    def test_progress_cards_monopoly_plenty_roads_and_once_per_turn(self) -> None:
        for card in (1, 2, 3):
            state = main_state()
            state.development[0][card] += 1
            state.development_supply[card] -= 1
            grant(state, 1, [0, 0, 2, 0, 0])
            grant(state, 2, [0, 0, 1, 0, 0])
            before = deepcopy(state.hands)
            state.apply_action(PLAY_DEVELOPMENT + card)
            if card == 1:
                roads_before = state.road_owner.count(0)
                state.apply_action(state.legal_actions()[0])
                state.apply_action(state.legal_actions()[0])
                self.assertEqual(state.road_owner.count(0), roads_before + 2)
                self.assertEqual(state.hands, before)
            elif card == 2:
                state.apply_action(RESOURCE + 4)
                state.apply_action(RESOURCE + 4)
                self.assertEqual(state.hands[0][4], before[0][4] + 2)
            else:
                state.apply_action(RESOURCE + 2)
                self.assertEqual(state.hands[0][2], sum(row[2] for row in before))
                self.assertEqual(state.hands[1][2], 0)
                self.assertEqual(state.hands[2][2], 0)
            self.assertEqual(state.phase, "main")
            self.assertTrue(state.development_played)
            self.assert_inventory(state)

    def test_longest_road_loop_branch_block_and_bonus_ties(self) -> None:
        state = main_state()
        state.road_owner = [-1] * 72
        state.building_owner = [-1] * 54
        state.building_level = [0] * 54
        vertices = TILE_VERTICES[9]
        loop = [i for i, (a, b) in enumerate(EDGES) if a in vertices and b in vertices]
        self.assertEqual(len(loop), 6)
        for edge in loop:
            state.road_owner[edge] = 0
        self.assertEqual(state._longest(0), 6)
        state._bonuses()
        self.assertEqual(state.longest_road_owner, 0)
        state.building_owner[vertices[0]], state.building_level[vertices[0]] = 1, 1
        # A single break opens the ring, retaining six edges; two breaks split it.
        self.assertEqual(state._longest(0), 6)
        state.building_owner[vertices[3]], state.building_level[vertices[3]] = 1, 1
        self.assertEqual(state._longest(0), 3)
        state._bonuses()
        self.assertIsNone(state.longest_road_owner)
        self.assertEqual(state._bonus_holder([5, 5, 4], 5, 1), 1)
        self.assertIsNone(state._bonus_holder([5, 5, 4], 5, None))
        self.assertEqual(state._bonus_holder([5, 6, 4], 5, 0), 1)

    def test_build_costs_city_stock_and_enemy_road_blocking(self) -> None:
        state = main_state()
        grant(state, 0, [4, 4, 3, 5, 6])
        road = next(a for a in state.legal_actions() if ROAD <= a < ROAD + 72)
        before = list(state.hands[0])
        state.apply_action(road)
        self.assertEqual(state.hands[0][:2], [before[0] - 1, before[1] - 1])
        city = next(a for a in state.legal_actions() if CITY <= a < CITY + 54)
        before = list(state.hands[0])
        state.apply_action(city)
        self.assertEqual(state.hands[0][3:], [before[3] - 2, before[4] - 3])
        self.assertEqual(state.building_level[city - CITY], 2)
        self.assert_inventory(state)
        # Opponent building cannot be used as a connecting endpoint.
        state.road_owner = [-1] * 72
        state.building_owner = [-1] * 54
        edge = next(
            i
            for i, (a, b) in enumerate(EDGES)
            if len(NEIGHBORS[a]) == 3 and len(NEIGHBORS[b]) == 3
        )
        a, b = EDGES[edge]
        state.road_owner[edge] = 0
        state.building_owner[b] = 1
        beyond = next(
            i for i, vertices in enumerate(EDGES) if b in vertices and i != edge
        )
        self.assertNotIn(beyond, state._road_sites(0))
        self.assertTrue(any(a in EDGES[e] for e in state._road_sites(0)))

    def test_private_hands_safe_public_snapshots_and_caps(self) -> None:
        state = main_state()
        grant(state, 0, [1, 0, 0, 0, 0])
        snapshot = state.public_state()
        self.assertNotIn("hands", snapshot)
        self.assertNotIn("resources", snapshot["players"][0])
        snapshot["board"]["vertices"][0]["owner"] = 99
        self.assertNotEqual(state.building_owner[0], 99)
        state.public_history = [str(i) for i in range(100)]
        observation = json.loads(state.observation_string(0))
        self.assertEqual(len(observation["public_history"]), 32)
        self.assertEqual(observation["public_history_total"], 100)
        self.assertEqual(set(observation["resources"]), set(RESOURCES))
        self.assertEqual(set(observation["development_cards"]), set(DEVELOPMENTS))
        capped = prepared(max_turns=1)
        capped.apply_action(ROLL)
        capped.apply_action(2)
        capped.apply_action(END)
        self.assertEqual(
            (capped.phase, capped.end_reason, capped.returns()),
            ("terminal", "turn_limit", [0, 0, 0]),
        )

    def test_fifty_random_games_are_legal_atomic_and_conserve_cards(self) -> None:
        for seed in range(50):
            rng = random.Random(seed)
            state = CatanGame(
                {"players": 3 + seed % 2, "max_turns": 12}
            ).new_initial_state()
            steps = 0
            while not state.is_terminal():
                actions = state.legal_actions()
                self.assertTrue(actions, (seed, state.phase))
                before = state.serialize()
                with self.assertRaises(ValueError):
                    state.apply_action(1234567)
                self.assertEqual(state.serialize(), before)
                if state.is_chance_node():
                    choices, weights = zip(*state.chance_outcomes(), strict=True)
                    self.assertAlmostEqual(sum(weights), 1)
                    action = rng.choices(choices, weights=weights)[0]
                else:
                    action = rng.choice(actions)
                label = state.action_to_string(state.current_player(), action)
                self.assertTrue(label)
                clone = state.clone()
                state.apply_action(action)
                clone.apply_action(action)
                self.assertEqual(state.serialize(), clone.serialize())
                self.assert_inventory(state)
                steps += 1
                self.assertLess(steps, 3000)
            self.assertEqual(state.returns(), [0] * state.game.players)


if __name__ == "__main__":
    unittest.main()
