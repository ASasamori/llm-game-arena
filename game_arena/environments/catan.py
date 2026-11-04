"""Three/four-player base CATAN, explicit randomness and consensual trading.

Rules edition: 2015/2020 base game and its combined trade/build variant.
https://www.catan.com/sites/default/files/2021-07/catan-25th-rules_eng-200313.pdf
No expansions, helpers, special building phase, or two-player variant.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any

from game_arena.environments.base import (
    CHANCE,
    TERMINAL,
    NativeGame,
    NativeState,
    integer_parameter,
    validate_parameters,
)

RESOURCES = ("brick", "lumber", "wool", "grain", "ore")
DEVELOPMENTS = (
    "knight",
    "road_building",
    "year_of_plenty",
    "monopoly",
    "victory_point",
)
ROLL, END, BUY_DEVELOPMENT, TRADE, CANCEL, SEND, ACCEPT, REJECT, COUNTER = range(9)
PLAY_DEVELOPMENT = 20
ROAD, SETTLEMENT, CITY = 100, 200, 300
RESOURCE, MARITIME, PARTNER, ROBBER, VICTIM = 400, 500, 600, 700, 800
GIVE_ADD, GIVE_REMOVE, RECEIVE_ADD, RECEIVE_REMOVE = 900, 910, 920, 930
ROAD_COST = (1, 1, 0, 0, 0)
SETTLEMENT_COST = (1, 1, 1, 1, 0)
CITY_COST = (0, 0, 0, 2, 3)
DEVELOPMENT_COST = (0, 0, 1, 1, 1)
NUMBER_TOKENS = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)


def _geometry() -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[list[int]],
    list[tuple[int, int]],
    list[int],
]:
    """Integer corner lattice avoids rounding or duplicate vertices/edges."""
    axial = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]
    corners = ((1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1))
    points = sorted({(2 * q + r + x, 3 * r + y) for q, r in axial for x, y in corners})
    lookup = {point: index for index, point in enumerate(points)}
    tiles = [[lookup[(2 * q + r + x, 3 * r + y)] for x, y in corners] for q, r in axial]
    occurrences = Counter(
        (min(v[i], v[(i + 1) % 6]), max(v[i], v[(i + 1) % 6]))
        for v in tiles
        for i in range(6)
    )
    edges = sorted(occurrences)
    coast = [i for i, e in enumerate(edges) if occurrences[e] == 1]
    coast.sort(
        key=lambda i: math.atan2(
            sum(points[v][1] for v in edges[i]), sum(points[v][0] for v in edges[i])
        )
    )
    return axial, points, tiles, edges, coast


AXIAL, VERTICES, TILE_VERTICES, EDGES, COAST = _geometry()
VERTEX_EDGES = [
    [i for i, e in enumerate(EDGES) if v in e] for v in range(len(VERTICES))
]
VERTEX_TILES = [
    [i for i, t in enumerate(TILE_VERTICES) if v in t] for v in range(len(VERTICES))
]
NEIGHBORS = [
    sorted({w for i in VERTEX_EDGES[v] for w in EDGES[i] if w != v})
    for v in range(len(VERTICES))
]


def _spiral() -> list[int]:
    result: list[int] = []
    for radius in (2, 1):
        q, r = radius, -radius
        for dq, dr in ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)):
            for _ in range(radius):
                result.append(AXIAL.index((q, r)))
                q, r = q + dq, r + dr
    return [*result, AXIAL.index((0, 0))]


class CatanGame(NativeGame):
    name = "catan"
    parameter_specification = {
        "players": 4,
        "starting_player": -1,
        "max_turns": 500,
        "max_actions_per_turn": 200,
    }
    rules = (
        "Base CATAN (3–4 players), 2015/2020 rules, combined trade/build variant. "
        "Terrain is dealt randomly from 3 brick, 4 lumber, 4 wool, 4 grain, 3 ore "
        "and 1 desert. Number tokens follow the standard alphabetical inward "
        "spiral skipping desert. Nine fixed coastal ports include four 3:1 ports "
        "and one 2:1 port per resource. Initial placement is clockwise then reverse; "
        "the second settlement supplies starting resources. Each turn roll two dice, "
        "resolve production or seven, then freely combine builds and trades. Roads "
        "cost brick+lumber; settlements cost brick+lumber+wool+grain; cities cost "
        "2 grain+3 ore; development cards cost wool+grain+ore. Respect settlement "
        "distance and connection rules and stock limits (15 roads,5 settlements,4 cities). "
        "Seven: players with over seven cards discard floor(half), then move robber "
        "to another hex and steal one random resource from an adjacent opponent. "
        "The bank starts with 19 of each resource. Production shortages pay nothing "
        "for that resource unless a single player is entitled, who receives the remainder. "
        "Maritime trades are 4:1 or owned port's 3:1/2:1. Domestic trades exchange "
        "arbitrary disjoint resource bundles with one partner; every trade involves "
        "the active player, requires both players' consent and existing resources. "
        "Use trade partner, edit give/receive quantities, send, accept/reject/counter. "
        "Drafts are private until sent. No gifts, loans, promises, or development-card trades. "
        "Only one non-VP development card per turn, never one bought that turn; "
        "it can be played before rolling. Knights move robber and compete for Largest "
        "Army (3+); Road Building places up to two free roads; Year of Plenty takes "
        "two available bank resources; Monopoly takes all opponents' named resources. "
        "Longest Road is the longest edge-simple trail of 5+, interrupted by enemy "
        "buildings; ties preserve the holder. Each bonus is 2 VP. Settlements are "
        "1 VP, cities 2 VP, hidden VP cards 1 VP. Reach 10 on your own turn to win. "
        "Declared max_turns and max_actions_per_turn tournament safety caps produce "
        "a draw. No expansions or optional helpers."
    )

    def __init__(self, parameters: dict[str, Any] | None = None):
        values = dict(parameters or {})
        validate_parameters(values, set(self.parameter_specification))
        players = integer_parameter(values, "players", 4, 3, 4)
        starting = integer_parameter(values, "starting_player", -1, -1, players - 1)
        turns = integer_parameter(values, "max_turns", 500, 1, 10000)
        actions = integer_parameter(values, "max_actions_per_turn", 200, 20, 10000)
        super().__init__(
            players,
            {
                "players": players,
                "starting_player": starting,
                "max_turns": turns,
                "max_actions_per_turn": actions,
            },
        )
        self.max_steps = 20 + 4 * players + turns * actions

    def new_initial_state(self) -> CatanState:
        return CatanState(self)


class CatanState(NativeState):
    def __init__(self, game: CatanGame):
        super().__init__(game)
        self.phase = "terrain"
        self.terrain: list[int] = []
        self.terrain_supply = [3, 4, 4, 4, 3, 1]
        self.numbers = [0] * 19
        self.robber = -1
        self.turn_player = game.parameters["starting_player"]
        self.turn_number = 0
        self.turn_actions = 0
        self.setup_order: list[int] = []
        self.setup_index = 0
        self.setup_vertex = -1
        self.building_owner = [-1] * 54
        self.building_level = [0] * 54
        self.road_owner = [-1] * 72
        self.hands = [[0] * 5 for _ in range(game.players)]
        self.bank = [19] * 5
        self.development_supply = [14, 2, 2, 2, 5]
        self.development = [[0] * 5 for _ in range(game.players)]
        self.new_development = [[0] * 5 for _ in range(game.players)]
        self.development_discard = [0] * 5
        self.knights = [0] * game.players
        self.development_played = False
        self.longest_road_owner: int | None = None
        self.largest_army_owner: int | None = None
        self.road_lengths = [0] * game.players
        self.dice: int | None = None
        self.discard_remaining = [0] * game.players
        self.discard_player = -1
        self.steal_victim = -1
        self.resume_phase = "main"
        self.free_roads = 0
        self.plenty_remaining = 0
        self.trade: dict[str, Any] | None = None
        self.trade_editor = -1
        self.winner: int | None = None
        self.end_reason: str | None = None
        self.last_event = "Building the island"
        self.public_history: list[str] = []
        self.private_events: list[list[str]] = [[] for _ in range(game.players)]
        self.ports: list[str | None] = [None] * 54
        for index, kind in zip(
            (0, 3, 7, 10, 13, 17, 20, 23, 27),
            ("any", "brick", "any", "lumber", "wool", "any", "grain", "any", "ore"),
            strict=True,
        ):
            for vertex in EDGES[COAST[index]]:
                self.ports[vertex] = kind

    def current_player(self) -> int:
        if self.phase == "terminal":
            return TERMINAL
        if self.phase in {
            "terrain",
            "starting_player",
            "dice",
            "development_draw",
            "steal",
        }:
            return CHANCE
        if self.phase.startswith("setup"):
            return self.setup_order[self.setup_index]
        if self.phase == "discard":
            return self.discard_player
        if self.phase == "trade_response":
            assert self.trade is not None
            return self.trade["recipient"]
        if self.phase == "trade_draft":
            return self.trade_editor
        return self.turn_player

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.phase == "terrain":
            supply = self.terrain_supply
        elif self.phase == "starting_player":
            return [(p, 1 / self.game.players) for p in range(self.game.players)]
        elif self.phase == "dice":
            return [(roll, (6 - abs(7 - roll)) / 36) for roll in range(2, 13)]
        elif self.phase == "development_draw":
            supply = self.development_supply
        elif self.phase == "steal":
            supply = self.hands[self.steal_victim]
        else:
            return []
        total = sum(supply)
        return [(i, n / total) for i, n in enumerate(supply) if n]

    def _settlement_sites(self, player: int, initial: bool = False) -> list[int]:
        if (
            self.building_owner.count(player)
            - sum(
                o == player and level == 2
                for o, level in zip(
                    self.building_owner, self.building_level, strict=True
                )
            )
            >= 5
        ):
            return []
        return [
            v
            for v in range(54)
            if self.building_owner[v] < 0
            and all(self.building_owner[w] < 0 for w in NEIGHBORS[v])
            and (initial or any(self.road_owner[e] == player for e in VERTEX_EDGES[v]))
        ]

    def _road_sites(self, player: int) -> list[int]:
        if self.road_owner.count(player) >= 15:
            return []

        def connected(vertex: int) -> bool:
            return self.building_owner[vertex] == player or (
                self.building_owner[vertex] < 0
                and any(self.road_owner[e] == player for e in VERTEX_EDGES[vertex])
            )

        return [
            e
            for e, (a, b) in enumerate(EDGES)
            if self.road_owner[e] < 0 and (connected(a) or connected(b))
        ]

    def _affords(self, player: int, cost: tuple[int, ...] | list[int]) -> bool:
        return all(
            have >= need for have, need in zip(self.hands[player], cost, strict=True)
        )

    def _ratio(self, player: int, resource: int) -> int:
        owned = {self.ports[v] for v in range(54) if self.building_owner[v] == player}
        return 2 if RESOURCES[resource] in owned else 3 if "any" in owned else 4

    def _development_actions(self) -> list[int]:
        if self.development_played:
            return []
        p = self.turn_player
        return [
            PLAY_DEVELOPMENT + i
            for i in range(4)
            if self.development[p][i] > self.new_development[p][i]
            and (i != 1 or self._road_sites(p))
            and (i != 2 or sum(self.bank))
        ]

    def legal_actions(self, player: int | None = None) -> list[int]:
        actor = self.current_player()
        if player is not None and player != actor:
            return []
        if actor == CHANCE:
            return [a for a, _ in self.chance_outcomes()]
        if actor == TERMINAL:
            return []
        if self.phase == "setup_settlement":
            return [SETTLEMENT + v for v in self._settlement_sites(actor, initial=True)]
        if self.phase == "setup_road":
            return [
                ROAD + e
                for e in VERTEX_EDGES[self.setup_vertex]
                if self.road_owner[e] < 0
            ]
        if self.phase == "pre_roll":
            return [ROLL, *self._development_actions()]
        if self.phase == "main":
            actions = [END]
            if sum(self.hands[actor]):
                actions.append(TRADE)
            if self._affords(actor, ROAD_COST):
                actions.extend(ROAD + e for e in self._road_sites(actor))
            if self._affords(actor, SETTLEMENT_COST):
                actions.extend(SETTLEMENT + v for v in self._settlement_sites(actor))
            if (
                self._affords(actor, CITY_COST)
                and sum(
                    owner == actor and level == 2
                    for owner, level in zip(
                        self.building_owner, self.building_level, strict=True
                    )
                )
                < 4
            ):
                actions.extend(
                    CITY + v
                    for v in range(54)
                    if self.building_owner[v] == actor and self.building_level[v] == 1
                )
            if self._affords(actor, DEVELOPMENT_COST) and sum(self.development_supply):
                actions.append(BUY_DEVELOPMENT)
            actions.extend(
                MARITIME + give * 5 + receive
                for give in range(5)
                for receive in range(5)
                if give != receive
                and self.hands[actor][give] >= self._ratio(actor, give)
                and self.bank[receive]
            )
            return [*actions, *self._development_actions()]
        if self.phase in {"discard", "plenty"}:
            supply = self.hands[actor] if self.phase == "discard" else self.bank
            return [RESOURCE + i for i, n in enumerate(supply) if n]
        if self.phase == "monopoly":
            return [RESOURCE + i for i in range(5)]
        if self.phase == "robber":
            return [ROBBER + i for i in range(19) if i != self.robber]
        if self.phase == "victim":
            return [VICTIM + p for p in self._victims()]
        if self.phase == "free_roads":
            return [ROAD + e for e in self._road_sites(actor)]
        if self.phase == "trade_partner":
            return [
                CANCEL,
                *(PARTNER + p for p in range(self.game.players) if p != actor),
            ]
        if self.phase == "trade_draft":
            assert self.trade is not None
            give, receive = self.trade["give"], self.trade["receive"]
            actions = [CANCEL]
            for r in range(5):
                if give[r] < self.hands[actor][r] and not receive[r]:
                    actions.append(GIVE_ADD + r)
                if give[r]:
                    actions.append(GIVE_REMOVE + r)
                if receive[r] < 19 and not give[r]:
                    actions.append(RECEIVE_ADD + r)
                if receive[r]:
                    actions.append(RECEIVE_REMOVE + r)
            if sum(give) and sum(receive):
                actions.append(SEND)
            return actions
        if self.phase == "trade_response":
            assert self.trade is not None
            actions = [REJECT]
            if sum(self.hands[actor]):
                actions.append(COUNTER)
            if self._affords(actor, self.trade["receive"]) and self._affords(
                self.trade["proposer"], self.trade["give"]
            ):
                actions.append(ACCEPT)
            return actions
        raise RuntimeError(f"Unknown CATAN phase {self.phase}")

    def _event(self, message: str) -> None:
        self.last_event = message
        self.public_history.append(message)

    def _pay(self, player: int, cost: tuple[int, ...]) -> None:
        for r, quantity in enumerate(cost):
            self.hands[player][r] -= quantity
            self.bank[r] += quantity

    def _give(self, player: int, resource: int, quantity: int) -> None:
        self.hands[player][resource] += quantity
        self.bank[resource] -= quantity

    def _start_setup(self, player: int) -> None:
        self.turn_player = player
        clockwise = [(player + i) % self.game.players for i in range(self.game.players)]
        self.setup_order = [*clockwise, *reversed(clockwise)]
        self.phase = "setup_settlement"

    def _finish_terrain(self) -> None:
        self.robber = self.terrain.index(5)
        tokens = iter(NUMBER_TOKENS)
        for tile in _spiral():
            if tile != self.robber:
                self.numbers[tile] = next(tokens)
        if self.turn_player < 0:
            self.phase = "starting_player"
        else:
            self._start_setup(self.turn_player)

    def _start_turn(self) -> None:
        self.turn_number += 1
        self.turn_actions = 0
        self.new_development[self.turn_player] = [0] * 5
        self.development_played = False
        self.dice = None
        self.phase = "pre_roll"
        self._event(f"P{self.turn_player} begins turn {self.turn_number}")
        self._check_winner()

    def _produce(self, roll: int) -> None:
        claims = [[0] * 5 for _ in range(self.game.players)]
        for tile, number in enumerate(self.numbers):
            if number == roll and tile != self.robber:
                r = self.terrain[tile]
                for vertex in TILE_VERTICES[tile]:
                    p = self.building_owner[vertex]
                    if p >= 0:
                        claims[p][r] += self.building_level[vertex]
        for r in range(5):
            entitled = [p for p in range(self.game.players) if claims[p][r]]
            total = sum(row[r] for row in claims)
            if total <= self.bank[r]:
                for p in entitled:
                    self._give(p, r, claims[p][r])
            elif len(entitled) == 1:
                self._give(entitled[0], r, self.bank[r])

    def _next_discard(self) -> None:
        remaining = [p for p, n in enumerate(self.discard_remaining) if n]
        if remaining:
            self.discard_player = remaining[0]
            self.phase = "discard"
        else:
            self.phase = "robber"

    def _victims(self) -> list[int]:
        return sorted(
            {
                self.building_owner[v]
                for v in TILE_VERTICES[self.robber]
                if self.building_owner[v] >= 0
                and self.building_owner[v] != self.turn_player
                and sum(self.hands[self.building_owner[v]])
            }
        )

    def _finish_robber(self) -> None:
        self.phase = self.resume_phase
        self._check_winner()

    def _longest(self, player: int) -> int:
        def walk(vertex: int, used: frozenset[int]) -> int:
            if used and self.building_owner[vertex] not in {-1, player}:
                return len(used)
            return max(
                [
                    len(used),
                    *(
                        walk(
                            EDGES[e][1] if EDGES[e][0] == vertex else EDGES[e][0],
                            used | {e},
                        )
                        for e in VERTEX_EDGES[vertex]
                        if self.road_owner[e] == player and e not in used
                    ),
                ]
            )

        return max((walk(v, frozenset()) for v in range(54)), default=0)

    @staticmethod
    def _bonus_holder(
        values: list[int], minimum: int, holder: int | None
    ) -> int | None:
        maximum = max(values)
        if maximum < minimum:
            return None
        tied = [p for p, value in enumerate(values) if value == maximum]
        return holder if holder in tied else tied[0] if len(tied) == 1 else None

    def _bonuses(self) -> None:
        self.road_lengths = [self._longest(p) for p in range(self.game.players)]
        self.longest_road_owner = self._bonus_holder(
            self.road_lengths, 5, self.longest_road_owner
        )
        self.largest_army_owner = self._bonus_holder(
            self.knights, 3, self.largest_army_owner
        )

    def _points(self, player: int, private: bool = False) -> int:
        points = sum(
            level
            for owner, level in zip(
                self.building_owner, self.building_level, strict=True
            )
            if owner == player
        )
        points += 2 * (self.longest_road_owner == player) + 2 * (
            self.largest_army_owner == player
        )
        return points + (self.development[player][4] if private else 0)

    def _check_winner(self) -> None:
        if (
            self.phase != "terminal"
            and self.turn_number
            and self._points(self.turn_player, private=True) >= 10
        ):
            self.winner = self.turn_player
            self.phase = "terminal"
            self.end_reason = "victory_points"
            self._event(
                f"P{self.turn_player} wins with {self._points(self.turn_player, private=True)} victory points"
            )

    def _finish_effect(self) -> None:
        self.phase = self.resume_phase
        self._check_winner()

    def apply_action(self, action: int) -> None:
        self.validate_action(action)
        actor = self.current_player()
        phase = self.phase
        if self.turn_number and phase != "terminal":
            self.turn_actions += 1
        if phase == "terrain":
            self.terrain.append(action)
            self.terrain_supply[action] -= 1
            if len(self.terrain) == 19:
                self._finish_terrain()
        elif phase == "starting_player":
            self._start_setup(action)
        elif phase == "setup_settlement":
            v = action - SETTLEMENT
            self.building_owner[v], self.building_level[v] = actor, 1
            self.setup_vertex = v
            if self.setup_index >= self.game.players:
                for tile in VERTEX_TILES[v]:
                    if self.terrain[tile] < 5:
                        self._give(actor, self.terrain[tile], 1)
            self.phase = "setup_road"
            self._event(f"P{actor} places starting settlement at vertex {v}")
        elif phase == "setup_road":
            self.road_owner[action - ROAD] = actor
            self.setup_index += 1
            if self.setup_index == 2 * self.game.players:
                self._start_turn()
            else:
                self.phase = "setup_settlement"
        elif phase == "pre_roll" and action == ROLL:
            self.phase = "dice"
        elif phase == "dice":
            self.dice = action
            self._event(f"P{self.turn_player} rolls {action}")
            if action == 7:
                self.discard_remaining = [
                    sum(hand) // 2 if sum(hand) > 7 else 0 for hand in self.hands
                ]
                self.resume_phase = "main"
                self._next_discard()
            else:
                self._produce(action)
                self.phase = "main"
        elif phase == "discard":
            r = action - RESOURCE
            self.hands[actor][r] -= 1
            self.bank[r] += 1
            self.discard_remaining[actor] -= 1
            if not self.discard_remaining[actor]:
                self._event(f"P{actor} completes required discard")
                self._next_discard()
        elif phase == "robber":
            self.robber = action - ROBBER
            self._event(f"P{actor} moves robber to tile {self.robber}")
            if self._victims():
                self.phase = "victim"
            else:
                self._finish_robber()
        elif phase == "victim":
            self.steal_victim = action - VICTIM
            self.phase = "steal"
        elif phase == "steal":
            p, victim = self.turn_player, self.steal_victim
            self.hands[victim][action] -= 1
            self.hands[p][action] += 1
            self.private_events[p].append(f"Stole {RESOURCES[action]} from P{victim}")
            self.private_events[victim].append(f"P{p} stole {RESOURCES[action]}")
            self._event(f"P{p} steals one resource from P{victim}")
            self._finish_robber()
        elif (
            phase in {"main", "pre_roll"}
            and PLAY_DEVELOPMENT <= action < PLAY_DEVELOPMENT + 4
        ):
            r = action - PLAY_DEVELOPMENT
            self.development[actor][r] -= 1
            self.development_discard[r] += 1
            self.development_played = True
            self.resume_phase = phase
            self._event(f"P{actor} plays {DEVELOPMENTS[r]}")
            if r == 0:
                self.knights[actor] += 1
                self._bonuses()
                self.phase = "robber"
            elif r == 1:
                self.free_roads = 2
                self.phase = "free_roads"
            elif r == 2:
                self.plenty_remaining = min(2, sum(self.bank))
                self.phase = "plenty"
            else:
                self.phase = "monopoly"
        elif phase == "free_roads":
            self.road_owner[action - ROAD] = actor
            self.free_roads -= 1
            self._bonuses()
            if not self.free_roads or not self._road_sites(actor):
                self._finish_effect()
            self._check_winner()
        elif phase == "plenty":
            self._give(actor, action - RESOURCE, 1)
            self.plenty_remaining -= 1
            if not self.plenty_remaining:
                self._finish_effect()
        elif phase == "monopoly":
            r = action - RESOURCE
            total = 0
            for p in range(self.game.players):
                if p != actor:
                    total += self.hands[p][r]
                    self.hands[p][r] = 0
            self.hands[actor][r] += total
            self._event(f"P{actor} collects {total} {RESOURCES[r]} with Monopoly")
            self._finish_effect()
        elif phase == "development_draw":
            self.development_supply[action] -= 1
            self.development[self.turn_player][action] += 1
            self.new_development[self.turn_player][action] += 1
            self.private_events[self.turn_player].append(
                f"Bought {DEVELOPMENTS[action]}"
            )
            self.phase = "main"
            self._check_winner()
        elif phase == "main":
            self._apply_main(action, actor)
        elif phase in {"trade_partner", "trade_draft", "trade_response"}:
            self._apply_trade(action, actor)
        else:
            raise RuntimeError(f"Unhandled CATAN transition {phase}")
        self._check_winner()
        if (
            self.phase != "terminal"
            and self.turn_number
            and self.turn_actions >= self.game.parameters["max_actions_per_turn"]
        ):
            self.phase, self.end_reason = "terminal", "action_limit"
            self._event("Declared actions-per-turn safety cap reached; draw")

    def _apply_main(self, action: int, actor: int) -> None:
        if action == END:
            if self.turn_number >= self.game.parameters["max_turns"]:
                self.phase, self.end_reason = "terminal", "turn_limit"
                self._event("Declared turn safety cap reached; draw")
            else:
                self.turn_player = (actor + 1) % self.game.players
                self._start_turn()
        elif action == BUY_DEVELOPMENT:
            self._pay(actor, DEVELOPMENT_COST)
            self.phase = "development_draw"
            self._event(f"P{actor} buys a development card")
        elif action == TRADE:
            self.phase = "trade_partner"
        elif ROAD <= action < ROAD + 72:
            self._pay(actor, ROAD_COST)
            self.road_owner[action - ROAD] = actor
            self._event(f"P{actor} builds road {action - ROAD}")
            self._bonuses()
        elif SETTLEMENT <= action < SETTLEMENT + 54:
            self._pay(actor, SETTLEMENT_COST)
            v = action - SETTLEMENT
            self.building_owner[v], self.building_level[v] = actor, 1
            self._event(f"P{actor} builds settlement {v}")
            self._bonuses()
        elif CITY <= action < CITY + 54:
            self._pay(actor, CITY_COST)
            self.building_level[action - CITY] = 2
            self._event(f"P{actor} upgrades vertex {action - CITY} to a city")
        elif MARITIME <= action < MARITIME + 25:
            give, receive = divmod(action - MARITIME, 5)
            ratio = self._ratio(actor, give)
            self.hands[actor][give] -= ratio
            self.bank[give] += ratio
            self._give(actor, receive, 1)
            self._event(
                f"P{actor} trades {ratio} {RESOURCES[give]} for 1 {RESOURCES[receive]} with bank"
            )
        self._check_winner()

    def _apply_trade(self, action: int, actor: int) -> None:
        if action in {CANCEL, REJECT}:
            if action == REJECT:
                self._event(f"P{actor} rejects trade")
            self.trade = None
            self.phase = "main"
        elif self.phase == "trade_partner":
            self.trade_editor = actor
            self.trade = {
                "proposer": actor,
                "recipient": action - PARTNER,
                "give": [0] * 5,
                "receive": [0] * 5,
                "status": "draft",
                "editor": actor,
            }
            self.phase = "trade_draft"
        elif action == COUNTER:
            assert self.trade is not None
            proposer = self.trade["proposer"]
            self.trade_editor = actor
            self.trade = {
                "proposer": actor,
                "recipient": proposer,
                "give": [0] * 5,
                "receive": [0] * 5,
                "status": "draft",
                "editor": actor,
            }
            self.phase = "trade_draft"
            self._event(f"P{actor} prepares a counteroffer")
        elif action == SEND:
            assert self.trade is not None
            self.trade["status"] = "offered"
            self.phase = "trade_response"
            self._event(
                f"P{actor} offers P{self.trade['recipient']} {self._bundle(self.trade['give'])} for {self._bundle(self.trade['receive'])}"
            )
        elif action == ACCEPT:
            assert self.trade is not None
            p, q = self.trade["proposer"], self.trade["recipient"]
            for r in range(5):
                delta = self.trade["give"][r] - self.trade["receive"][r]
                self.hands[p][r] -= delta
                self.hands[q][r] += delta
            self._event(
                f"P{q} accepts P{p}'s trade: {self._bundle(self.trade['give'])} for {self._bundle(self.trade['receive'])}"
            )
            self.trade = None
            self.phase = "main"
        else:
            assert self.trade is not None
            for start, side, change in (
                (GIVE_ADD, "give", 1),
                (GIVE_REMOVE, "give", -1),
                (RECEIVE_ADD, "receive", 1),
                (RECEIVE_REMOVE, "receive", -1),
            ):
                if start <= action < start + 5:
                    self.trade[side][action - start] += change
                    break

    @staticmethod
    def _bundle(values: list[int]) -> str:
        return (
            ", ".join(
                f"{quantity} {RESOURCES[r]}"
                for r, quantity in enumerate(values)
                if quantity
            )
            or "nothing"
        )

    def _trade_view(self, player: int | None = None) -> dict[str, Any] | None:
        if self.trade is None or (
            self.trade["status"] == "draft" and player != self.trade_editor
        ):
            return None
        return {
            **self.trade,
            "give": dict(zip(RESOURCES, self.trade["give"], strict=True)),
            "receive": dict(zip(RESOURCES, self.trade["receive"], strict=True)),
        }

    def public_state(self) -> dict[str, Any]:
        tiles = [
            {
                "id": i,
                "x": round(math.sqrt(3) * (q + r / 2), 6),
                "y": 1.5 * r,
                "resource": (
                    RESOURCES[self.terrain[i]] if self.terrain[i] < 5 else "desert"
                )
                if i < len(self.terrain)
                else "undealt",
                "number": self.numbers[i],
                "vertices": list(TILE_VERTICES[i]),
            }
            for i, (q, r) in enumerate(AXIAL)
        ]
        vertices = [
            {
                "id": i,
                "x": round(math.sqrt(3) * x / 2, 6),
                "y": y / 2,
                "owner": owner if owner >= 0 else None,
                "level": self.building_level[i],
                "port": self.ports[i],
            }
            for i, ((x, y), owner) in enumerate(
                zip(VERTICES, self.building_owner, strict=True)
            )
        ]
        edges = [
            {
                "id": i,
                "a": a,
                "b": b,
                "owner": self.road_owner[i] if self.road_owner[i] >= 0 else None,
            }
            for i, (a, b) in enumerate(EDGES)
        ]
        players = [
            {
                "seat": p,
                "resource_count": sum(self.hands[p]),
                "development_count": sum(self.development[p]),
                "settlements": sum(
                    o == p and level == 1
                    for o, level in zip(
                        self.building_owner, self.building_level, strict=True
                    )
                ),
                "cities": sum(
                    o == p and level == 2
                    for o, level in zip(
                        self.building_owner, self.building_level, strict=True
                    )
                ),
                "roads": self.road_owner.count(p),
                "knights": self.knights[p],
                "public_points": self._points(p),
                "longest_road": self.road_lengths[p],
            }
            for p in range(self.game.players)
        ]
        result: dict[str, Any] = {
            "schema_version": 1,
            "game": "catan",
            "phase": self.phase,
            "current_player": self.current_player(),
            "turn_player": self.turn_player,
            "turn_number": self.turn_number,
            "players": players,
            "board": {
                "tiles": tiles,
                "vertices": vertices,
                "edges": edges,
                "robber": self.robber,
            },
            "trade": self._trade_view(),
            "dice": self.dice,
            "last_event": self.last_event,
            "longest_road_owner": self.longest_road_owner,
            "largest_army_owner": self.largest_army_owner,
            "development_cards_remaining": sum(self.development_supply),
            "winner": self.winner,
            "end_reason": self.end_reason,
        }
        if self.phase == "terminal":
            result["final_points"] = [
                self._points(p, private=True) for p in range(self.game.players)
            ]
        return result

    def observation_string(self, player: int) -> str:
        self.validate_player(player)
        return json.dumps(
            {
                **self.public_state(),
                "player": player,
                "resources": dict(zip(RESOURCES, self.hands[player], strict=True)),
                "development_cards": dict(
                    zip(DEVELOPMENTS, self.development[player], strict=True)
                ),
                "bought_this_turn": dict(
                    zip(DEVELOPMENTS, self.new_development[player], strict=True)
                ),
                "victory_points": self._points(player, private=True),
                "private_events": list(self.private_events[player][-32:]),
                "trade": self._trade_view(player),
                "public_history": list(self.public_history[-32:]),
                "public_history_total": len(self.public_history),
            },
            sort_keys=True,
        )

    def returns(self) -> list[float]:
        return [float(p == self.winner) for p in range(self.game.players)]

    def action_to_string(self, player: int, action: int) -> str:
        if self.phase == "terrain":
            return f"Deal terrain {RESOURCES[action] if action < 5 else 'desert'}"
        if self.phase == "starting_player":
            return f"Starting player P{action}"
        if self.phase == "dice":
            return f"Dice total {action}"
        if self.phase == "development_draw":
            return "Draw hidden development card"
        if self.phase == "steal":
            return "Steal one hidden resource"
        labels = {
            ROLL: "Roll dice",
            END: "End turn",
            BUY_DEVELOPMENT: "Buy development card",
            TRADE: "Propose domestic trade",
            CANCEL: "Cancel trade",
            SEND: "Send trade offer",
            ACCEPT: "Accept trade offer",
            REJECT: "Reject trade offer",
            COUNTER: "Make counteroffer",
        }
        if action in labels:
            return labels[action]
        if PLAY_DEVELOPMENT <= action < PLAY_DEVELOPMENT + 4:
            return f"Play {DEVELOPMENTS[action - PLAY_DEVELOPMENT]}"
        if ROAD <= action < ROAD + 72:
            return f"Build road on edge {action - ROAD}"
        if SETTLEMENT <= action < SETTLEMENT + 54:
            return f"Build settlement at vertex {action - SETTLEMENT}"
        if CITY <= action < CITY + 54:
            return f"Upgrade vertex {action - CITY} to city"
        if RESOURCE <= action < RESOURCE + 5:
            verb = {
                "discard": "Discard",
                "plenty": "Take from bank",
                "monopoly": "Monopolize",
            }.get(self.phase, "Choose")
            return f"{verb} {RESOURCES[action - RESOURCE]}"
        if MARITIME <= action < MARITIME + 25:
            give, receive = divmod(action - MARITIME, 5)
            return f"Maritime trade {self._ratio(player, give)} {RESOURCES[give]} for 1 {RESOURCES[receive]}"
        if PARTNER <= action < PARTNER + self.game.players:
            return f"Trade with P{action - PARTNER}"
        if ROBBER <= action < ROBBER + 19:
            return f"Move robber to tile {action - ROBBER}"
        if VICTIM <= action < VICTIM + self.game.players:
            return f"Steal from P{action - VICTIM}"
        for start, description in (
            (GIVE_ADD, "Add one to offer"),
            (GIVE_REMOVE, "Remove one from offer"),
            (RECEIVE_ADD, "Request one"),
            (RECEIVE_REMOVE, "Remove one requested"),
        ):
            if start <= action < start + 5:
                return f"{description}: {RESOURCES[action - start]}"
        raise ValueError(f"Unknown CATAN action {action}")
