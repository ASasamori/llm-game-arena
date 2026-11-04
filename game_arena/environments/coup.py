"""Standalone base Coup rules with explicit chance and private influence.

The publisher-authored rulebook is mirrored at
https://officialgamerules.org/wp-content/uploads/2025/02/Coup-Rulebook.pdf.
Reformation and the optional two-player alternate deal are not implemented.
Clockwise reaction priority resolves the physical game's simultaneous calls.
"""

from __future__ import annotations

import json
from typing import Any

from game_arena.environments.base import (
    CHANCE,
    TERMINAL,
    NativeGame,
    NativeState,
    integer_parameter,
    validate_parameters,
)

ROLES = ("duke", "assassin", "captain", "ambassador", "contessa")
INCOME, FOREIGN_AID, TAX, EXCHANGE = 0, 1, 2, 3
COUP, ASSASSINATE, STEAL = 10, 20, 30
PASS, CHALLENGE = 100, 101
BLOCK_DUKE, BLOCK_CONTESSA, BLOCK_CAPTAIN, BLOCK_AMBASSADOR = 110, 111, 112, 113
PROVE, CONCEDE = 120, 121
LOSE, RETURN = 200, 300
BLOCK_ROLES = {
    BLOCK_DUKE: "duke",
    BLOCK_CONTESSA: "contessa",
    BLOCK_CAPTAIN: "captain",
    BLOCK_AMBASSADOR: "ambassador",
}


class CoupGame(NativeGame):
    """Two to six players, the standard fifteen-card deck, and no AI dependency."""

    name = "coup"
    parameter_specification = {"players": 3, "starting_player": 0, "max_turns": 200}
    rules = (
        "Base Coup: each player has two hidden influence cards from three copies "
        "each of Duke, Assassin, Captain, Ambassador and Contessa. Public coins "
        "start at two; with two players the starting player has one. On your turn "
        "take income (+1), foreign aid (+2; any opponent may block with Duke), "
        "tax (+3; claim Duke), exchange (claim Ambassador; draw two and return "
        "two), steal up to two coins (claim Captain; target may block with "
        "Captain or Ambassador), assassinate (pay three, claim Assassin; target "
        "may block with Contessa), or coup (pay seven; target loses influence). "
        "At ten or more coins you must coup. You may bluff any character claim. "
        "Other living players can challenge any character action or block; "
        "challenge windows are resolved before blocking. Reactions have clockwise "
        "priority, starting after the claimant. A challenged player may concede "
        "or prove the claimed role using a hidden card. A proved card returns "
        "to the deck and is replaced at random, and the challenger loses an "
        "influence. A conceded claim loses an influence. A successfully challenged "
        "action fails and its action cost is refunded; a blocked action's cost "
        "remains spent. Losing influence means choosing one of your hidden cards "
        "to reveal permanently. Losing both eliminates you and returns your coins. "
        "Only the last living player wins. The declared max_turns tournament cap "
        "ends an unfinished game as a draw with zero utility for every player. "
        "Influence and exchange slot numbers refer only to your own private view."
    )

    def __init__(self, parameters: dict[str, Any] | None = None):
        values = dict(parameters or {})
        validate_parameters(values, set(self.parameter_specification))
        players = integer_parameter(values, "players", 3, 2, 6)
        starting = integer_parameter(values, "starting_player", 0, 0, players - 1)
        turns = integer_parameter(values, "max_turns", 200, 1, 10000)
        super().__init__(
            players,
            {"players": players, "starting_player": starting, "max_turns": turns},
        )
        # Includes dealing, every reaction, proof/replacement and exchange choice.
        self.max_steps = 2 * players + turns * (4 * players + 20)

    def new_initial_state(self) -> CoupState:
        return CoupState(self)


class CoupState(NativeState):
    def __init__(self, game: CoupGame):
        super().__init__(game)
        self.phase = "deal"
        self.turn_player = game.parameters["starting_player"]
        self.hands: list[list[str]] = [[] for _ in range(game.players)]
        self.revealed: list[list[str]] = [[] for _ in range(game.players)]
        self.coins = [2] * game.players
        if game.players == 2:
            self.coins[self.turn_player] = 1
        self.court = [3] * len(ROLES)
        self.dealt = 0
        self.pending: dict[str, Any] = {}
        self.reactors: list[int] = []
        self.reaction_index = 0
        self.challenge: dict[str, Any] = {}
        self.loss_player = -1
        self.continuation = ""
        self.exchange_pool: list[str] = []
        self.exchange_drawn = 0
        self.exchange_returns = 0
        self.completed_turns = 0
        self.winner: int | None = None
        self.finish_reason: str | None = None
        self.history: list[dict[str, Any]] = []
        self.private_events: list[list[dict[str, Any]]] = [
            [] for _ in range(game.players)
        ]

    def current_player(self) -> int:
        if self.phase == "terminal":
            return TERMINAL
        if self.phase in {"deal", "replacement", "exchange_draw"}:
            return CHANCE
        if self.phase in {"action_challenge", "block", "block_challenge"}:
            return self.reactors[self.reaction_index]
        if self.phase == "defend":
            return self.challenge["claimant"]
        if self.phase == "lose":
            return self.loss_player
        return self.turn_player

    def _living(self) -> list[int]:
        return [player for player, hand in enumerate(self.hands) if hand]

    def _clockwise(self, after: int, exclude: set[int] | None = None) -> list[int]:
        excluded = exclude or set()
        return [
            player
            for offset in range(1, self.game.players + 1)
            if (player := (after + offset) % self.game.players) not in excluded
            and self.hands[player]
        ]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if not self.is_chance_node():
            return []
        total = sum(self.court)
        return [(role, count / total) for role, count in enumerate(self.court) if count]

    def legal_actions(self, player: int | None = None) -> list[int]:
        current = self.current_player()
        if player is not None:
            if (
                type(player) is not int
                or player < CHANCE
                or player >= self.game.players
            ):
                raise ValueError("Player index out of range")
            if player != current:
                return []
        if self.is_terminal():
            return []
        if self.is_chance_node():
            return [action for action, _ in self.chance_outcomes()]
        if self.phase == "turn":
            targets = [p for p in self._living() if p != current]
            coups = [COUP + p for p in targets] if self.coins[current] >= 7 else []
            if self.coins[current] >= 10:
                return coups
            actions = [INCOME, FOREIGN_AID, TAX, EXCHANGE, *coups]
            if self.coins[current] >= 3:
                actions.extend(ASSASSINATE + p for p in targets)
            actions.extend(STEAL + p for p in targets)
            return actions
        if self.phase in {"action_challenge", "block_challenge"}:
            return [PASS, CHALLENGE]
        if self.phase == "block":
            block_actions = {
                "foreign_aid": [BLOCK_DUKE],
                "assassinate": [BLOCK_CONTESSA],
                "steal": [BLOCK_CAPTAIN, BLOCK_AMBASSADOR],
            }
            return [PASS, *block_actions[self.pending["action"]]]
        if self.phase == "defend":
            role = self.challenge["role"]
            return [CONCEDE, PROVE] if role in self.hands[current] else [CONCEDE]
        if self.phase == "lose":
            return [LOSE + index for index in range(len(self.hands[current]))]
        if self.phase == "exchange_return":
            return [RETURN + index for index in range(len(self.exchange_pool))]
        raise RuntimeError(f"Unknown Coup phase {self.phase}")

    def action_to_string(self, player: int, action: int) -> str:
        if player == CHANCE:
            return (
                f"draw hidden {ROLES[action]}"
                if 0 <= action < len(ROLES)
                else "invalid draw"
            )
        labels = {
            INCOME: "income",
            FOREIGN_AID: "foreign aid",
            TAX: "tax (claim Duke)",
            EXCHANGE: "exchange (claim Ambassador)",
            PASS: "pass reaction",
            CHALLENGE: "challenge claim",
            BLOCK_DUKE: "block (claim Duke)",
            BLOCK_CONTESSA: "block (claim Contessa)",
            BLOCK_CAPTAIN: "block (claim Captain)",
            BLOCK_AMBASSADOR: "block (claim Ambassador)",
            PROVE: "prove claimed influence",
            CONCEDE: "concede challenge",
        }
        if action in labels:
            return labels[action]
        for base, label in (
            (COUP, "coup"),
            (ASSASSINATE, "assassinate (claim Assassin)"),
            (STEAL, "steal (claim Captain)"),
        ):
            if base <= action < base + self.game.players:
                return f"{label} player {action - base}"
        if LOSE <= action < LOSE + 2:
            return f"reveal influence slot {action - LOSE}"
        if RETURN <= action < RETURN + 4:
            return f"return exchange slot {action - RETURN}"
        return f"invalid action {action}"

    def apply_action(self, action: int) -> None:
        self.validate_action(action)
        phase = self.phase
        if self.is_chance_node():
            self._draw(action)
        elif phase == "turn":
            self._take_turn(action)
        elif phase in {"action_challenge", "block_challenge"}:
            if action == PASS:
                self._next_reactor()
            else:
                context = "action" if phase == "action_challenge" else "block"
                claimant = (
                    self.pending["actor"]
                    if context == "action"
                    else self.pending["blocker"]
                )
                role = (
                    self.pending["role"]
                    if context == "action"
                    else self.pending["block_role"]
                )
                self.challenge = {
                    "claimant": claimant,
                    "challenger": self.current_player(),
                    "role": role,
                    "context": context,
                }
                self.history.append({"event": "challenge", **self.challenge})
                self.phase = "defend"
        elif phase == "block":
            if action == PASS:
                self._next_reactor()
            else:
                blocker = self.current_player()
                role = BLOCK_ROLES[action]
                self.pending.update(blocker=blocker, block_role=role)
                self.history.append({"event": "block", "player": blocker, "role": role})
                self._reaction_window(
                    "block_challenge", self._clockwise(blocker, {blocker})
                )
        elif phase == "defend":
            self._defend(action)
        elif phase == "lose":
            player = self.loss_player
            role = self.hands[player].pop(action - LOSE)
            self.revealed[player].append(role)
            self.history.append(
                {"event": "influence_lost", "player": player, "role": role}
            )
            if not self.hands[player]:
                self.coins[player] = 0
                self.history.append({"event": "eliminated", "player": player})
            if not self._check_winner():
                self._continue(self.continuation)
        elif phase == "exchange_return":
            role = self.exchange_pool.pop(action - RETURN)
            self.court[ROLES.index(role)] += 1
            self.private_events[self.turn_player].append(
                {"event": "exchange_return", "role": role}
            )
            self.exchange_returns += 1
            if self.exchange_returns == 2:
                self.hands[self.turn_player] = list(self.exchange_pool)
                self.exchange_pool = []
                self.history.append(
                    {"event": "exchange_completed", "player": self.turn_player}
                )
                self._finish_turn()

    def _draw(self, action: int) -> None:
        self.court[action] -= 1
        role = ROLES[action]
        if self.phase == "deal":
            player = self.dealt // 2
            self.hands[player].append(role)
            self.private_events[player].append({"event": "deal", "role": role})
            self.dealt += 1
            if self.dealt == 2 * self.game.players:
                self.phase = "turn"
                self.history.append(
                    {"event": "game_started", "starting_player": self.turn_player}
                )
        elif self.phase == "replacement":
            claimant = self.challenge["claimant"]
            self.hands[claimant].append(role)
            self.private_events[claimant].append({"event": "replacement", "role": role})
            continuation = (
                "action_approved"
                if self.challenge["context"] == "action"
                else "block_upheld"
            )
            self._request_loss(self.challenge["challenger"], continuation)
        else:
            self.exchange_pool.append(role)
            self.exchange_drawn += 1
            self.private_events[self.turn_player].append(
                {"event": "exchange_draw", "role": role}
            )
            if self.exchange_drawn == 2:
                self.phase = "exchange_return"

    def _take_turn(self, action: int) -> None:
        actor = self.turn_player
        target: int | None = None
        cost = 0
        role: str | None = None
        if action == INCOME:
            self.coins[actor] += 1
            self.history.append(
                {"event": "action", "actor": actor, "action": "income", "target": None}
            )
            self._finish_turn()
            return
        if action == FOREIGN_AID:
            name = "foreign_aid"
        elif action == TAX:
            name, role = "tax", "duke"
        elif action == EXCHANGE:
            name, role = "exchange", "ambassador"
        elif COUP <= action < COUP + self.game.players:
            name, target, cost = "coup", action - COUP, 7
        elif ASSASSINATE <= action < ASSASSINATE + self.game.players:
            name, target, cost, role = (
                "assassinate",
                action - ASSASSINATE,
                3,
                "assassin",
            )
        else:
            name, target, role = "steal", action - STEAL, "captain"
        self.coins[actor] -= cost
        self.pending = {
            "actor": actor,
            "action": name,
            "target": target,
            "cost": cost,
            "role": role,
        }
        self.history.append({"event": "action", **self.pending})
        if name == "coup":
            self._request_loss(int(target), "finish")  # type: ignore[arg-type]
        elif role is not None:
            self._reaction_window("action_challenge", self._clockwise(actor, {actor}))
        else:
            self._open_blocks()

    def _reaction_window(self, phase: str, players: list[int]) -> None:
        self.phase, self.reactors, self.reaction_index = phase, players, 0
        if not players:
            self._reaction_complete()

    def _next_reactor(self) -> None:
        self.reaction_index += 1
        if self.reaction_index == len(self.reactors):
            self._reaction_complete()

    def _reaction_complete(self) -> None:
        if self.phase == "action_challenge":
            self._open_blocks()
        elif self.phase == "block":
            self._resolve_effect()
        elif self.phase == "block_challenge":
            self.history.append(
                {
                    "event": "action_blocked",
                    "actor": self.pending["actor"],
                    "blocker": self.pending["blocker"],
                }
            )
            self._finish_turn()

    def _defend(self, action: int) -> None:
        claimant, role = self.challenge["claimant"], self.challenge["role"]
        if action == PROVE:
            self.hands[claimant].remove(role)
            self.court[ROLES.index(role)] += 1
            self.history.append(
                {"event": "claim_proved", "player": claimant, "role": role}
            )
            self.phase = "replacement"
        else:
            self.history.append(
                {"event": "claim_conceded", "player": claimant, "role": role}
            )
            if self.challenge["context"] == "action":
                self.coins[self.pending["actor"]] += self.pending["cost"]
                self._request_loss(claimant, "action_cancelled")
            else:
                self._request_loss(claimant, "resume_action")

    def _open_blocks(self) -> None:
        name, target = self.pending["action"], self.pending["target"]
        if name == "foreign_aid":
            self._reaction_window(
                "block", self._clockwise(self.turn_player, {self.turn_player})
            )
        elif name in {"assassinate", "steal"} and self.hands[target]:
            self._reaction_window("block", [target])
        else:
            self._resolve_effect()

    def _resolve_effect(self) -> None:
        name, actor, target = (
            self.pending["action"],
            self.pending["actor"],
            self.pending["target"],
        )
        if name == "tax":
            self.coins[actor] += 3
        elif name == "foreign_aid":
            self.coins[actor] += 2
        elif name == "steal":
            amount = min(2, self.coins[target])
            self.coins[target] -= amount
            self.coins[actor] += amount
        elif name == "assassinate" and self.hands[target]:
            self._request_loss(target, "finish")
            return
        elif name == "exchange":
            self.exchange_pool = list(self.hands[actor])
            self.exchange_drawn = self.exchange_returns = 0
            self.phase = "exchange_draw"
            return
        self._finish_turn()

    def _request_loss(self, player: int, continuation: str) -> None:
        self.loss_player, self.continuation = player, continuation
        if self.hands[player]:
            self.phase = "lose"
        else:
            self._continue(continuation)

    def _continue(self, continuation: str) -> None:
        if continuation == "action_approved":
            self._open_blocks()
        elif continuation == "resume_action":
            self._resolve_effect()
        else:
            if continuation == "action_cancelled":
                self.history.append(
                    {"event": "action_cancelled", "actor": self.pending["actor"]}
                )
            elif continuation == "block_upheld":
                self.history.append(
                    {
                        "event": "action_blocked",
                        "actor": self.pending["actor"],
                        "blocker": self.pending["blocker"],
                    }
                )
            self._finish_turn()

    def _check_winner(self) -> bool:
        alive = self._living()
        if len(alive) == 1:
            self.winner, self.finish_reason, self.phase = (
                alive[0],
                "last_survivor",
                "terminal",
            )
            return True
        return False

    def _finish_turn(self) -> None:
        self.completed_turns += 1
        if self._check_winner():
            return
        if self.completed_turns >= self.game.parameters["max_turns"]:
            self.finish_reason, self.phase = "turn_limit_draw", "terminal"
            return
        self.turn_player = self._clockwise(self.turn_player, {self.turn_player})[0]
        self.pending, self.challenge = {}, {}
        self.reactors = []
        self.phase = "turn"

    def observation_string(self, player: int) -> str:
        self.validate_player(player)
        # Full deck counts, other hands and their private events never leave this view.
        view: dict[str, Any] = {
            "game": self.game.name,
            "phase": self.phase,
            "player": player,
            "current_player": self.current_player(),
            "turn_player": self.turn_player,
            "coins": list(self.coins),
            # A proved card briefly leaves its hand while chance replaces it;
            # proof does not lose influence and must not advertise elimination.
            "influence_counts": [2 - len(cards) for cards in self.revealed],
            "revealed_influence": self.revealed,
            "own_influence": list(self.hands[player]),
            "own_private_history": self.private_events[player],
            "public_history": self.history,
            "pending_action": self.pending,
            "challenge": self.challenge,
            "completed_turns": self.completed_turns,
            "max_turns": self.game.parameters["max_turns"],
            "winner": self.winner,
            "finish_reason": self.finish_reason,
        }
        if player == self.turn_player and self.phase in {
            "exchange_draw",
            "exchange_return",
        }:
            view["exchange_pool"] = list(self.exchange_pool)
            view["returns_remaining"] = 2 - self.exchange_returns
        return json.dumps(view, sort_keys=True, separators=(",", ":"))

    def returns(self) -> list[float]:
        if not self.is_terminal() or self.winner is None:
            return [0.0] * self.game.players
        return [
            1.0 if player == self.winner else -1.0
            for player in range(self.game.players)
        ]
