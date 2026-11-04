"""Model-independent One Night Werewolf, using the Bezier base-game rules.

All base-game roles except Doppelganger are supported. Discussion is a bounded
sequence of public claims/accusations rather than unbounded natural-language
chat. The English rulebook's no-wolf Minion condition is used literally: another
player must die; the Minion need not survive a tied execution/Hunter shot.

Rules: https://cdn.shopify.com/s/files/1/0740/4855/files/ONUW_rules-updated_for_BGG.pdf
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations
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

ROLES = (
    "werewolf",
    "minion",
    "mason",
    "seer",
    "robber",
    "troublemaker",
    "drunk",
    "insomniac",
    "villager",
    "tanner",
    "hunter",
)
ROLE_IDS = {role: index for index, role in enumerate(ROLES)}
NIGHT_ORDER = ROLES[:8]
CARD_LIMITS = {role: 1 for role in ROLES} | {
    "werewolf": 2,
    "mason": 2,
    "villager": 3,
}
DEFAULT_DECK = (
    "werewolf",
    "werewolf",
    "seer",
    "robber",
    "troublemaker",
    "villager",
    "villager",
    "villager",
    "minion",
    "tanner",
    "hunter",
    "drunk",
    "insomniac",
)


class WerewolfGame(NativeGame):
    """Three to ten players, exactly three undealt center cards, one night."""

    name = "one_night_werewolf"
    parameter_specification = {
        "players": 3,
        "roles": "",
        "discussion_rounds": 1,
        "lone_wolf": True,
    }
    rules = (
        "One Night Werewolf: players are numbered from zero. Your initial role "
        "determines your night action, but your final card determines your team. "
        "Two werewolves may be in the deck; there are three hidden center cards. "
        "Night order: werewolves, minion, masons, seer, robber, troublemaker, "
        "drunk, insomniac. A lone werewolf may inspect one center card when enabled. "
        "Seer may inspect one other player or two center cards. Robber may swap "
        "with another player and view the taken card. Troublemaker may swap two "
        "other players without viewing. Drunk must swap with a center card without "
        "viewing. Insomniac views their current card. Public claims may be lies. "
        "After bounded public discussion, secretly vote for another player. "
        "All tied highest-vote players die if the highest count is at least two; "
        "otherwise nobody dies. A killed hunter also kills their vote target. "
        "Village wins if any final werewolf dies, or no final werewolves exist "
        "and nobody dies. Werewolf/minion team wins when final werewolves exist "
        "and no werewolf or tanner dies. Without final werewolves, a minion wins "
        "if some other player dies and the tanner does not. Tanner wins only by "
        "dying; village can share a tanner win only if a werewolf also dies. "
        "Doppelganger and expansion roles are unsupported. Select a listed action id."
    )

    def __init__(self, parameters: dict[str, Any] | None = None):
        supplied = dict(parameters or {})
        validate_parameters(supplied, set(self.parameter_specification))
        players = integer_parameter(supplied, "players", 3, 3, 10)
        rounds = integer_parameter(supplied, "discussion_rounds", 1, 0, 3)
        lone_wolf = supplied.get("lone_wolf", True)
        if type(lone_wolf) is not bool:
            raise ValueError("lone_wolf must be a boolean")
        value = supplied.get("roles", "")
        if value == "":
            deck = list(DEFAULT_DECK[: players + 3])
        elif isinstance(value, str):
            deck = [role.strip().lower() for role in value.split("|")]
        elif isinstance(value, list) and all(isinstance(role, str) for role in value):
            deck = [role.strip().lower() for role in value]
        else:
            raise ValueError(
                "roles must be a pipe-separated string or a list of role names"
            )
        if len(deck) != players + 3:
            raise ValueError("roles must contain exactly players + 3 cards")
        unknown = set(deck) - set(ROLES)
        if unknown:
            raise ValueError(f"Unsupported roles: {', '.join(sorted(unknown))}")
        counts = Counter(deck)
        for role, count in counts.items():
            if count > CARD_LIMITS[role]:
                raise ValueError(
                    f"At most {CARD_LIMITS[role]} {role} cards are supported"
                )
        if counts["mason"] not in (0, 2):
            raise ValueError("Masons must be included as a pair")
        if counts["insomniac"] and not (counts["robber"] or counts["troublemaker"]):
            raise ValueError(
                "Insomniac requires robber and/or troublemaker in the deck"
            )
        self.role_deck = tuple(sorted(deck, key=ROLE_IDS.__getitem__))
        self.discussion_rounds = rounds
        self.lone_wolf = lone_wolf
        self.max_steps = players + 3 + players + rounds * players + players
        super().__init__(
            players,
            {
                "players": players,
                "roles": "|".join(self.role_deck),
                "discussion_rounds": rounds,
                "lone_wolf": lone_wolf,
            },
        )

    def new_initial_state(self) -> WerewolfState:
        return WerewolfState(self)


class WerewolfState(NativeState):
    """Auditable private state; only observation_string is a player-safe view."""

    def __init__(self, game: WerewolfGame):
        super().__init__(game)
        self.phase = "deal"
        self.cards: list[str] = []
        self.original_roles: list[str] = []
        self.remaining = dict(Counter(game.role_deck))
        self.night_queue: list[tuple[int, str]] = []
        self.night_index = 0
        self.actor = 0
        self.discussion_turn = 0
        self.private_findings: list[list[dict[str, Any]]] = [
            [] for _ in range(game.players)
        ]
        self.public_transcript: list[dict[str, Any]] = []
        self.votes: list[int | None] = [None] * game.players
        self.dead: list[int] = []
        self.winners: list[int] = []

    @property
    def werewolf_game(self) -> WerewolfGame:
        assert isinstance(self.game, WerewolfGame)
        return self.game

    def current_player(self) -> int:
        if self.phase == "deal":
            return CHANCE
        if self.phase == "terminal":
            return TERMINAL
        if self.phase == "night":
            return self.night_queue[self.night_index][0]
        return self.actor

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.phase != "deal":
            return []
        total = sum(self.remaining.values())
        return [
            (ROLE_IDS[role], self.remaining[role] / total)
            for role in ROLES
            if self.remaining.get(role, 0)
        ]

    def _actions(self) -> dict[int, str]:
        n = self.game.players
        if self.phase == "deal":
            return {
                action: "deal " + ROLES[action] for action, _ in self.chance_outcomes()
            }
        if self.phase == "terminal":
            return {}
        actor = self.current_player()
        if self.phase == "vote":
            return {1000 + p: f"vote player {p}" for p in range(n) if p != actor}
        if self.phase == "discussion":
            return (
                {0: "pass discussion"}
                | {
                    500 + ROLE_IDS[r]: f"claim initial role {r}"
                    for r in ROLES
                    if r in self.remaining or r in self.werewolf_game.role_deck
                }
                | {600 + p: f"accuse player {p} of being a werewolf" for p in range(n)}
            )
        role = self.night_queue[self.night_index][1]
        actions = {0: "finish night observation"}
        if role == "werewolf":
            wolves = [p for p, r in enumerate(self.original_roles) if r == "werewolf"]
            if len(wolves) == 1 and self.werewolf_game.lone_wolf:
                actions = {0: "decline lone-wolf center inspection"} | {
                    400 + i: f"inspect center card {i}" for i in range(3)
                }
        elif role == "seer":
            actions = (
                {0: "decline seer inspection"}
                | {100 + p: f"inspect player {p}" for p in range(n) if p != actor}
                | {
                    200 + i: f"inspect center cards {a} and {b}"
                    for i, (a, b) in enumerate(combinations(range(3), 2))
                }
            )
        elif role == "robber":
            actions = {0: "decline robbery"} | {
                100 + p: f"rob player {p}" for p in range(n) if p != actor
            }
        elif role == "troublemaker":
            actions = {0: "decline troublemaking"} | {
                300 + a * n + b: f"swap players {a} and {b}"
                for a, b in combinations([p for p in range(n) if p != actor], 2)
            }
        elif role == "drunk":
            actions = {
                400 + i: f"exchange with center card {i} without looking"
                for i in range(3)
            }
        return actions

    def legal_actions(self, player: int | None = None) -> list[int]:
        if player is not None and (
            type(player) is not int or player != self.current_player()
        ):
            return []
        return sorted(self._actions())

    def action_to_string(self, player: int, action: int) -> str:
        if type(action) is not int or action not in self._actions():
            raise ValueError("Unknown action for current phase")
        if player != self.current_player():
            raise ValueError("Action belongs to a different player")
        return self._actions()[action]

    def apply_action(self, action: int) -> None:
        self.validate_action(action)
        n = self.game.players
        if self.phase == "deal":
            role = ROLES[action]
            self.cards.append(role)
            self.remaining[role] -= 1
            if len(self.cards) == n + 3:
                self.original_roles = list(self.cards[:n])
                self.night_queue = [
                    (p, role)
                    for role in NIGHT_ORDER
                    for p, dealt in enumerate(self.original_roles)
                    if dealt == role
                ]
                if self.night_queue:
                    self.phase = "night"
                else:
                    self._start_day()
            return
        actor = self.current_player()
        if self.phase == "night":
            self._night_action(actor, self.night_queue[self.night_index][1], action)
            self.night_index += 1
            if self.night_index == len(self.night_queue):
                self._start_day()
        elif self.phase == "discussion":
            self.public_transcript.append(
                {
                    "player": actor,
                    "statement": self._actions()[action],
                    "action": action,
                }
            )
            self.discussion_turn += 1
            self.actor = self.discussion_turn % n
            if self.discussion_turn == n * self.werewolf_game.discussion_rounds:
                self.phase = "vote"
                self.actor = 0
        elif self.phase == "vote":
            self.votes[actor] = action - 1000
            self.actor += 1
            if self.actor == n:
                self._finish()

    def _start_day(self) -> None:
        self.phase = "discussion" if self.werewolf_game.discussion_rounds else "vote"
        self.actor = 0

    def _night_action(self, actor: int, role: str, action: int) -> None:
        n = self.game.players
        findings = self.private_findings[actor]
        if role in {"werewolf", "minion", "mason"}:
            sought = "mason" if role == "mason" else "werewolf"
            findings.append(
                {
                    "kind": sought + "_players",
                    "players": [
                        p
                        for p, r in enumerate(self.original_roles)
                        if r == sought and p != actor
                    ],
                }
            )
            if role == "werewolf" and action >= 400:
                center = action - 400
                findings.append(
                    {
                        "kind": "center_card",
                        "index": center,
                        "role": self.cards[n + center],
                    }
                )
        elif role == "seer" and action >= 200:
            pair = list(combinations(range(3), 2))[action - 200]
            findings.append(
                {
                    "kind": "center_cards",
                    "cards": [{"index": i, "role": self.cards[n + i]} for i in pair],
                }
            )
        elif role == "seer" and action >= 100:
            target = action - 100
            findings.append(
                {"kind": "player_card", "player": target, "role": self.cards[target]}
            )
        elif role == "robber" and action >= 100:
            target = action - 100
            self.cards[actor], self.cards[target] = (
                self.cards[target],
                self.cards[actor],
            )
            findings.append(
                {"kind": "robbed", "player": target, "role": self.cards[actor]}
            )
        elif role == "troublemaker" and action >= 300:
            a, b = divmod(action - 300, n)
            self.cards[a], self.cards[b] = self.cards[b], self.cards[a]
            findings.append({"kind": "swapped_players", "players": [a, b]})
        elif role == "drunk":
            center = action - 400
            self.cards[actor], self.cards[n + center] = (
                self.cards[n + center],
                self.cards[actor],
            )
            findings.append({"kind": "exchanged_with_center", "index": center})
        elif role == "insomniac":
            findings.append({"kind": "own_current_card", "role": self.cards[actor]})

    def _finish(self) -> None:
        counts = Counter(self.votes)
        highest = max(counts.values())
        dead = (
            {p for p in range(self.game.players) if counts[p] == highest}
            if highest >= 2
            else set()
        )
        hunters = {p for p in dead if self.cards[p] == "hunter"}
        for hunter in hunters:
            target = self.votes[hunter]
            assert target is not None
            dead.add(target)
        wolves = {p for p in range(self.game.players) if self.cards[p] == "werewolf"}
        killed_wolf = bool(wolves & dead)
        killed_tanner = any(self.cards[p] == "tanner" for p in dead)
        village_wins = killed_wolf or (not wolves and not dead)
        wolf_wins = bool(wolves) and not killed_wolf and not killed_tanner
        winners = []
        for p in range(self.game.players):
            role = self.cards[p]
            if role == "tanner":
                wins = p in dead
            elif role == "werewolf":
                wins = wolf_wins
            elif role == "minion":
                wins = wolf_wins if wolves else bool(dead - {p}) and not killed_tanner
            else:
                wins = village_wins
            if wins:
                winners.append(p)
        self.dead = sorted(dead)
        self.winners = winners
        self.phase = "terminal"

    def _public_view(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "game": self.game.name,
            "players": self.game.players,
            "phase": self.phase,
            "role_deck": list(self.werewolf_game.role_deck),
            "lone_wolf": self.werewolf_game.lone_wolf,
            "discussion_rounds": self.werewolf_game.discussion_rounds,
            "transcript": self.public_transcript,
        }
        if self.phase in {"discussion", "vote"}:
            result["discussion_turn"] = self.discussion_turn
        if self.is_terminal():
            result.update(
                final_roles=self.cards[: self.game.players],
                center_cards=self.cards[self.game.players :],
                votes=self.votes,
                dead=self.dead,
                winners=self.winners,
            )
        return result

    def observation_string(self, player: int) -> str:
        self.validate_player(player)
        result = self._public_view()
        result["player"] = player
        result["initial_role"] = (
            self.cards[player]
            if self.phase == "deal" and player < len(self.cards)
            else self.original_roles[player]
            if self.original_roles
            else None
        )
        result["private_findings"] = self.private_findings[player]
        result["own_vote"] = self.votes[player]
        return json.dumps(result, sort_keys=True, separators=(",", ":"))

    def __str__(self) -> str:
        return json.dumps(self._public_view(), sort_keys=True, separators=(",", ":"))

    def public_state(self) -> dict[str, Any]:
        # Night actor order can identify original roles. The public table only
        # knows that the village is asleep, not which seat is taking an action.
        view = self._public_view()
        players: list[dict[str, Any]] = [
            {"seat": seat} for seat in range(self.game.players)
        ]
        if self.is_terminal():
            for player in players:
                seat = player["seat"]
                player.update(
                    role=self.cards[seat],
                    dead=seat in self.dead,
                    winner=seat in self.winners,
                )
        view.update(
            schema_version=1,
            current_player=None if self.phase == "night" else self.current_player(),
            players=players,
            public_history=view.pop("transcript")[-32:],
            public_history_total=len(self.public_transcript),
        )
        return deepcopy(view)

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0] * self.game.players
        return [1.0 if p in self.winners else -1.0 for p in range(self.game.players)]
