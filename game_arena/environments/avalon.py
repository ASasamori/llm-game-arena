"""Model-independent Avalon (original 2012 rules) with explicit chance.

Rules reference: the publisher's original rulebook, mirrored at
https://avalon.fun/pdfs/rules.pdf. This is the original game: Oberon is visible
to Merlin, but is unknown to the other evil players. Big Box character revisions
and modules such as Lady of the Lake are deliberately not enabled.

Discussion is a bounded arena protocol of public claims and accusations, rather
than unbounded real-time conversation. Team votes and quest cards remain secret
until their respective resolutions, even though submissions are serialized.
"""

from __future__ import annotations

from collections import Counter
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
    "merlin",
    "servant",
    "percival",
    "assassin",
    "minion",
    "morgana",
    "mordred",
    "oberon",
)
EVIL_ROLES = frozenset(("assassin", "minion", "morgana", "mordred", "oberon"))
EVIL_COUNTS = {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}
TEAM_SIZES = {
    5: (2, 3, 2, 3, 3),
    6: (2, 3, 4, 3, 4),
    7: (2, 3, 3, 4, 4),
    8: (3, 4, 4, 5, 5),
    9: (3, 4, 4, 5, 5),
    10: (3, 4, 4, 5, 5),
}
RULES_SOURCE = "https://avalon.fun/pdfs/rules.pdf"


class AvalonGame(NativeGame):
    name = "avalon"
    parameter_specification = {"players": 5, "roles": "", "discussion_rounds": 1}

    def __init__(self, parameters: dict[str, Any] | None = None):
        values = dict(parameters or {})
        validate_parameters(values, set(self.parameter_specification))
        players = integer_parameter(values, "players", 5, 5, 10)
        rounds = integer_parameter(values, "discussion_rounds", 1, 0, 3)
        role_spec = values.get("roles", "")
        if not isinstance(role_spec, str):
            raise ValueError("roles must be a pipe-separated string")
        evil_count = EVIL_COUNTS[players]
        if role_spec:
            roles = [role.strip().lower() for role in role_spec.split("|")]
        else:
            roles = ["merlin"] + ["servant"] * (players - evil_count - 1)
            roles += ["assassin"] + ["minion"] * (evil_count - 1)
        counts = Counter(roles)
        if len(roles) != players or set(roles) - set(ROLES):
            raise ValueError("roles must contain one supported role per player")
        if counts["merlin"] != 1 or counts["assassin"] != 1:
            raise ValueError("Avalon requires exactly one Merlin and one Assassin")
        for role in ("percival", "morgana", "mordred", "oberon"):
            if counts[role] > 1:
                raise ValueError(f"At most one {role} is supported")
        if sum(counts[role] for role in EVIL_ROLES) != evil_count:
            raise ValueError(
                f"A {players}-player game requires exactly {evil_count} evil roles"
            )
        # Deck order is canonical; explicit chance selects every assignment.
        self.role_deck = sorted(roles, key=ROLES.index)
        self.discussion_rounds = rounds
        super().__init__(
            players,
            {
                "players": players,
                "roles": "|".join(self.role_deck),
                "discussion_rounds": rounds,
            },
        )
        self.max_steps = (
            players
            + 1
            + 25 * (1 + players * (rounds + 1))
            + sum(TEAM_SIZES[players])
            + 1
        )
        self.rules = (
            "Avalon, original 2012 rules. Seats are numbered 0 through "
            f"{players - 1}. Public role deck: {', '.join(self.role_deck)}. "
            "Good wins three quests and then survives the Assassin's attempt to name Merlin. "
            "Evil wins three failed quests, five consecutive rejected teams for one quest, "
            "or correctly assassinating Merlin. A leader proposes an entire team; a strict "
            "majority must approve it, so tied approval votes reject. Approval votes become "
            "public together; quest cards reveal only the total failures, not each contributor. "
            "Good must play success. Evil may play success or fail. Quest four requires two "
            "fail cards with seven or more players; all other quests fail with one. "
            f"Team sizes for quests 1-5 are {TEAM_SIZES[players]}. Leaders rotate clockwise. "
            "Merlin sees evil except Mordred; original-rule Oberon is visible to Merlin. "
            "Evil knows fellow evil except Oberon; Oberon knows no allies. Percival sees "
            "Merlin and Morgana as indistinguishable candidates. Public claims may be false. "
            f"There are {rounds} structured discussion turns per player after each proposal. "
            "Only the stated private observation is available to your seat."
        )

    def new_initial_state(self) -> AvalonState:
        return AvalonState(self)


class AvalonState(NativeState):
    game: AvalonGame

    def __init__(self, game: AvalonGame):
        super().__init__(game)
        self.phase = "deal"
        self.remaining_roles = list(game.role_deck)
        self.roles: list[str] = []
        self.leader = 0
        self.proposal: list[int] = []
        self.votes: dict[int, bool] = {}
        self.quest_votes: dict[int, bool] = {}
        self.quest_results: list[dict[str, Any]] = []
        self.rejections = 0
        self.discussion_turn = 0
        self.public_history: list[dict[str, Any]] = []
        self.winner: str | None = None
        self.end_reason: str | None = None

    @property
    def quest_index(self) -> int:
        return len(self.quest_results)

    @property
    def team_size(self) -> int:
        return TEAM_SIZES[self.game.players][min(self.quest_index, 4)]

    def current_player(self) -> int:
        if self.winner is not None:
            return TERMINAL
        if self.phase in {"deal", "leader"}:
            return CHANCE
        if self.phase == "proposal":
            return self.leader
        if self.phase == "discussion":
            return (self.leader + self.discussion_turn) % self.game.players
        if self.phase == "vote":
            return len(self.votes)
        if self.phase == "quest":
            return self.proposal[len(self.quest_votes)]
        if self.phase == "assassinate":
            return self.roles.index("assassin")
        raise RuntimeError(f"Unknown Avalon phase: {self.phase}")

    def chance_outcomes(self) -> list[tuple[int, float]]:
        if self.phase == "deal":
            counts = Counter(self.remaining_roles)
            total = len(self.remaining_roles)
            return [
                (ROLES.index(role), counts[role] / total)
                for role in ROLES
                if counts[role]
            ]
        if self.phase == "leader":
            return [
                (player, 1.0 / self.game.players) for player in range(self.game.players)
            ]
        return []

    def _teams(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.game.players), self.team_size))

    def legal_actions(self, player: int | None = None) -> list[int]:
        current = self.current_player()
        if current == TERMINAL or (player is not None and player != current):
            return []
        if current == CHANCE:
            return [action for action, _ in self.chance_outcomes()]
        if self.phase == "proposal":
            return list(range(len(self._teams())))
        if self.phase == "discussion":
            return list(range(3 + len(ROLES) + 2 * self.game.players))
        if self.phase == "vote":
            return [0, 1]
        if self.phase == "quest":
            return [0, 1] if self.roles[current] in EVIL_ROLES else [0]
        if self.phase == "assassinate":
            # Filtering by actual Good alignment would expose hidden Oberon.
            return [target for target in range(self.game.players) if target != current]
        return []

    def action_to_string(self, player: int, action: int) -> str:
        if self.phase == "deal":
            return f"deal_role:{ROLES[action]}"
        if self.phase == "leader":
            return f"initial_leader:{action}"
        if self.phase == "proposal":
            return "propose_team:" + ",".join(map(str, self._teams()[action]))
        if self.phase == "discussion":
            if action < 3:
                return ("pass", "claim_alignment:good", "claim_alignment:evil")[action]
            action -= 3
            if action < len(ROLES):
                return "claim_role:" + ROLES[action]
            action -= len(ROLES)
            if action < self.game.players:
                return f"accuse_player:{action}"
            return f"trust_player:{action - self.game.players}"
        if self.phase == "vote":
            return "approve_team" if action else "reject_team"
        if self.phase == "quest":
            return "quest_fail" if action else "quest_success"
        if self.phase == "assassinate":
            return f"assassinate_player:{action}"
        raise ValueError("Terminal state has no action labels")

    def apply_action(self, action: int) -> None:
        self.validate_action(action)
        actor = self.current_player()
        if self.phase == "deal":
            role = ROLES[action]
            self.roles.append(role)
            self.remaining_roles.remove(role)
            if not self.remaining_roles:
                self.phase = "leader"
        elif self.phase == "leader":
            self.leader = action
            self.phase = "proposal"
        elif self.phase == "proposal":
            self.proposal = list(self._teams()[action])
            self.public_history.append(
                {
                    "kind": "proposal",
                    "quest": self.quest_index + 1,
                    "leader": actor,
                    "team": list(self.proposal),
                }
            )
            self.discussion_turn = 0
            self.phase = "discussion" if self.game.discussion_rounds else "vote"
        elif self.phase == "discussion":
            self.public_history.append(
                {
                    "kind": "discussion",
                    "player": actor,
                    "statement": self.action_to_string(actor, action),
                }
            )
            self.discussion_turn += 1
            if self.discussion_turn >= self.game.discussion_rounds * self.game.players:
                self.phase = "vote"
        elif self.phase == "vote":
            self.votes[actor] = bool(action)
            if len(self.votes) == self.game.players:
                self._resolve_approval()
        elif self.phase == "quest":
            self.quest_votes[actor] = bool(action)
            if len(self.quest_votes) == len(self.proposal):
                self._resolve_quest()
        elif self.phase == "assassinate":
            self.public_history.append(
                {"kind": "assassination", "player": actor, "target": action}
            )
            self._finish(
                "evil" if self.roles[action] == "merlin" else "good",
                "merlin_assassinated"
                if self.roles[action] == "merlin"
                else "assassination_missed",
            )

    def _resolve_approval(self) -> None:
        approved = sum(self.votes.values()) > self.game.players / 2
        self.public_history.append(
            {
                "kind": "team_vote",
                "quest": self.quest_index + 1,
                "leader": self.leader,
                "team": list(self.proposal),
                "votes": [self.votes[p] for p in range(self.game.players)],
                "approved": approved,
            }
        )
        self.votes = {}
        if approved:
            self.rejections = 0
            self.quest_votes = {}
            self.phase = "quest"
            return
        self.rejections += 1
        if self.rejections == 5:
            self._finish("evil", "five_rejected_teams")
            return
        self.leader = (self.leader + 1) % self.game.players
        self.proposal = []
        self.phase = "proposal"

    def _resolve_quest(self) -> None:
        failures = sum(self.quest_votes.values())
        threshold = 2 if self.game.players >= 7 and self.quest_index == 3 else 1
        result = {
            "quest": self.quest_index + 1,
            "team": list(self.proposal),
            "fails": failures,
            "fails_required": threshold,
            "success": failures < threshold,
        }
        self.quest_results.append(result)
        self.public_history.append({"kind": "quest_result", **result})
        self.quest_votes = {}
        successes = sum(result["success"] for result in self.quest_results)
        failed = len(self.quest_results) - successes
        if failed == 3:
            self._finish("evil", "three_failed_quests")
        elif successes == 3:
            self.phase = "assassinate"
        else:
            self.leader = (self.leader + 1) % self.game.players
            self.proposal = []
            self.phase = "proposal"

    def _finish(self, winner: str, reason: str) -> None:
        self.winner, self.end_reason, self.phase = winner, reason, "terminal"

    def observation_string(self, player: int) -> str:
        self.validate_player(player)
        own_role = self.roles[player] if len(self.roles) > player else None
        knowledge: dict[str, Any] = {}
        if len(self.roles) == self.game.players:
            if own_role == "merlin":
                knowledge["known_evil"] = [
                    p
                    for p, role in enumerate(self.roles)
                    if role in EVIL_ROLES and role != "mordred"
                ]
            if own_role in EVIL_ROLES and own_role != "oberon":
                knowledge["evil_allies"] = [
                    p
                    for p, role in enumerate(self.roles)
                    if role in EVIL_ROLES and role != "oberon" and p != player
                ]
            if own_role == "percival":
                knowledge["merlin_candidates"] = [
                    p
                    for p, role in enumerate(self.roles)
                    if role in {"merlin", "morgana"}
                ]
        view = {
            "game": "avalon",
            "ruleset": "original_2012",
            "player": player,
            "role": own_role,
            "alignment": None
            if own_role is None
            else "evil"
            if own_role in EVIL_ROLES
            else "good",
            "knowledge": knowledge,
            "phase": self.phase,
            "current_player": self.current_player(),
            "leader": self.leader,
            "quest_number": self.quest_index + 1,
            "team_size": self.team_size,
            "proposed_team": list(self.proposal),
            "consecutive_rejections": self.rejections,
            "quest_results": self.quest_results,
            "public_history": self.public_history,
            "own_approval_vote": self.votes.get(player),
            "own_quest_card": self.quest_votes.get(player),
            "winner": self.winner,
            "end_reason": self.end_reason,
        }
        # Terminal results reveal roles for audit; no future decision can use them.
        if self.is_terminal():
            view["revealed_roles"] = list(self.roles)
        return json.dumps(view, sort_keys=True, separators=(",", ":"))

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0] * self.game.players
        return [
            1.0 if ("evil" if role in EVIL_ROLES else "good") == self.winner else -1.0
            for role in self.roles
        ]
