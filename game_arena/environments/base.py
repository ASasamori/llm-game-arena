"""Small model-independent contract for explicit, replayable game rules."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Self

CHANCE = -1
TERMINAL = -2
SIMULTANEOUS = -3


def integer_parameter(
    parameters: dict[str, Any], name: str, default: int, low: int, high: int
) -> int:
    value = parameters.get(name, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def validate_parameters(parameters: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")


class NativeGame:
    name = "native"
    rules = ""
    max_steps = 10000
    parameter_specification: dict[str, Any] = {}

    def __init__(self, players: int, parameters: dict[str, Any]):
        self.players = players
        self.parameters = parameters

    def num_players(self) -> int:
        return self.players

    def max_history_length(self) -> int:
        return self.max_steps

    def new_initial_state(self) -> NativeState:
        raise NotImplementedError


class NativeState:
    def __init__(self, game: NativeGame):
        self.game = game

    def get_game(self) -> NativeGame:
        return self.game

    def clone(self) -> Self:
        return deepcopy(self)

    def serialize(self) -> str:
        def encode(value: Any) -> Any:
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, tuple):
                return list(value)
            raise TypeError(
                f"Nonserializable environment state: {type(value).__name__}"
            )

        return json.dumps(
            {
                "game": self.game.name,
                "parameters": self.game.parameters,
                "state": {k: v for k, v in vars(self).items() if k != "game"},
            },
            sort_keys=True,
            separators=(",", ":"),
            default=encode,
        )

    def __str__(self) -> str:
        return self.serialize()

    def current_player(self) -> int:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        return self.current_player() == TERMINAL

    def is_chance_node(self) -> bool:
        return self.current_player() == CHANCE

    def is_simultaneous_node(self) -> bool:
        return self.current_player() == SIMULTANEOUS

    def is_mean_field_node(self) -> bool:
        return False

    def legal_actions(self, player: int | None = None) -> list[int]:
        raise NotImplementedError

    def action_to_string(self, player: int, action: int) -> str:
        raise NotImplementedError

    def apply_action(self, action: int) -> None:
        raise NotImplementedError

    def apply_actions(self, actions: list[int]) -> None:
        raise ValueError("This environment uses sequential actions")

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return []

    def observation_string(self, player: int) -> str:
        raise NotImplementedError

    def returns(self) -> list[float]:
        return [0.0] * self.game.num_players()

    def rewards(self) -> list[float]:
        return self.returns() if self.is_terminal() else [0.0] * self.game.num_players()

    def validate_action(self, action: int) -> None:
        if type(action) is not int or action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r} for current phase")

    def validate_player(self, player: int) -> None:
        if type(player) is not int or not 0 <= player < self.game.num_players():
            raise ValueError("Player index out of range")
