"""Compatibility bridge into the existing runner; native rules remain independent."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pyspiel

from game_arena.environments.base import NativeGame, NativeState
from game_arena.environments.registry import canonical_spec


class ArenaGame:
    environment_backend = "native"

    def __init__(self, game: NativeGame):
        self.native = game
        self.rules = game.rules

    def get_type(self) -> Any:
        return SimpleNamespace(
            short_name=self.native.name,
            long_name=self.native.name,
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            min_num_players=self.native.players,
            max_num_players=self.native.players,
            default_loadable=True,
            is_concrete=True,
            parameter_specification=self.native.parameter_specification,
            provides_information_state_string=False,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=False,
        )

    def num_players(self) -> int:
        return self.native.num_players()

    def new_initial_state(self) -> ArenaState:
        return ArenaState(self, self.native.new_initial_state())

    def max_history_length(self) -> int:
        return self.native.max_history_length()

    def get_parameters(self) -> dict[str, Any]:
        return self.native.parameters

    def to_string(self) -> str:
        return canonical_spec(self.native)


class ArenaState:
    def __init__(self, game: ArenaGame, state: NativeState):
        self.game = game
        self.native = state

    def get_game(self) -> ArenaGame:
        return self.game

    def clone(self) -> ArenaState:
        return ArenaState(self.game, self.native.clone())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def __str__(self) -> str:
        return str(self.native)
