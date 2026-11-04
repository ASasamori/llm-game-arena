"""Strict configuration parsing and discovery, without importing model SDKs."""

from __future__ import annotations

import ast
import json
from typing import Any

from game_arena.environments.base import NativeGame

ALIASES = {
    "werewolf": "one_night_werewolf",
    "one_night_ultimate_werewolf": "one_night_werewolf",
}
NAMES = ("one_night_werewolf", "avalon", "coup")


def native_name(spec: str) -> str | None:
    name = spec.strip().split("(", 1)[0].strip()
    name = ALIASES.get(name, name)
    return name if name in NAMES else None


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    name = native_name(spec)
    if name is None:
        raise ValueError("Unknown native environment")
    suffix = spec.strip()[len(spec.strip().split("(", 1)[0]) :]
    try:
        expression = ast.parse("environment" + (suffix or "()"), mode="eval").body
        if not isinstance(expression, ast.Call) or expression.args:
            raise ValueError("Use named game parameters")
        result: dict[str, Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None or keyword.arg in result:
                raise ValueError("Duplicate or expanded parameters are not supported")
            value = keyword.value
            if isinstance(value, ast.Name) and value.id in {"true", "false"}:
                result[keyword.arg] = value.id == "true"
            else:
                result[keyword.arg] = ast.literal_eval(value)
        return name, result
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid environment specification: {exc}") from exc


def create_game(name: str, parameters: dict[str, Any] | None = None) -> NativeGame:
    from game_arena.environments.avalon import AvalonGame
    from game_arena.environments.coup import CoupGame
    from game_arena.environments.werewolf import WerewolfGame

    factories = {
        "one_night_werewolf": WerewolfGame,
        "avalon": AvalonGame,
        "coup": CoupGame,
    }
    name = ALIASES.get(name, name)
    if name not in factories:
        raise ValueError(f"Unknown native environment {name!r}")
    return factories[name](parameters)


def load_game(spec: str) -> NativeGame:
    name, parameters = parse_spec(spec)
    return create_game(name, parameters)


def canonical_spec(game: NativeGame) -> str:
    parameters = ",".join(
        f"{key}={json.dumps(value, separators=(',', ':'))}"
        for key, value in sorted(game.parameters.items())
    )
    return f"{game.name}({parameters})"


def describe(game: NativeGame) -> dict[str, Any]:
    low, high = {"avalon": (5, 10), "coup": (2, 6), "one_night_werewolf": (3, 10)}[
        game.name
    ]
    return {
        "short_name": game.name,
        "long_name": game.name.replace("_", " ").title(),
        "backend": "native",
        "dynamics": "SEQUENTIAL",
        "chance_mode": "EXPLICIT_STOCHASTIC",
        "information": "IMPERFECT_INFORMATION",
        "utility": "GENERAL_SUM",
        "reward_model": "TERMINAL",
        "min_num_players": low,
        "max_num_players": high,
        "num_players": game.num_players(),
        "num_populations": 1,
        "default_loadable": True,
        "is_concrete": True,
        "parameter_specification": game.parameter_specification,
        "parameters": game.parameters,
        "observations": {
            "information_state_string": False,
            "information_state_tensor": False,
            "observation_string": True,
            "observation_tensor": False,
        },
        "supported": True,
        "unsupported_reasons": [],
        "requirements": [],
        "verified": False,
        "reproducibility": "recorded_explicit_chance",
        "game": canonical_spec(game),
        "canonical_game": canonical_spec(game),
        "implementation_game": game.name,
        "source_files": [],
        "rules": game.rules,
    }
