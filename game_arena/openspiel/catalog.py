"""Discover OpenSpiel games and validate configurations without playing episodes.

A registry entry is a capability declaration, not a successful conformance test.
Untrusted game configurations are inspected in a disposable subprocess: some
native OpenSpiel constructors can terminate a process on invalid parameters.
"""

from __future__ import annotations

import contextlib
import copy
from functools import lru_cache
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any

import pyspiel

ENV_ALIASES = {"connect_four_v1": "connect_four", "chess_v1": "chess"}
_REGISTRATION_LOCK = threading.Lock()
_PYTHON_GAMES_REGISTERED = False
_REGISTRATION_WARNINGS: list[str] = []
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_SPEC_LENGTH = 100_000


class UnsupportedGameError(ValueError):
    """A game cannot be evaluated safely with the generic arena contract."""


def register_python_games() -> None:
    """Import OpenSpiel's maintained Python registry once, without stdout noise.

    Optional upstream dependencies (for example pokerkit) are not installed on
    the user's behalf. Their absence is exposed in catalog metadata.
    """
    global _PYTHON_GAMES_REGISTERED
    with _REGISTRATION_LOCK:
        if _PYTHON_GAMES_REGISTERED:
            return
        captured = io.StringIO()
        for module in (
            "open_spiel.python.games",
            "open_spiel.python.mfg.games",
            "open_spiel.python.mfg.games.normal_form_game",
        ):
            try:
                with contextlib.redirect_stdout(captured):
                    importlib.import_module(module)
            except ImportError as exc:
                _REGISTRATION_WARNINGS.append(f"{module}: {exc}")
        warning = captured.getvalue().strip()
        if warning:
            _REGISTRATION_WARNINGS.append(warning)
        _PYTHON_GAMES_REGISTERED = True


def _types() -> dict[str, Any]:
    register_python_games()
    return {entry.short_name: entry for entry in pyspiel.registered_games()}


def _enum_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def _metadata(game_type: Any) -> dict[str, Any]:
    observations = {
        "information_state_string": bool(game_type.provides_information_state_string),
        "information_state_tensor": bool(game_type.provides_information_state_tensor),
        "observation_string": bool(game_type.provides_observation_string),
        "observation_tensor": bool(game_type.provides_observation_tensor),
    }
    requirements: list[str] = []
    reasons: list[str] = []
    if not game_type.default_loadable:
        requirements.append("explicit_game_parameters")
    if game_type.dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
        requirements.append("mean_field_distribution")
    sampled = game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC
    parameters = game_type.parameter_specification
    seeded = "rng_seed" in parameters or "seed" in parameters
    if sampled and not seeded:
        requirements.append("reproducible_underlying_game")
    if (
        game_type.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION
        and not any(observations.values())
    ):
        reasons.append("No player-scoped observation API is implemented in this build")
    if game_type.short_name == "chat_game":
        reasons.append(
            "Requires a separately configured external language-model environment"
        )
    return {
        "short_name": game_type.short_name,
        "long_name": game_type.long_name,
        "dynamics": _enum_name(game_type.dynamics),
        "chance_mode": _enum_name(game_type.chance_mode),
        "information": _enum_name(game_type.information),
        "utility": _enum_name(game_type.utility),
        "reward_model": _enum_name(game_type.reward_model),
        "min_num_players": game_type.min_num_players,
        "max_num_players": game_type.max_num_players,
        "num_players": None,
        "num_populations": None,
        "default_loadable": bool(game_type.default_loadable),
        "is_concrete": bool(game_type.is_concrete),
        "parameter_specification": parameters,
        "observations": observations,
        "supported": not reasons,
        "unsupported_reasons": reasons,
        "requirements": requirements,
        "verified": False,
        "reproducibility": (
            "seeded_native_rng"
            if sampled and seeded
            else "underlying_game_required"
            if sampled
            else "recorded_explicit_chance"
        ),
    }


def list_games() -> list[dict[str, Any]]:
    """List every game registered by this installation without instantiating it.

    ``supported`` describes the generic observation/state contract. It does not
    mean default parameters are sufficient or that a full game was tested.
    ``requirements`` must be satisfied by the caller before execution.
    """
    return [_metadata(entry) for _, entry in sorted(_types().items())]


def registration_warnings() -> list[str]:
    register_python_games()
    return list(_REGISTRATION_WARNINGS)


def resolve_game_spec(env: str) -> str:
    """Resolve legacy environment aliases while preserving parameter values."""
    if not isinstance(env, str) or not env.strip():
        raise UnsupportedGameError(
            "A non-empty OpenSpiel game specification is required"
        )
    spec = env.strip()
    spec = ENV_ALIASES.get(spec.lower(), spec)
    if len(spec) > _MAX_SPEC_LENGTH:
        raise UnsupportedGameError(
            "Game specification exceeds the configured size limit"
        )
    name = spec.split("(", 1)[0].strip()
    if not _NAME_PATTERN.fullmatch(name) or name not in _types():
        raise UnsupportedGameError(f"Unknown OpenSpiel game '{name}'")
    if "(" in spec and not spec.endswith(")"):
        raise UnsupportedGameError("Game specification has unbalanced parentheses")
    if spec.count("(") != spec.count(")"):
        raise UnsupportedGameError("Game specification has unbalanced parentheses")
    return spec


def _seed_parameters(parameters: dict[str, Any], seed: int) -> None:
    """Seed native sampled RNGs, including nested game transformations.

    An explicit game parameter wins over the episode seed. This is important
    when loading the canonical game specification again for replay.
    """
    for value in parameters.values():
        if isinstance(value, dict) and "name" in value:
            _seed_parameters(value, seed)
    name = parameters.get("name")
    if name == "chat_game":
        raise UnsupportedGameError(
            "chat_game requires an external environment; offline loading is disabled"
        )
    game_type = _types().get(name) if isinstance(name, str) else None
    if game_type is None:
        raise UnsupportedGameError(f"Unknown nested OpenSpiel game '{name}'")
    if game_type.chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC:
        specification = game_type.parameter_specification
        if "rng_seed" in specification:
            if int(parameters.get("rng_seed", -1)) < 0:
                parameters["rng_seed"] = seed % 2_147_483_647
            else:
                parameters.setdefault("rng_seed", seed % 2_147_483_647)
        elif "seed" in specification:
            parameters.setdefault("seed", seed % 2_147_483_647)


def _is_resource_parameter(values: dict[str, Any], key: str, value: Any) -> bool:
    """Recognize only declared, explicit file/path string parameters."""
    if (
        not isinstance(value, str)
        or not value
        or not any(marker in key.lower() for marker in ("file", "path"))
    ):
        return False
    name = values.get("name")
    game_type = _types().get(name) if isinstance(name, str) else None
    return game_type is not None and key in game_type.parameter_specification


def canonical_game_spec(spec: str, seed: int = 0) -> str:
    """Keep the registered constructor name and its effective explicit inputs.

    Some OpenSpiel transformations return an implementation-specific short name
    from Game.to_string() (e.g. matrix_nfg), which cannot be loaded again. The
    original parsed constructor is therefore the source of replay configuration.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise UnsupportedGameError("seed must be a non-negative integer")
    resolved = resolve_game_spec(spec)
    try:
        parameters = pyspiel.game_parameters_from_string(resolved)
        _seed_parameters(parameters, seed)

        def absolute_sources(values: dict[str, Any]) -> None:
            for key, value in list(values.items()):
                if _is_resource_parameter(values, key, value):
                    values[key] = str(Path(value).expanduser().resolve())
                elif isinstance(value, dict) and "name" in value:
                    absolute_sources(value)

        absolute_sources(parameters)
        return pyspiel.game_parameters_to_string(parameters)
    except UnsupportedGameError:
        raise
    except Exception as exc:
        raise UnsupportedGameError(
            f"Invalid OpenSpiel configuration '{resolved}': {exc}"
        ) from exc


def game_source_files(
    spec: str, *, max_file_bytes: int = 64 * 1024 * 1024
) -> list[dict[str, Any]]:
    """Fingerprint declared external file/path inputs, including nested transforms.

    Definitions are not embedded or returned: introspection must not turn a
    caller-supplied game filename into arbitrary file-content retrieval. Replay
    requires the same definition files and validates these digests before play.
    """
    parameters = pyspiel.game_parameters_from_string(resolve_game_spec(spec))
    sources: list[dict[str, Any]] = []

    def visit(values: dict[str, Any], parameter_path: list[str]) -> None:
        name = values.get("name")
        for key, value in sorted(values.items()):
            if _is_resource_parameter(values, key, value):
                path = Path(value).expanduser().resolve()
                digest = hashlib.sha256()
                total = 0
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_file_bytes:
                            raise UnsupportedGameError(
                                f"Game resource exceeds {max_file_bytes} bytes"
                            )
                        digest.update(chunk)
                sources.append(
                    {
                        "path": str(path),
                        "parameter_path": parameter_path + [key],
                        "sha256": digest.hexdigest(),
                        "size_bytes": total,
                        "format": ("efg" if name == "efg_game" else "nfg")
                        if key == "filename" and name in {"efg_game", "nfg_game"}
                        else "resource",
                    }
                )
            elif isinstance(value, dict) and "name" in value:
                visit(value, parameter_path + [key])

    visit(parameters, [])
    return sources


def _load(spec: str, seed: int) -> Any:
    resolved = canonical_game_spec(spec, seed=seed)
    try:
        parameters = pyspiel.game_parameters_from_string(resolved)
        name = parameters.pop("name")
        game_type = _types()[name]
        if not game_type.default_loadable and not parameters:
            raise UnsupportedGameError(f"{name} requires explicit game parameters")
        return pyspiel.load_game(name, parameters)
    except UnsupportedGameError:
        raise
    except Exception as exc:
        raise UnsupportedGameError(
            f"Invalid OpenSpiel configuration '{resolved}': {exc}"
        ) from exc


def load_game(spec: str, seed: int = 0) -> Any:
    """Load a validated game with deterministic native RNG configuration.

    Use :func:`inspect_game` for untrusted API admission. Loading already
    validated configurations directly avoids spawning a process per episode.
    A fresh Game must be used for each episode, since native RNG state may be
    stored on the Game rather than the State.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise UnsupportedGameError("seed must be a non-negative integer")
    game = _load(spec, seed)
    info = _metadata(game.get_type())
    if not info["supported"]:
        raise UnsupportedGameError("; ".join(info["unsupported_reasons"]))
    if (
        game.get_type().chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC
        and info["reproducibility"] == "underlying_game_required"
    ):
        # Transformations generally expose their underlying concrete chance mode
        # after construction. Without a seed contract, exact replay is unsafe.
        raise UnsupportedGameError(
            "Sampled stochastic game exposes no reproducible RNG seed parameter"
        )
    return game


def _inspect_in_process(spec: str, seed: int) -> dict[str, Any]:
    resolved = resolve_game_spec(spec)
    if resolved.split("(", 1)[0] == "chat_game":
        info = _metadata(_types()["chat_game"])
        info.update(game=resolved, canonical_game=resolved)
        return info
    game = _load(resolved, seed)
    info = _metadata(game.get_type())
    if (
        game.get_type().chance_mode == pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC
        and info["reproducibility"] == "underlying_game_required"
    ):
        info["supported"] = False
        info["unsupported_reasons"].append(
            "Sampled stochastic game exposes no reproducible RNG seed parameter"
        )
    info.update(
        game=canonical_game_spec(resolved, seed),
        canonical_game=canonical_game_spec(resolved, seed),
        implementation_game=game.to_string(),
        source_files=game_source_files(resolved),
        num_players=game.num_players(),
        parameters=game.get_parameters(),
    )
    info["requirements"] = [
        requirement
        for requirement in info["requirements"]
        if requirement != "explicit_game_parameters"
    ]
    info["num_populations"] = 1
    if info["supported"]:
        # Constructor success alone does not establish that a usable state exists.
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
            populations = game.new_initial_states()
            info["num_populations"] = len(populations)
            if not populations:
                raise UnsupportedGameError("Game provides no initial population states")
        else:
            game.new_initial_state()
    return info


@lru_cache(maxsize=256)
def _inspect_cached(
    spec: str, seed: int, timeout: float, working_directory: str
) -> dict[str, Any]:
    project_root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_root, env.get("PYTHONPATH")])
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "game_arena.openspiel.catalog", "_inspect"],
            input=json.dumps({"game": spec, "seed": seed}),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=working_directory,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedGameError(
            f"Game configuration inspection timed out after {timeout}s"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or [
            "Native game inspection failed"
        ]
        raise UnsupportedGameError(
            f"Cannot load game configuration '{spec}': {detail[0]}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedGameError(
            "Game inspection produced an invalid response"
        ) from exc
    if "error" in payload:
        raise UnsupportedGameError(payload["error"])
    return payload


def inspect_game(spec: str, seed: int = 0, timeout: float = 10) -> dict[str, Any]:
    """Safely inspect a parameterized configuration without calling model APIs."""
    resolved = resolve_game_spec(spec)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise UnsupportedGameError("seed must be a non-negative integer")
    if timeout <= 0:
        raise ValueError("inspection timeout must be positive")
    working_directory = os.getcwd()
    info = copy.deepcopy(_inspect_cached(resolved, seed, timeout, working_directory))
    if info.get("source_files") and game_source_files(resolved) != info["source_files"]:
        # Never serve a cached inspection for a changed external game definition.
        _inspect_cached.cache_clear()
        info = copy.deepcopy(
            _inspect_cached(resolved, seed, timeout, working_directory)
        )
    return info


get_game_info = inspect_game


def _main() -> None:
    request = json.loads(sys.stdin.read())
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            info = _inspect_in_process(request["game"], request["seed"])
        print(json.dumps(info, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    _main()
