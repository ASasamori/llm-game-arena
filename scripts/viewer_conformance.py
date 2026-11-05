"""Audit the installed OpenSpiel viewer contract with model-free state walks.

Each configuration runs in an isolated interpreter with a deadline. Registry
declarations, unavailable public information, unsupported games and constructor
failures remain distinct. Passing a bounded prefix never certifies every rule,
reachable state, parameterization, or strategic agent behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import UTC, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_arena.openspiel import catalog  # noqa: E402 - standalone script bootstrap
from game_arena.openspiel.provenance import (  # noqa: E402
    derive_seed,
    runtime_manifest,
)


PARAMETER_FIXTURES = {
    "add_noise": "add_noise(game=tic_tac_toe(),epsilon=0.1,seed=1)",
    "cached_tree": "cached_tree(game=tic_tac_toe())",
    "coop_to_1p": "coop_to_1p(game=tiny_hanabi())",
    "misere": "misere(game=tic_tac_toe())",
    "normal_form_extensive_game": "normal_form_extensive_game(game=matrix_rps())",
    "repeated_game": "repeated_game(stage_game=matrix_rps(),num_repetitions=2)",
    "repeated_poker": (
        "repeated_poker(universal_poker_game_string=universal_poker(),"
        "max_num_hands=2,reset_stacks=false,rotate_dealer=true)"
    ),
    "restricted_nash_response": (
        "restricted_nash_response(game=kuhn_poker(),fixed_player=0,p=0.5)"
    ),
    "start_at": "start_at(game=tic_tac_toe(),history=0;1)",
    "turn_based_simultaneous_game": "turn_based_simultaneous_game(game=matrix_rps())",
    "zerosum": "zerosum(game=matrix_pd())",
}
_FILE_FIXTURES = {"efg_game", "nfg_game"}


def _limits(max_steps: int, timeout: float, seed: int, workers: int = 1) -> None:
    if type(max_steps) is not int or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be an integer in [1, 10000]")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer in [1, 32]")


def _fixture_spec(name: str, directory: Path) -> tuple[str, dict[str, Any]]:
    if name in PARAMETER_FIXTURES:
        return PARAMETER_FIXTURES[name], {
            "source": "declared_parameter_fixture",
            "specification": PARAMETER_FIXTURES[name],
        }
    import pyspiel

    if name == "efg_game":
        contents = pyspiel.get_kuhn_poker_efg_data()
        definition_source = "pyspiel.get_kuhn_poker_efg_data()"
        suffix = "efg"
    elif name == "nfg_game":
        contents = pyspiel.game_to_nfg_string(pyspiel.load_game("matrix_rps"))
        definition_source = "pyspiel.game_to_nfg_string(matrix_rps)"
        suffix = "nfg"
    else:
        raise ValueError(f"No parameter fixture declared for {name}")
    path = directory / f"definition.{suffix}"
    path.write_text(contents)
    return f"{name}(filename={path})", {
        "source": definition_source,
        "definition_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "ephemeral_file": True,
        "reproduction": "The checker recreates this definition from the installed OpenSpiel build.",
    }


def _node_kind(state: Any) -> str:
    if state.is_terminal():
        return "terminal"
    if state.is_chance_node():
        return "chance"
    if state.is_mean_field_node():
        return "mean_field"
    if state.is_simultaneous_node():
        return "simultaneous"
    return "decision"


def _check_configuration(spec: str, max_steps: int, seed: int) -> dict[str, Any]:
    """Walk real state APIs without constructing a model or importing its SDKs."""
    import pyspiel

    from game_arena.openspiel.presentation import describe_state
    from game_arena.openspiel.views import player_view

    game = catalog.load_game(spec, seed=seed)
    game_type = game.get_type()
    if getattr(game, "environment_backend", "openspiel") != "openspiel":
        return {
            "game": spec,
            "status": "outside_scope",
            "reason": "Native game boards have a separate public_state contract.",
        }
    states = (
        game.new_initial_states()
        if game_type.dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD
        else [game.new_initial_state()]
    )
    if not states:
        raise ValueError("Game supplies no initial state")
    # Mean-field games may have distinct populations; cover each one, each with
    # the same declared prefix bound and independent policy/chance streams.
    formats: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    views: Counter[str] = Counter()
    population_results = []
    decisions = 0
    transitions = 0
    snapshot_digest = hashlib.sha256()
    for population, state in enumerate(states):
        rng = random.Random(derive_seed(seed, "viewer_policy", population))
        chance_rng = random.Random(derive_seed(seed, "viewer_chance", population))
        count = 0
        while True:
            snapshot = describe_state(state)
            if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
                raise ValueError("OpenSpiel state has no supported presentation")
            encoded = json.dumps(snapshot, allow_nan=False, sort_keys=True)
            if json.loads(encoded) != snapshot:
                raise ValueError("Presentation does not round-trip through JSON")
            board = snapshot["board"]
            board_format = board.get("format")
            if board_format not in {"grid", "text", "unavailable"}:
                raise ValueError("Unknown presentation board format")
            hidden = (
                game_type.information
                == pyspiel.GameType.Information.IMPERFECT_INFORMATION
            )
            if hidden:
                if board_format != "unavailable" or board.get("text") is not None:
                    raise ValueError("Hidden game exposes an evaluator board publicly")
                if snapshot["visibility"]["public"] is not False:
                    raise ValueError("Hidden game is labelled public")
            elif board_format != "unavailable" and board.get("text") != str(state):
                raise ValueError(
                    "Presentation discarded or altered the evaluator snapshot"
                )
            if board_format == "grid" and not isinstance(board.get("rows"), list):
                raise ValueError("Grid presentation has no cell rows")
            formats[board_format] += 1
            kind = _node_kind(state)
            if snapshot["node"]["kind"] != kind:
                raise ValueError("Presentation node kind differs from engine state")
            kinds[kind] += 1
            snapshot_digest.update(encoded.encode())
            snapshot_digest.update(b"\n")
            if kind == "terminal" or count == max_steps:
                break
            if kind == "chance":
                outcomes = list(state.chance_outcomes())
                if not outcomes or not math.isclose(sum(p for _, p in outcomes), 1):
                    raise ValueError("Chance distribution is absent or invalid")
                action = chance_rng.choices(
                    [a for a, _ in outcomes], weights=[p for _, p in outcomes]
                )[0]
                state.apply_action(action)
            elif kind == "mean_field":
                support = list(state.distribution_support())
                state.update_distribution(
                    [1 / len(support)] * len(support) if support else []
                )
            else:
                actors = (
                    range(game.num_players())
                    if kind == "simultaneous"
                    else [state.current_player()]
                )
                actions = []
                for actor in actors:
                    legal = list(state.legal_actions(actor))
                    if not legal and kind == "simultaneous":
                        actions.append(pyspiel.INVALID_ACTION)
                        continue
                    if not legal:
                        raise ValueError(
                            "Nonterminal acting player has no legal actions"
                        )
                    source, observation = player_view(state, actor)
                    if not isinstance(observation, str):
                        raise ValueError("Player observation is not preserved as text")
                    views[source] += 1
                    action = rng.choice(legal)
                    label = state.action_to_string(actor, action)
                    if not isinstance(label, str):
                        raise ValueError("Selected action has no readable engine label")
                    actions.append(action)
                    decisions += 1
                if kind == "simultaneous":
                    state.apply_actions(actions)
                else:
                    state.apply_action(actions[0])
            count += 1
            transitions += 1
        population_results.append(
            {
                "population": population,
                "transitions": count,
                "terminal": state.is_terminal(),
                "verification_scope": "terminal_episode"
                if state.is_terminal()
                else "bounded_prefix",
            }
        )
    return {
        "game": spec,
        "registered_game": game_type.short_name,
        "canonical_game": catalog.canonical_game_spec(spec, seed=seed),
        "status": "passed",
        "information": str(game_type.information).rsplit(".", 1)[-1],
        "dynamics": str(game_type.dynamics).rsplit(".", 1)[-1],
        "num_players": game.num_players(),
        "num_populations": len(states),
        "transitions": transitions,
        "decisions": decisions,
        "formats": dict(formats),
        "kinds": dict(kinds),
        "observation_sources": dict(views),
        "snapshot_sha256": snapshot_digest.hexdigest(),
        "populations": population_results,
        "mean_field_assumption": "uniform exogenous baseline"
        if game_type.dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD
        else None,
    }


def audit_one(
    spec: str,
    *,
    max_steps: int = 96,
    timeout: float = 20,
    seed: int = 0,
    parameter_fixture: bool = False,
) -> dict[str, Any]:
    _limits(max_steps, timeout, seed)
    started = time.monotonic()
    try:
        resolved = catalog.resolve_game_spec(spec)
    except ValueError as exc:
        return {"game": spec, "status": "configuration_error", "reason": str(exc)}
    entry = next(
        item
        for item in catalog.list_games()
        if item["short_name"] == resolved.split("(", 1)[0]
    )
    if not entry["supported"]:
        return {
            "game": spec,
            "status": "unsupported",
            "reason": "; ".join(entry["unsupported_reasons"]),
        }
    if not entry["default_loadable"] and "(" not in resolved and not parameter_fixture:
        return {
            "game": spec,
            "status": "requires_parameters",
            "reason": "Supply explicit game parameters or use the registry audit's declared fixture.",
        }
    env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "TYPESAFE_API_KEY",
        "JEV_API_KEY",
    ):
        env.pop(key, None)
    env["MODEL_API_MOCK_MODE"] = "true"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH")])
    )
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_worker"],
            input=json.dumps(
                {
                    "game": resolved,
                    "max_steps": max_steps,
                    "seed": seed,
                    "parameter_fixture": parameter_fixture,
                }
            ),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        return {
            "game": spec,
            "status": "timeout",
            "reason": f"Exceeded {timeout}s isolated-process deadline",
            "elapsed_seconds": time.monotonic() - started,
        }
    payload: dict[str, Any]
    if result.returncode:
        payload = {
            "game": spec,
            "status": "failed",
            "reason": f"Process exited {result.returncode}: {result.stderr[-2000:]}",
        }
    else:
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict) or "status" not in payload:
                raise ValueError("Missing status object")
        except (ValueError, TypeError):
            payload = {
                "game": spec,
                "status": "failed",
                "reason": f"Invalid worker output: {result.stdout[-2000:]}",
            }
    payload["elapsed_seconds"] = time.monotonic() - started
    return payload


def run_audit(
    games: Sequence[str] | None = None,
    *,
    max_steps: int = 96,
    timeout: float = 20,
    seed: int = 0,
    workers: int = 4,
) -> dict[str, Any]:
    _limits(max_steps, timeout, seed, workers)
    tasks: list[tuple[str, bool, str]] = []
    results: list[dict[str, Any]] = []
    if games is None:
        entries = [
            entry
            for entry in catalog.list_games()
            if entry.get("backend", "openspiel") == "openspiel"
        ]
        for entry in entries:
            name = entry["short_name"]
            if not entry["supported"]:
                results.append(
                    {
                        "game": name,
                        "status": "unsupported",
                        "reason": "; ".join(entry["unsupported_reasons"]),
                    }
                )
            elif entry["default_loadable"]:
                tasks.append((name, False, "default_configuration"))
            elif name in PARAMETER_FIXTURES or name in _FILE_FIXTURES:
                tasks.append((name, True, "parameter_fixture"))
            else:
                results.append(
                    {
                        "game": name,
                        "status": "requires_parameters",
                        "reason": "No default or declared parameter fixture is available.",
                    }
                )
        registry_count = len(entries)
    else:
        tasks = [(spec, False, "explicit_configuration") for spec in games]
        registry_count = None

    def check(task: tuple[str, bool, str]) -> dict[str, Any]:
        spec, fixture, origin = task
        result = audit_one(
            spec,
            max_steps=max_steps,
            timeout=timeout,
            seed=seed,
            parameter_fixture=fixture,
        )
        result["configuration_origin"] = origin
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results.extend(executor.map(check, tasks))
    results.sort(key=lambda result: result["game"])
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "runtime": runtime_manifest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "seeded_random_legal_actions_without_models",
        "paid_api_calls": 0,
        "registered_openspiel_games": registry_count,
        "max_steps_per_population": max_steps,
        "seed": seed,
        "timeout_seconds_per_configuration": timeout,
        "counts": dict(Counter(result["status"] for result in results)),
        "scope": (
            "Only visited states and legal actions are checked. Parameter fixtures "
            "are examples, not exhaustive configuration coverage. Hidden games pass "
            "with an explicitly unavailable public board and separate player views."
        ),
        "games": results,
        "registration_warnings": catalog.registration_warnings(),
    }


def _write(path: Path, report: dict[str, Any], jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        if jsonl:
            metadata = {key: value for key, value in report.items() if key != "games"}
            stream.write(
                json.dumps({"type": "summary", **metadata}, sort_keys=True) + "\n"
            )
            for record in report["games"]:
                stream.write(
                    json.dumps({"type": "configuration", **record}, sort_keys=True)
                    + "\n"
                )
        else:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        request = json.loads(sys.stdin.read())
        captured = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = None
                spec = request["game"]
                if request.get("parameter_fixture"):
                    spec, fixture = _fixture_spec(spec, Path(temporary))
                payload = _check_configuration(
                    spec, request["max_steps"], request["seed"]
                )
                if fixture is not None:
                    payload["game"] = request["game"]
                    payload["fixture"] = fixture
        except Exception as exc:
            payload = {
                "game": request["game"],
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(payload, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jsonl-output", type=Path)
    args = parser.parse_args(arguments)
    report = run_audit(
        args.games,
        max_steps=args.max_steps,
        timeout=args.timeout,
        seed=args.seed,
        workers=args.workers,
    )
    if args.output:
        _write(args.output, report)
    if args.jsonl_output:
        _write(args.jsonl_output, report, jsonl=True)
    if not args.output and not args.jsonl_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "counts": report["counts"],
                    "output": str(args.output) if args.output else None,
                    "jsonl_output": str(args.jsonl_output)
                    if args.jsonl_output
                    else None,
                },
                sort_keys=True,
            )
        )
    return int(
        any(
            report["counts"].get(status)
            for status in ("failed", "timeout", "configuration_error")
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
