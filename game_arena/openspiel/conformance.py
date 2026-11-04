"""Bounded, process-isolated conformance checks using local mock policies only.

This is a build check, not benchmark data generation. A passing truncated
prefix establishes only the transitions actually visited, never complete game
coverage. Registry entries requiring parameters remain visible as skipped.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from game_arena.openspiel import catalog


def _check_episode(spec: str, max_steps: int, seed: int) -> dict[str, Any]:
    # No provider factories or model identifiers are accepted by this checker.
    from game_arena.openspiel.harness import harness
    from game_arena.openspiel.model_generation import LLMModel, MockModel
    from game_arena.openspiel.utils import HarnessOptions
    from game_arena.openspiel.replay import replay_episode

    info = catalog.inspect_game(spec, seed=seed)
    if not info["supported"]:
        return {
            "game": spec,
            "status": "skipped",
            "reason": "; ".join(info["unsupported_reasons"]),
        }
    models: list[LLMModel] = [
        MockModel(name=f"mock-player-{player}") for player in range(info["num_players"])
    ]
    options = HarnessOptions(
        type="PVP",
        models=models,
        num_steps=max_steps,
        input=None,
        game=spec,
        seed=seed,
        failure_policy="raise",
        mean_field_distribution="uniform",
    )
    first: dict[str, Any] = dict(harness(options))
    second: dict[str, Any] = dict(harness(options))
    expected_players = info["num_players"]
    for result in (first, second):
        if not replay_episode(result)["verified"]:
            raise AssertionError("Recorded episode failed deterministic action replay")
        if result.get("fallback_count", 0):
            raise AssertionError(
                "Mock conformance episode unexpectedly used a fallback"
            )
        if len(result["returns"]) != expected_players:
            raise AssertionError("Return vector does not match the player count")
        previous = result["manifest"]["initial_state_hash"]
        for index, event in enumerate(result["steps"]):
            if event["index"] != index or event["state_hash_before"] != previous:
                raise AssertionError(
                    "Transition log has a discontinuous state fingerprint"
                )
            for decision in event["decisions"]:
                if (
                    decision["legal_actions"]
                    and decision["action"] not in decision["legal_actions"]
                ):
                    raise AssertionError(
                        "Applied action is outside its recorded legal action set"
                    )
            previous = event["state_hash"]
        if previous != result["final_state_hash"]:
            raise AssertionError(
                "Final state fingerprint does not match the last transition"
            )
        if bool(result["terminal"]) != (result["status"] == "SUCCESS"):
            raise AssertionError("Episode termination status is inconsistent")

    def deterministic_record(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "game_spec": result["game_spec"],
            "initial_state_hash": result["manifest"]["initial_state_hash"],
            "final_state_hash": result["final_state_hash"],
            "returns": result["returns"],
            "transitions": [
                {key: value for key, value in event.items() if key != "decisions"}
                | {
                    "decisions": [
                        {
                            key: value
                            for key, value in decision.items()
                            if key != "latency_ms"
                        }
                        for decision in event["decisions"]
                    ]
                }
                for event in result["steps"]
            ],
        }

    if deterministic_record(dict(first)) != deterministic_record(dict(second)):
        raise AssertionError(
            "Repeated mock episodes produced different states or decisions for the same seed"
        )
    return {
        "game": spec,
        "canonical_game": first["game_spec"],
        "status": "passed",
        "episode_status": first["status"],
        "num_steps": first["num_steps"],
        "decision_count": first["decision_count"],
        "num_players": expected_players,
        "kinds": sorted({event["kind"] for event in first["steps"]}),
        "terminal": first["terminal"],
        "returns": first["returns"],
        "repeat_verified": True,
        "replay_verified": True,
        "verification_scope": "terminal_episode"
        if first["terminal"]
        else "bounded_prefix",
        "mean_field_distribution": "uniform"
        if info["dynamics"] == "MEAN_FIELD"
        else None,
    }


def check_game(
    spec: str, *, max_steps: int = 16, timeout: float = 10, seed: int = 0
) -> dict[str, Any]:
    """Run one configuration in a disposable interpreter with no provider keys."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    spec = catalog.resolve_game_spec(spec)
    project_root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ):
        env.pop(key, None)
    env["MODEL_API_MOCK_MODE"] = "true"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [project_root, env.get("PYTHONPATH")])
    )
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "game_arena.openspiel.conformance", "_worker"],
            input=json.dumps({"game": spec, "max_steps": max_steps, "seed": seed}),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=os.getcwd(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "game": spec,
            "status": "timeout",
            "reason": f"Exceeded {timeout}s",
            "elapsed_seconds": time.monotonic() - started,
        }
    if result.returncode != 0:
        return {
            "game": spec,
            "status": "failed",
            "reason": f"Native process exited with code {result.returncode}: {result.stderr[-2000:]}",
            "elapsed_seconds": time.monotonic() - started,
        }
    payload: dict[str, Any]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "game": spec,
            "status": "failed",
            "reason": f"Invalid checker output: {result.stdout[-2000:]}",
        }
    payload["elapsed_seconds"] = time.monotonic() - started
    return payload


def run_conformance(
    games: Sequence[str] | None = None,
    *,
    max_steps: int = 16,
    timeout: float = 10,
    seed: int = 0,
    workers: int = 4,
) -> dict[str, Any]:
    """Check all default-loadable registered games, or explicit configurations."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    skipped: list[dict[str, Any]] = []
    if games is None:
        selected = []
        for game in catalog.list_games():
            name = game["short_name"]
            if not game["supported"]:
                skipped.append(
                    {
                        "game": name,
                        "status": "skipped",
                        "reason": "; ".join(game["unsupported_reasons"]),
                    }
                )
            elif not game["default_loadable"]:
                skipped.append(
                    {
                        "game": name,
                        "status": "skipped",
                        "reason": "Requires explicit game parameters",
                    }
                )
            else:
                selected.append(name)
    else:
        selected = [catalog.resolve_game_spec(game) for game in games]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda spec: check_game(
                    spec, max_steps=max_steps, timeout=timeout, seed=seed
                ),
                selected,
            )
        )
    results = sorted(results + skipped, key=lambda result: result["game"])
    counts = Counter(result["status"] for result in results)
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "policy": "local_mock_only",
        "max_steps": max_steps,
        "seed": seed,
        "timeout_seconds": timeout,
        "counts": dict(counts),
        "games": results,
        "scope": "Only recorded prefixes are verified. A prefix pass does not certify every game state or parameterization.",
        "registration_warnings": catalog.registration_warnings(),
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        request = json.loads(sys.stdin.read())
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                payload = _check_episode(
                    request["game"], request["max_steps"], request["seed"]
                )
        except Exception as exc:
            payload = {
                "game": request["game"],
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(payload, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        nargs="+",
        help="Explicit game specifications; omitted checks the installed registry",
    )
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_conformance(
        args.games,
        max_steps=args.max_steps,
        timeout=args.timeout,
        seed=args.seed,
        workers=args.workers,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "output": str(args.output) if args.output else None,
            },
            sort_keys=True,
        )
    )
    if not args.output:
        print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(report["counts"].get("failed") or report["counts"].get("timeout")))


if __name__ == "__main__":
    raise SystemExit(main())
