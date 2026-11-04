"""Exercise and replay native rules without importing AI or OpenSpiel code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from game_arena.environments.registry import NAMES, canonical_spec, describe, load_game


def state_hash(state: Any) -> str:
    return hashlib.sha256(state.serialize().encode()).hexdigest()


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name == "adapter.py":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def simulate(
    game_spec: str, *, seed: int = 0, policy: str = "random", max_steps: int = 10000
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_steps) is not int or not 1 <= max_steps <= 100000:
        raise ValueError("max_steps must be in [1, 100000]")
    if policy not in {"random", "first"}:
        raise ValueError("policy must be random or first")
    game = load_game(game_spec)
    state = game.new_initial_state()
    chance_rng, policy_rng = random.Random(seed), random.Random(seed ^ 0xA5A5A5A5)
    record: dict[str, Any] = {
        "schema_version": 1,
        "implementation_sha256": implementation_hash(),
        "backend": "native",
        "game": canonical_spec(game),
        "seed": seed,
        "policy": policy,
        "max_steps": max_steps,
        "initial_state_hash": state_hash(state),
        "steps": [],
    }
    for index in range(max_steps):
        if state.is_terminal():
            break
        actor = state.current_player()
        before = state_hash(state)
        observation = None
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = chance_rng.choices(
                [a for a, _ in outcomes], weights=[p for _, p in outcomes]
            )[0]
        else:
            observation = state.observation_string(actor)
            legal = state.legal_actions(actor)
            if not legal:
                raise ValueError("Nonterminal player has no legal actions")
            action = policy_rng.choice(legal) if policy == "random" else legal[0]
        label = state.action_to_string(actor, action)
        state.apply_action(action)
        record["steps"].append(
            {
                "index": index,
                "player": actor,
                "action": action,
                "label": label,
                "observation": observation,
                "state_hash_before": before,
                "state_hash_after": state_hash(state),
            }
        )
    record.update(
        terminal=state.is_terminal(),
        status="COMPLETED" if state.is_terminal() else "TRUNCATED",
        returns=state.returns(),
        final_state_hash=state_hash(state),
    )
    return record


def replay(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != 1 or record.get("backend") != "native":
        raise ValueError("Unsupported environment trace")
    if record.get("implementation_sha256") != implementation_hash():
        raise ValueError(
            "Environment implementation changed; restore the recorded build"
        )
    state = load_game(record["game"]).new_initial_state()
    seed, policy, max_steps = (
        record.get("seed"),
        record.get("policy"),
        record.get("max_steps"),
    )
    if type(seed) is not int or seed < 0 or policy not in {"random", "first"}:
        raise ValueError("Invalid recorded baseline policy or seed")
    if type(max_steps) is not int or not 1 <= max_steps <= 100000:
        raise ValueError("Invalid recorded transition limit")
    chance_rng, policy_rng = random.Random(seed), random.Random(seed ^ 0xA5A5A5A5)
    if state_hash(state) != record["initial_state_hash"]:
        raise ValueError("Initial state mismatch")
    for index, event in enumerate(record["steps"]):
        if (
            state.is_terminal()
            or index != event["index"]
            or state.current_player() != event["player"]
        ):
            raise ValueError(f"Invalid transition {index}")
        if state_hash(state) != event["state_hash_before"]:
            raise ValueError(f"Input hash mismatch at {index}")
        if type(event["action"]) is not int or event[
            "action"
        ] not in state.legal_actions(event["player"]):
            raise ValueError(f"Illegal recorded action at {index}")
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            expected = chance_rng.choices(
                [a for a, _ in outcomes], weights=[p for _, p in outcomes]
            )[0]
        else:
            legal = state.legal_actions(event["player"])
            expected = policy_rng.choice(legal) if policy == "random" else legal[0]
        if event["action"] != expected:
            raise ValueError(f"Seeded policy mismatch at {index}")
        observation = (
            None
            if state.is_chance_node()
            else state.observation_string(event["player"])
        )
        if (
            observation != event["observation"]
            or state.action_to_string(event["player"], event["action"])
            != event["label"]
        ):
            raise ValueError(f"Player evidence mismatch at {index}")
        state.apply_action(event["action"])
        if state_hash(state) != event["state_hash_after"]:
            raise ValueError(f"Output hash mismatch at {index}")
    expected_status = "COMPLETED" if state.is_terminal() else "TRUNCATED"
    if len(record["steps"]) > max_steps or (
        not state.is_terminal() and len(record["steps"]) != max_steps
    ):
        raise ValueError("Trace ends outside its recorded transition limit")
    if (
        state_hash(state) != record["final_state_hash"]
        or state.returns() != record["returns"]
        or state.is_terminal() != record["terminal"]
        or expected_status != record["status"]
    ):
        raise ValueError("Final result mismatch")
    return {
        "verified": True,
        "game": record["game"],
        "transitions": len(record["steps"]),
        "terminal": state.is_terminal(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    inspection = commands.add_parser("inspect")
    inspection.add_argument("--game", required=True)
    simulation = commands.add_parser("simulate")
    simulation.add_argument("--game", required=True)
    simulation.add_argument("--seed", type=int, default=0)
    simulation.add_argument("--policy", choices=["random", "first"], default="random")
    simulation.add_argument("--max-steps", type=int, default=10000)
    simulation.add_argument("--output", type=Path, required=True)
    reproduction = commands.add_parser("replay")
    reproduction.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = [describe(load_game(name)) for name in NAMES]
        elif args.command == "inspect":
            result = describe(load_game(args.game))
        elif args.command == "replay":
            result = replay(json.loads(args.trace.read_text()))
        else:
            record = simulate(
                args.game, seed=args.seed, policy=args.policy, max_steps=args.max_steps
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "w") as stream:
                json.dump(record, stream, indent=2)
            args.output.chmod(0o600)
            result = {
                **replay(record),
                "status": record["status"],
                "returns": record["returns"],
                "output": str(args.output),
            }
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
