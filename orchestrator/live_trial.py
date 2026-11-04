"""One explicitly authorized, budgeted Avalon trial through the existing runner.

This is intentionally separate from ordinary API submissions: no API request can
silently enable paid generation. Configure local database/Redis explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import uuid

DEFAULT_SEATS = [
    "claude-haiku-5-5",
    "claude-haiku-5-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
]


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def submit_trial(directory: Path, *, seed: int = 123) -> dict[str, Any]:
    import redis
    from rq import Queue
    from game_arena.openspiel.catalog import canonical_game_spec, inspect_game
    from game_arena.openspiel.provenance import runtime_manifest
    from orchestrator.models import (
        Run,
        RunConfiguration,
        RunDispatch,
        RunProgress,
        get_session,
    )

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (directory / "trial.json").exists():
        raise ValueError(
            "This trial directory was already submitted; do not repeat paid runs"
        )
    run_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid5(uuid.UUID(run_id), "budgeted-avalon"))
    game = canonical_game_spec("avalon(players=6,discussion_rounds=1)")
    configuration = {
        "schema_version": 1,
        "run_type": "PVP",
        "num_episodes": 1,
        "num_steps": 1000,
        "game": game,
        "game_info": inspect_game(game),
        "model_names": ["CLAUDE:" + name for name in DEFAULT_SEATS],
        "seed": seed,
        "rotate_seats": False,
        "failure_policy": "raise",
        "runtime": runtime_manifest(),
        "stop_after_quests": 3,
        "budget_usd": "0.90",
        "max_output_tokens": 192,
        "max_calls": 100,
        "trial_kind": "explicitly_authorized_paid_avalon",
        "episode_id": episode_id,
    }
    with get_session() as session:
        session.add(
            Run(run_id=run_id, run_type="PVP", num_episodes=1, status="PENDING")
        )
        session.flush()
        session.add(RunConfiguration(run_id=run_id, payload=configuration))
        session.add(RunDispatch(run_id=run_id, complete=True))
        session.add(RunProgress(run_id=run_id, completed=0, failed=0))
        session.commit()
    trial = {"run_id": run_id, "episode_id": episode_id, "configuration": configuration}
    _save(directory / "trial.json", trial)
    connection = redis.Redis(
        host=os.environ.get("REDIS_HOST", "127.0.0.1"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
    )
    queue = Queue(
        os.environ.get("RUNNER_QUEUE_NAME", "avalon_trial"), connection=connection
    )
    job = queue.enqueue(
        run_trial,
        run_id,
        str(directory),
        job_id="capped-trial-" + run_id,
        job_timeout=1200,
        result_ttl=86400,
    )
    trial["job_id"] = job.id
    _save(directory / "trial.json", trial)
    return trial


def run_trial(run_id: str, directory: str) -> dict[str, Any]:
    from game_arena.openspiel.budgeted_anthropic import (
        BudgetLedger,
        BudgetedClaudeModel,
    )
    from game_arena.openspiel.harness import harness
    from game_arena.openspiel.model_generation import LLMModel
    from game_arena.openspiel.provenance import runtime_manifest
    from game_arena.openspiel.replay import replay_episode
    from game_arena.openspiel.utils import HarnessOptions
    from orchestrator.models import EpisodeTrace, Run, RunConfiguration, get_session
    from orchestrator.tasks import _json_result, _persist_result

    root = Path(directory).resolve()
    with get_session() as session:
        row = session.get(RunConfiguration, run_id)
        if (
            row is None
            or row.payload.get("trial_kind") != "explicitly_authorized_paid_avalon"
        ):
            raise ValueError("Missing authorized trial configuration")
        configuration = row.payload
        existing = session.get(EpisodeTrace, configuration["episode_id"])
        if existing is not None:
            return {"reused": True, "episode_id": existing.episode_id}
    runtime = runtime_manifest()
    for key in (
        "source_sha256",
        "dependency_lock_sha256",
        "open_spiel_version",
        "prompt_version",
        "parser_version",
    ):
        if configuration["runtime"].get(key) != runtime.get(key):
            raise ValueError("Trial build changed before execution")
    ledger = BudgetLedger(
        root / "budget.sqlite",
        budget_usd=configuration["budget_usd"],
        max_calls=configuration["max_calls"],
    )
    if not ledger.claim_episode_once(configuration["episode_id"]):
        return {"reused": True, "paid_execution_already_claimed": True}
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = "RUNNING"
            session.commit()
    models: list[LLMModel] = [
        BudgetedClaudeModel(
            name.removeprefix("CLAUDE:"),
            ledger,
            max_tokens=configuration["max_output_tokens"],
        )
        for name in configuration["model_names"]
    ]
    result = harness(
        HarnessOptions(
            type="PVP",
            num_steps=configuration["num_steps"],
            input=None,
            models=models,
            game=configuration["game"],
            seed=configuration["seed"],
            episode_id=configuration["episode_id"],
            failure_policy="raise",
            stop_after_quests=3,
            metadata={"run_id": run_id, "trial_kind": configuration["trial_kind"]},
        )
    )
    result.setdefault("manifest", {})["trial_budget"] = ledger.summary()
    payload = _json_result(result)
    _save(root / "episode.json", payload)
    verification = replay_episode(payload)
    _save(root / "verification.json", verification)
    _persist_result(result, run_id, None)
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is not None and result["status"] == "TRUNCATED":
            run.status = "STOPPED"
            session.commit()
    summary = {
        "run_id": run_id,
        "episode_id": configuration["episode_id"],
        "status": result["status"],
        "stop_reason": payload.get("stop_reason"),
        "stop_message": payload.get("stop_message"),
        "transitions": result["num_steps"],
        "budget": ledger.summary(),
        "replay_verified": verification["verified"],
    }
    _save(root / "result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("Pass --allow-paid only for an explicitly authorized paid trial")
    if not os.environ.get("DATABASE_URL") or not os.environ.get("REDIS_PORT"):
        parser.error("Configure DATABASE_URL and local REDIS_PORT explicitly")
    trial = submit_trial(args.directory, seed=args.seed)
    print(
        json.dumps(
            {
                "run_id": trial["run_id"],
                "episode_id": trial["episode_id"],
                "job_id": trial["job_id"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
