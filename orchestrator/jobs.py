"""Validate and schedule independent, reproducible episodes on the runner queue."""

from __future__ import annotations

import hashlib
import math
import logging
from datetime import UTC, datetime
import os
import re
import uuid
from itertools import combinations
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry

from game_arena.openspiel.catalog import inspect_game, resolve_game_spec
from game_arena.openspiel.provenance import runtime_manifest
from orchestrator.models import (
    Episode,
    Matchup,
    MatchupProgress,
    Run,
    RunConfiguration,
    RunDispatch,
    RunProgress,
    get_session,
)
from orchestrator.tasks import run_episode_pvp, run_episode_single, validate_model_name

LOGGER = logging.getLogger("orchestrator.jobs")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT)
queue = Queue("episodes", connection=redis_conn)
SUPPORTED_TYPES = ["INPUT", "RAND", "PVP", "PVP_RR"]
JOB_TIMEOUT_SECONDS = max(30, int(os.environ.get("ORCH_JOB_TIMEOUT_SECONDS", "1800")))
ENQUEUE_BATCH_SIZE = 100
MAX_JOBS_PER_RUN = int(os.environ.get("ORCH_MAX_JOBS_PER_RUN", "1000000"))


def episode_seed(seed: int, matchup_index: int, episode_num: int) -> int:
    """Stable across process restarts, worker counts, and scheduling order."""
    value = f"arena-v1:{seed}:{matchup_index}:{episode_num}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def episode_identity(run_id: str, matchup_index: int, episode_num: int) -> str:
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"arena:{run_id}:{matchup_index}:{episode_num}")
    )


def validate_api_game_spec(env: str) -> str:
    """Queue/API callers cannot select server-local files via game parameters.

    Offline CLI runs can use file-backed games and record their source digests.
    Constructor inspection itself may read those files, so this check must run
    before calling catalog.inspect_game, including for nested transformations.
    """
    spec = resolve_game_spec(env)
    for match in re.finditer(r"(?:^|[,(])\s*([A-Za-z_][A-Za-z_0-9]*)\s*=", spec):
        parameter = match.group(1).lower()
        if "file" in parameter or "path" in parameter:
            raise ValueError(
                "File/path game parameters are disabled for queue and API submission; use the offline CLI with trusted game resources"
            )
    return spec


def validate_run_spec(
    *,
    run_type: str,
    num_episodes: int,
    num_steps: int | None,
    model_names: list[str],
    input_actions: list[int] | None,
    env: str,
    seed: int = 0,
    mean_field_distribution: str | dict[str, float] | None = None,
    mean_field_population: int = 0,
    failure_policy: str = "random",
) -> dict[str, Any]:
    """Admission checks happen before touching the database or queue.

    Game loading is isolated by the catalog because invalid OpenSpiel parameters
    can abort the process in native code.
    """
    if failure_policy not in {"random", "raise"}:
        raise ValueError("failure_policy must be 'random' or 'raise'")
    if run_type not in SUPPORTED_TYPES:
        raise ValueError(f"run_type must be one of {SUPPORTED_TYPES}")
    if (
        isinstance(num_episodes, bool)
        or not isinstance(num_episodes, int)
        or num_episodes < 1
    ):
        raise ValueError("num_episodes must be a positive integer")
    if num_steps is not None and (isinstance(num_steps, bool) or num_steps < 1):
        raise ValueError("num_steps must be positive or null")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not model_names:
        raise ValueError("At least one model name is required")
    for model_name in model_names:
        validate_model_name(model_name)
    game = validate_api_game_spec(env)
    info = inspect_game(game, seed=seed)
    if not info["supported"]:
        raise ValueError("; ".join(info["unsupported_reasons"]))
    players = int(info["num_players"])
    if players < 1:
        raise ValueError("Game must have at least one player")
    if info["dynamics"] == "MEAN_FIELD" and mean_field_distribution is None:
        raise ValueError("Mean-field games require an explicit mean_field_distribution")
    if mean_field_distribution is not None:
        if mean_field_distribution != "uniform" and not isinstance(
            mean_field_distribution, dict
        ):
            raise ValueError(
                "mean_field_distribution must be 'uniform' or a probability mapping"
            )
        if isinstance(mean_field_distribution, dict) and (
            not mean_field_distribution
            or any(
                not math.isfinite(v) or v < 0 for v in mean_field_distribution.values()
            )
        ):
            raise ValueError("Mean-field probabilities must be finite and nonnegative")
    if (
        isinstance(mean_field_population, bool)
        or not isinstance(mean_field_population, int)
        or mean_field_population < 0
    ):
        raise ValueError("mean_field_population must be nonnegative")
    if mean_field_population >= int(info.get("num_populations", 1)):
        raise ValueError(
            f"mean_field_population must be below {info.get('num_populations', 1)} for {game}"
        )
    if run_type == "PVP" and len(model_names) != players:
        raise ValueError(f"PVP requires exactly {players} model_names for {game}")
    if run_type == "PVP_RR":
        if len(model_names) < players:
            raise ValueError(f"PVP_RR requires at least {players} model_names")
        if len(set(model_names)) != len(model_names):
            raise ValueError("PVP_RR model_names must be unique")
    if run_type in {"RAND", "INPUT"} and len(model_names) != 1:
        raise ValueError(f"{run_type} requires exactly one model name")
    if run_type == "INPUT":
        if players < 2:
            raise ValueError("INPUT requires a game with opponents")
        if num_episodes != 1 or not input_actions:
            raise ValueError("INPUT requires num_episodes=1 and input_actions")
        if any(
            isinstance(action, bool) or not isinstance(action, int)
            for action in input_actions
        ):
            raise ValueError("input_actions must contain integer action IDs")
    matchups = math.comb(len(model_names), players) if run_type == "PVP_RR" else 1
    total = matchups * num_episodes
    if total > MAX_JOBS_PER_RUN:
        raise ValueError(
            f"Run would create {total} jobs; maximum is {MAX_JOBS_PER_RUN}"
        )
    return {
        "game": game,
        "num_players": players,
        "total_episodes": total,
        "game_info": info,
    }


def _enqueue_batch(batch: list[Any]) -> None:
    """Batch Redis round trips; recover failed deliveries on pending dispatches."""
    with queue.connection.pipeline() as pipeline:
        for item in batch:
            pipeline.hget(Job.key_for(item.job_id), "status")
        statuses = pipeline.execute()
    statuses = [
        status.decode() if isinstance(status, bytes) else status for status in statuses
    ]
    pending = [
        item for item, status in zip(batch, statuses, strict=True) if status is None
    ]
    failed = [
        item for item, status in zip(batch, statuses, strict=True) if status == "failed"
    ]
    if failed:
        session = get_session()
        try:
            saved = {
                episode_id
                for (episode_id,) in session.query(Episode.episode_id).filter(
                    Episode.episode_id.in_([item.job_id for item in failed])
                )
            }
        finally:
            session.close()
        registry = FailedJobRegistry(queue=queue)
        for item in failed:
            if item.job_id not in saved:
                registry.requeue(item.job_id)
    if pending:
        queue.enqueue_many(pending)


def _set_dispatch(run_id: str, *, complete: bool, error: str | None = None) -> None:
    session = get_session()
    try:
        dispatch = session.get(RunDispatch, run_id)
        if dispatch is None:
            dispatch = RunDispatch(run_id=run_id)
            session.add(dispatch)
        dispatch.complete = complete
        dispatch.last_error = error[:2000] if error else None
        dispatch.updated_at = datetime.now(UTC)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _request_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    # Catalog descriptions can change independently from the submitted request.
    return {
        key: value
        for key, value in payload.items()
        if key not in {"game_info", "runtime"}
    }


def enqueue_run(
    run_type: str,
    num_episodes: int,
    num_steps: int | None,
    model_names: list[str],
    input_actions: list[int] | None = None,
    env: str = "connect_four_v1",
    *,
    seed: int = 0,
    rotate_seats: bool = True,
    mean_field_distribution: str | dict[str, float] | None = None,
    mean_field_population: int = 0,
    failure_policy: str = "random",
    run_id: str | None = None,
) -> tuple[str, list[str]]:
    admission = validate_run_spec(
        run_type=run_type,
        num_episodes=num_episodes,
        num_steps=num_steps,
        model_names=model_names,
        input_actions=input_actions,
        env=env,
        seed=seed,
        mean_field_distribution=mean_field_distribution,
        mean_field_population=mean_field_population,
        failure_policy=failure_policy,
    )
    run_id = run_id or str(uuid.uuid4())
    run_namespace = uuid.UUID(run_id)
    game = admission["game"]
    players = admission["num_players"]
    configuration = {
        "schema_version": 1,
        "game": game,
        "game_info": admission["game_info"],
        "runtime": dict(runtime_manifest()),
        "run_type": run_type,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "model_names": model_names,
        "input_actions": input_actions,
        "seed": seed,
        "rotate_seats": rotate_seats,
        "mean_field_distribution": mean_field_distribution,
        "mean_field_population": mean_field_population,
        "failure_policy": failure_policy,
    }
    pairings = (
        list(combinations(model_names, players))
        if run_type == "PVP_RR"
        else [tuple(model_names)]
    )
    matchup_ids = [
        str(uuid.uuid5(run_namespace, f"matchup:{index}"))
        for index in range(len(pairings))
    ]
    session = get_session()
    try:
        existing = session.get(Run, run_id)
        if existing is not None:
            stored = session.get(RunConfiguration, run_id)
            stored_request = _request_configuration(stored.payload) if stored else None
            if stored_request is not None:
                stored_request.setdefault("failure_policy", "random")
            if stored_request != _request_configuration(configuration):
                raise ValueError("run_id already exists with different configuration")
            expected_runtime = stored.payload.get("runtime") if stored else None
            if session.get(RunDispatch, run_id) is None:
                session.add(RunDispatch(run_id=run_id, complete=False))
                session.commit()
        else:
            expected_runtime = configuration["runtime"]
            session.add(
                Run(
                    run_id=run_id,
                    run_type=run_type,
                    num_episodes=admission["total_episodes"],
                    status="PENDING",
                )
            )
            session.flush()
            session.add(RunConfiguration(run_id=run_id, payload=configuration))
            session.add(RunDispatch(run_id=run_id, complete=False))
            session.add(RunProgress(run_id=run_id, completed=0, failed=0))
            if run_type in {"PVP", "PVP_RR"}:
                for matchup_id, models in zip(matchup_ids, pairings, strict=True):
                    session.add(
                        Matchup(
                            matchup_id=matchup_id,
                            run_id=run_id,
                            models=list(models),
                            num_episodes=num_episodes,
                            status="PENDING",
                        )
                    )
                session.flush()
                for matchup_id in matchup_ids:
                    session.add(
                        MatchupProgress(matchup_id=matchup_id, completed=0, failed=0)
                    )
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    try:
        job_ids: list[str] = []
        batch: list[Any] = []
        for matchup_index, (matchup_id, models) in enumerate(
            zip(matchup_ids, pairings, strict=True)
        ):
            for episode_num in range(num_episodes):
                identity = episode_identity(run_id, matchup_index, episode_num)
                kwargs: dict[str, Any] = {
                    "run_id": run_id,
                    "episode_num": episode_num,
                    "num_steps": num_steps,
                    "game": game,
                    "seed": episode_seed(seed, matchup_index, episode_num),
                    "episode_id": identity,
                    "rotate_seats": rotate_seats,
                    "metadata": {
                        "run_seed": seed,
                        "matchup_index": matchup_index,
                        "episode_num": episode_num,
                        "expected_runtime": expected_runtime,
                    },
                    "mean_field_distribution": mean_field_distribution,
                    "mean_field_population": mean_field_population,
                    "failure_policy": failure_policy,
                }
                if run_type in {"PVP", "PVP_RR"}:
                    kwargs["matchup_id"] = matchup_id
                    func = run_episode_pvp
                else:
                    kwargs.update(
                        model_name=models[0],
                        run_type=run_type,
                        input_actions=input_actions,
                    )
                    func = run_episode_single
                batch.append(
                    queue.prepare_data(
                        func,
                        kwargs=kwargs,
                        job_id=identity,
                        timeout=JOB_TIMEOUT_SECONDS,
                        result_ttl=0,
                    )
                )
                job_ids.append(identity)
                if len(batch) >= ENQUEUE_BATCH_SIZE:
                    _enqueue_batch(batch)
                    batch.clear()
        if batch:
            _enqueue_batch(batch)
        _set_dispatch(run_id, complete=True)
    except Exception as exc:
        try:
            _set_dispatch(run_id, complete=False, error=str(exc))
        except Exception:
            LOGGER.exception(
                "Could not update pending dispatch marker", extra={"run_id": run_id}
            )
        raise
    return run_id, job_ids


def dispatch_pending_runs(limit: int = 10) -> dict[str, int]:
    """Recover committed submissions after Redis outages or process crashes.

    Stable IDs and episode delivery deduplication make repeated recovery safe.
    The runner calls this at startup and on its existing recovery interval.
    """
    if not 1 <= limit <= 100:
        raise ValueError("dispatch recovery limit must be between 1 and 100")
    session = get_session()
    try:
        pending = [
            (run_id, dict(payload))
            for run_id, payload in session.query(
                RunDispatch.run_id, RunConfiguration.payload
            )
            .join(RunConfiguration, RunDispatch.run_id == RunConfiguration.run_id)
            .filter(RunDispatch.complete.is_(False))
            .order_by(RunDispatch.updated_at, RunDispatch.run_id)
            .limit(limit)
        ]
    finally:
        session.close()
    completed = 0
    for run_id, payload in pending:
        try:
            enqueue_run(
                payload["run_type"],
                payload["num_episodes"],
                payload["num_steps"],
                payload["model_names"],
                input_actions=payload.get("input_actions"),
                env=payload["game"],
                seed=payload.get("seed", 0),
                rotate_seats=payload.get("rotate_seats", True),
                mean_field_distribution=payload.get("mean_field_distribution"),
                mean_field_population=payload.get("mean_field_population", 0),
                failure_policy=payload.get("failure_policy", "random"),
                run_id=run_id,
            )
            completed += 1
        except Exception:
            LOGGER.exception(
                "Pending run dispatch failed; next recovery will retry",
                extra={"run_id": run_id},
            )
    return {
        "attempted": len(pending),
        "completed": completed,
        "failed": len(pending) - completed,
    }


# Compatibility entry points used by scripts and older clients.
def enqueue_pvp_run(
    num_episodes: int, num_steps: int | None, model_names: list[str], game: str
):
    return enqueue_run("PVP", num_episodes, num_steps, model_names, env=game)


def enqueue_pvp_round_robin_run(
    num_episodes: int, num_steps: int | None, model_names: list[str], game: str
):
    return enqueue_run("PVP_RR", num_episodes, num_steps, model_names, env=game)


def enqueue_rand_run(
    num_episodes: int, num_steps: int | None, model_name: str, game: str
):
    return enqueue_run("RAND", num_episodes, num_steps, [model_name], env=game)


def enqueue_input_run(
    num_steps: int | None, model_name: str, input_actions: list[int], game: str
):
    return enqueue_run("INPUT", 1, num_steps, [model_name], input_actions, env=game)
