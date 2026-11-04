"""Runner jobs: execute one seeded episode and persist its trace atomically."""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from datetime import datetime
from typing import Any, cast

from rq import get_current_job
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from game_arena.openspiel import harness, model_generation, utils
from orchestrator.failures import classify_exception
from game_arena.openspiel.provenance import runtime_manifest
from orchestrator.models import (
    Episode,
    EpisodeStep,
    EpisodeTrace,
    EpisodeTransition,
    Matchup,
    MatchupProgress,
    Run,
    RunProgress,
    get_session,
)
from orchestrator.observability import configure_json_logging, increment_counter

configure_json_logging(service="worker_tasks")
LOGGER = logging.getLogger("orchestrator.tasks")
MODEL_FACTORIES = {
    "GEMINI": model_generation.GeminiModel,
    "GPT": model_generation.GPTModel,
    "CLAUDE": model_generation.ClaudeModel,
}


def get_worker_name() -> str | None:
    job = get_current_job()
    return job.worker_name if job else None


def validate_model_name(model_name: str) -> tuple[str, str | None]:
    """Accept provider aliases or PROVIDER:model-version without creating clients."""
    if not isinstance(model_name, str):
        raise ValueError("Model identifier must be a string")
    provider, separator, version = model_name.strip().partition(":")
    provider = provider.upper()
    if provider not in {*MODEL_FACTORIES, "MOCK"}:
        raise ValueError(
            f"Unsupported model provider '{provider}'. Use GEMINI, GPT, CLAUDE, or MOCK"
        )
    if separator and not version.strip():
        raise ValueError("Model version must be nonempty after ':'")
    return provider, version.strip() if separator else None


def _get_model(model_name: str) -> model_generation.LLMModel:
    provider, version = validate_model_name(model_name)
    # Fresh wrappers isolate every episode and avoid sharing conversational state
    # or mutable mock RNGs between runner worker threads.
    if provider == "MOCK":
        return model_generation.MockModel(model_name=version or "mock")
    factory = MODEL_FACTORIES[provider]
    return factory(model_name=version) if version else factory()


def _json_result(result: utils.EpisodeResult) -> dict[str, Any]:
    def encode(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Episode contains a non-JSON value: {type(value).__name__}")

    return json.loads(json.dumps(result, default=encode, allow_nan=False))


def _failed_episode_result(
    *,
    run_type: str,
    game: str,
    models: list[str],
    error: str,
    episode_id: str,
    seed: int,
    metadata: dict[str, Any],
) -> utils.EpisodeResult:
    now = datetime.now()
    return cast(
        utils.EpisodeResult,
        {
            "episode_id": episode_id,
            "env": "OPENSPIEL",
            "game": game,
            "arena_type": run_type,
            "agent": models,
            "player_agents": models,
            "status": "FAILED",
            "winner": "UNDETERMINED/TIE",
            "num_steps": 0,
            "error": error,
            "summary": [],
            "action": [],
            "start_time": now,
            "end_time": now,
            "seed": seed,
            "returns": [],
            "terminal": False,
            "steps": [],
            "manifest": {
                **runtime_manifest(),
                "schema_version": 1,
                "game": game,
                "seed": seed,
                "metadata": metadata,
            },
        },
    )


def _prepare_progress(session: Session, run_id: str, matchup_id: str | None) -> None:
    """Backfill once for jobs queued before progress tables were introduced."""
    missing_run = session.get(RunProgress, run_id) is None
    missing_matchup = (
        matchup_id is not None and session.get(MatchupProgress, matchup_id) is None
    )
    if not missing_run and not missing_matchup:
        return
    # New submissions create counters up front. Only legacy jobs take this lock.
    session.query(Run).filter_by(run_id=run_id).with_for_update().one()
    if session.get(RunProgress, run_id) is None:
        session.add(
            RunProgress(
                run_id=run_id,
                completed=session.query(Episode).filter_by(run_id=run_id).count(),
                failed=session.query(Episode)
                .filter(Episode.run_id == run_id, Episode.status != "SUCCESS")
                .count(),
            )
        )
    if matchup_id is not None and session.get(MatchupProgress, matchup_id) is None:
        session.add(
            MatchupProgress(
                matchup_id=matchup_id,
                completed=session.query(Episode)
                .filter_by(matchup_id=matchup_id)
                .count(),
                failed=session.query(Episode)
                .filter(Episode.matchup_id == matchup_id, Episode.status != "SUCCESS")
                .count(),
            )
        )
    session.flush()


def _record_completion(
    session: Session, run_id: str, matchup_id: str | None, failed: bool
) -> None:
    # Acquire the shared counter lock after inserting the independent trace, so
    # large episode writes can proceed concurrently. Refresh cached ORM state
    # after waiting for another worker's commit.
    progress = (
        session.query(RunProgress)
        .filter_by(run_id=run_id)
        .populate_existing()
        .with_for_update()
        .one()
    )
    progress.completed += 1
    progress.failed += int(failed)
    run = session.get(Run, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    if progress.completed >= run.num_episodes:
        run.status = "FAILED" if progress.failed else "COMPLETED"
    if matchup_id is not None:
        matchup = session.get(Matchup, matchup_id)
        if matchup is None:
            raise ValueError(f"Matchup {matchup_id} not found")
        matchup_progress = (
            session.query(MatchupProgress)
            .filter_by(matchup_id=matchup_id)
            .populate_existing()
            .with_for_update()
            .one()
        )
        matchup_progress.completed += 1
        matchup_progress.failed += int(failed)
        if matchup_progress.completed >= matchup.num_episodes:
            matchup.status = "FAILED" if matchup_progress.failed else "COMPLETED"


def _persist_result(
    result: utils.EpisodeResult, run_id: str, matchup_id: str | None
) -> None:
    """A single transaction guarantees no partial traces or lost completion race.

    PostgreSQL serializes completion counters after independent trace inserts.
    Stable episode primary keys make at-least-once queue delivery safe.
    """
    session = get_session()
    try:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if session.get(Episode, result["episode_id"]) is not None:
            return
        _prepare_progress(session, run_id, matchup_id)
        payload = _json_result(result)
        transitions = payload.pop("steps", [])
        episode = Episode(
            episode_id=result["episode_id"],
            run_id=run_id,
            matchup_id=matchup_id,
            worker_name=get_worker_name(),
            env=result["env"],
            agent=result["agent"],
            type=result["arena_type"],
            game=result["game"].split("(", 1)[0][:50],
            status=result["status"],
            winner=result["winner"][:50],
            num_steps=result["num_steps"],
            error=result["error"][:255],
            summary=result["summary"],
            action=result["action"],
            start_time=result["start_time"],
            end_time=result["end_time"],
        )
        session.add(episode)
        session.flush()
        session.add(EpisodeTrace(episode_id=result["episode_id"], payload=payload))
        for transition in transitions:
            index = int(transition["index"])
            session.add(
                EpisodeTransition(
                    episode_id=result["episode_id"],
                    step_index=index,
                    payload=transition,
                )
            )
            # Retain the legacy normalized decision table for existing clients.
            # Simultaneous decisions share a transition index; chance is omitted.
            for decision in transition.get("decisions", []):
                session.add(
                    EpisodeStep(
                        step_id=str(
                            uuid.uuid5(
                                uuid.UUID(result["episode_id"]),
                                f"{index}:{decision['player']}",
                            )
                        ),
                        episode_id=result["episode_id"],
                        step_index=index,
                        player=str(decision["player"]),
                        action=str(decision["action_string"])[:50],
                    )
                )
        session.flush()
        _record_completion(session, run_id, matchup_id, result["status"] != "SUCCESS")
        session.commit()
        increment_counter(
            "worker.episode.persist.success", tags={"run_type": result["arena_type"]}
        )
    except IntegrityError:
        session.rollback()
        # Another delivery may have won the stable episode primary key while
        # this transaction was writing. Other constraint failures still fail.
        if session.get(Episode, result["episode_id"]) is None:
            raise
    except Exception:
        session.rollback()
        increment_counter(
            "worker.episode.persist.error", tags={"run_type": result["arena_type"]}
        )
        raise
    finally:
        session.close()


def _execute_episode(
    *,
    run_id: str,
    episode_num: int,
    num_steps: int | None,
    model_names: list[str],
    run_type: str,
    game: str,
    matchup_id: str | None = None,
    input_actions: list[int] | None = None,
    seed: int = 0,
    episode_id: str | None = None,
    rotate_seats: bool = True,
    metadata: dict[str, Any] | None = None,
    mean_field_distribution: str | dict[str, float] | None = None,
    mean_field_population: int = 0,
    failure_policy: str = "random",
) -> dict[str, Any]:
    identity = episode_id or str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"arena:{run_id}:{matchup_id}:{episode_num}")
    )
    metadata = dict(metadata or {})
    metadata.update(run_id=run_id, matchup_id=matchup_id, episode_num=episode_num)
    session = get_session()
    try:
        existing = session.get(EpisodeTrace, identity)
        if existing is not None:
            return {
                "run_id": run_id,
                "matchup_id": matchup_id,
                "episode_num": episode_num,
                "result": existing.payload,
                "reused": True,
            }
    finally:
        session.close()
    if rotate_seats and run_type == "PVP" and len(model_names) > 1:
        offset = episode_num % len(model_names)
        model_names = model_names[offset:] + model_names[:offset]
    increment_counter("worker.episode.started", tags={"run_type": run_type})
    try:
        expected_runtime = metadata.get("expected_runtime")
        if expected_runtime is not None:
            current_runtime = runtime_manifest()
            pinned_keys = (
                "source_sha256",
                "dependency_lock_sha256",
                "dependency_versions",
                "open_spiel_version",
                "python_version",
                "prompt_version",
                "parser_version",
            )
            mismatches = [
                key
                for key in pinned_keys
                if expected_runtime.get(key) != current_runtime.get(key)
            ]
            if mismatches:
                raise ValueError(
                    f"Worker runtime does not match the submitted run build: {', '.join(mismatches)}"
                )
        models = [_get_model(name) for name in model_names]
        options: dict[str, Any] = {
            "type": run_type,
            "num_steps": num_steps,
            "models": models,
            "input": deque(input_actions) if input_actions is not None else None,
            "game": game,
            "seed": seed,
            "episode_id": identity,
            "metadata": metadata,
            "mean_field_distribution": mean_field_distribution,
            "mean_field_population": mean_field_population,
            "failure_policy": failure_policy,
        }
        result = harness.harness(cast(utils.HarnessOptions, options))
    except Exception as exc:
        failure = classify_exception(
            exc,
            stage="worker.episode_execution",
            default_code="episode_execution_failed",
        )
        increment_counter(
            "worker.failure.total",
            tags={"category": failure.category.value, "code": failure.code},
        )
        LOGGER.exception(
            "episode_execution_failed",
            extra={"run_id": run_id, "episode_num": episode_num},
        )
        result = _failed_episode_result(
            run_type=run_type,
            game=game,
            models=model_names,
            error=f"{failure.compact()} {exc}",
            episode_id=identity,
            seed=seed,
            metadata=metadata,
        )
    _persist_result(result, run_id, matchup_id)
    increment_counter(
        "worker.episode.completed",
        tags={"run_type": run_type, "status": result["status"]},
    )
    return {
        "run_id": run_id,
        "matchup_id": matchup_id,
        "episode_num": episode_num,
        "result": result,
    }


def run_episode_pvp(
    matchup_id: str,
    run_id: str,
    episode_num: int,
    num_steps: int | None,
    game: str = "connect_four",
    **settings: Any,
) -> dict[str, Any]:
    session = get_session()
    try:
        matchup = session.get(Matchup, matchup_id)
        if matchup is None or matchup.run_id != run_id:
            raise ValueError(f"Matchup {matchup_id} not found in run {run_id}")
        model_names = list(matchup.models)
    finally:
        session.close()
    return _execute_episode(
        matchup_id=matchup_id,
        run_id=run_id,
        episode_num=episode_num,
        num_steps=num_steps,
        game=game,
        model_names=model_names,
        run_type="PVP",
        **settings,
    )


def run_episode_single(
    run_id: str,
    episode_num: int,
    num_steps: int | None,
    model_name: str,
    run_type: str,
    input_actions: list[int] | None = None,
    game: str = "connect_four",
    **settings: Any,
) -> dict[str, Any]:
    return _execute_episode(
        run_id=run_id,
        episode_num=episode_num,
        num_steps=num_steps,
        model_names=[model_name],
        run_type=run_type,
        input_actions=input_actions,
        game=game,
        **settings,
    )
