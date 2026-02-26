import uuid
import logging
from collections import deque

from rq import get_current_job

from game_arena.openspiel import harness
from game_arena.openspiel import utils
from orchestrator.models import Episode, EpisodeStep, Run, Matchup, get_session
from game_arena.openspiel import model_generation
from orchestrator.observability import (
    configure_json_logging,
    increment_counter,
)
from orchestrator.failures import classify_exception


configure_json_logging(service="worker_tasks")
LOGGER = logging.getLogger("orchestrator.tasks")


def get_worker_name() -> str | None:
    """Get the name of the worker executing this job."""
    job = get_current_job()
    return job.worker_name if job else None


MODEL_FACTORIES = {
    "GEMINI": model_generation.GeminiModel,
    "GPT": model_generation.GPTModel,
    "CLAUDE": model_generation.ClaudeModel,
}
MODEL_CACHE: dict[str, model_generation.LLMModel] = {}

TERMINAL_MATCHUP_STATUSES = {"COMPLETED", "FAILED"}


def _get_model(model_name: str) -> model_generation.LLMModel:
    upper_name = model_name.upper()
    if upper_name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {list(MODEL_FACTORIES.keys())}"
        )
    if upper_name not in MODEL_CACHE:
        MODEL_CACHE[upper_name] = MODEL_FACTORIES[upper_name]()
    return MODEL_CACHE[upper_name]


def _failed_episode_result(
    *,
    run_type: str,
    game: str,
    models: list[str],
    num_steps: int | None,
    error: str,
) -> utils.EpisodeResult:
    now = utils.datetime.now()
    return utils.EpisodeResult(
        episode_id=str(uuid.uuid4()),
        env="OPENSPIEL",
        game=game,
        arena_type=run_type,
        agent=models,
        status="FAILED",
        winner="UNDETERMINED/TIE",
        num_steps=num_steps if isinstance(num_steps, int) else 0,
        error=error[:255],
        summary=["Episode failed before completion."],
        action=[],
        start_time=now,
        end_time=now,
    )


def run_episode_pvp(
    matchup_id: str, run_id: str, episode_num: int, num_steps: int | None
):
    """
    PVP episode: two models compete.
    Uses Matchup to track model pairs.
    """
    increment_counter("worker.episode.started", tags={"run_type": "PVP"})
    LOGGER.info(
        "episode_started",
        extra={
            "run_type": "PVP",
            "run_id": run_id,
            "matchup_id": matchup_id,
            "episode_num": episode_num,
            "num_steps": num_steps,
        },
    )

    # Look up matchup to get model pair
    session = get_session()
    matchup = session.query(Matchup).filter_by(matchup_id=matchup_id).first()
    if matchup is None:
        session.close()
        raise ValueError(f"Matchup {matchup_id} not found")
    model_names = matchup.models  # e.g., ["GEMINI", "GPT"]
    session.close()

    model_list = [_get_model(model) for model in model_names]
    harness_options = utils.HarnessOptions(
        type="PVP", num_steps=num_steps, models=model_list, input=None
    )
    try:
        result: utils.EpisodeResult = harness.harness(harness_options)
    except Exception as e:
        failure = classify_exception(
            e, stage="worker.episode_execution", default_code="episode_execution_failed"
        )
        increment_counter(
            "worker.episode.error", tags={"reason": "execution_exception"}
        )
        increment_counter(
            "worker.failure.total",
            tags={"category": failure.category.value, "code": failure.code},
        )
        error = (
            f"Episode execution failed for matchup {matchup_id[:8]} "
            f"episode {episode_num}: {failure.compact()}"
        )
        LOGGER.exception(
            "episode_execution_failed",
            extra={
                "run_type": "PVP",
                "run_id": run_id,
                "matchup_id": matchup_id,
                "episode_num": episode_num,
            },
        )
        result = _failed_episode_result(
            run_type="PVP",
            game="connect_four",
            models=[str(model) for model in model_list],
            num_steps=num_steps,
            error=error,
        )

    increment_counter("worker.episode.completed", tags={"run_type": "PVP"})
    if result["status"] != "SUCCESS":
        increment_counter(
            "worker.episode.non_success_status",
            tags={"run_type": "PVP", "status": result["status"]},
        )
    LOGGER.info(
        "episode_completed",
        extra={
            "run_type": "PVP",
            "run_id": run_id,
            "matchup_id": matchup_id,
            "episode_num": episode_num,
            "status": result["status"],
            "winner": result["winner"],
            "models": model_names,
        },
    )

    # Write episode result to Postgres
    session = get_session()
    try:
        episode = Episode(
            episode_id=result["episode_id"],
            run_id=run_id,
            matchup_id=matchup_id,
            worker_name=get_worker_name(),
            env=result["env"],
            agent=result["agent"],
            type=result["arena_type"],
            game=result["game"],
            status=result["status"],
            winner=result["winner"],
            num_steps=result["num_steps"],
            error=result["error"],
            summary=result["summary"],
            action=result["action"],
            start_time=result["start_time"],
            end_time=result["end_time"],
        )
        session.add(episode)

        # Create EpisodeStep records for each action
        actions = result.get("action") or []
        for step_index, action_str in enumerate(actions):
            if action_str and isinstance(action_str, str):
                step = EpisodeStep(
                    step_id=str(uuid.uuid4()),
                    episode_id=result["episode_id"],
                    step_index=step_index,
                    player=action_str[0].lower(),
                    action=action_str,
                )
                session.add(step)

        session.commit()
        increment_counter("worker.episode.persist.success", tags={"run_type": "PVP"})
        LOGGER.info(
            "episode_persisted",
            extra={
                "run_type": "PVP",
                "run_id": run_id,
                "matchup_id": matchup_id,
                "episode_id": result["episode_id"],
                "step_count": len(actions),
            },
        )

        _check_pvp_completion(session, matchup_id, run_id)
    except Exception:
        session.rollback()
        increment_counter("worker.episode.persist.error", tags={"run_type": "PVP"})
        LOGGER.exception(
            "episode_persist_failed",
            extra={
                "run_type": "PVP",
                "run_id": run_id,
                "matchup_id": matchup_id,
                "episode_num": episode_num,
            },
        )
        raise
    finally:
        session.close()

    return {
        "run_id": run_id,
        "matchup_id": matchup_id,
        "episode_num": episode_num,
        "result": result,
    }


def run_episode_single(
    run_id: str,
    episode_num: int,
    num_steps: int | None,
    model_name: str,
    run_type: str,
    input_actions: list[int] | None = None,
):
    """
    Single-model episode: RAND or INPUT mode.
    No matchup - links directly to run.
    """
    increment_counter("worker.episode.started", tags={"run_type": run_type})
    LOGGER.info(
        "episode_started",
        extra={
            "run_type": run_type,
            "run_id": run_id,
            "episode_num": episode_num,
            "num_steps": num_steps,
            "model_name": model_name,
        },
    )

    model = _get_model(model_name)

    # Build input queue if INPUT mode
    input_queue = deque(input_actions) if input_actions else None

    harness_options = utils.HarnessOptions(
        type=run_type,
        num_steps=num_steps,
        models=[model],
        input=input_queue,
    )
    try:
        result: utils.EpisodeResult = harness.harness(harness_options)
    except Exception as e:
        failure = classify_exception(
            e, stage="worker.episode_execution", default_code="episode_execution_failed"
        )
        increment_counter(
            "worker.episode.error", tags={"reason": "execution_exception"}
        )
        increment_counter(
            "worker.failure.total",
            tags={"category": failure.category.value, "code": failure.code},
        )
        error = (
            f"Episode execution failed for run {run_id[:8]} episode {episode_num}: "
            f"{failure.compact()}"
        )
        LOGGER.exception(
            "episode_execution_failed",
            extra={
                "run_type": run_type,
                "run_id": run_id,
                "episode_num": episode_num,
            },
        )
        result = _failed_episode_result(
            run_type=run_type,
            game="connect_four",
            models=[str(model)],
            num_steps=num_steps,
            error=error,
        )

    increment_counter("worker.episode.completed", tags={"run_type": run_type})
    if result["status"] != "SUCCESS":
        increment_counter(
            "worker.episode.non_success_status",
            tags={"run_type": run_type, "status": result["status"]},
        )
    LOGGER.info(
        "episode_completed",
        extra={
            "run_type": run_type,
            "run_id": run_id,
            "episode_num": episode_num,
            "status": result["status"],
            "winner": result["winner"],
            "model_name": model_name,
        },
    )

    # Write episode result to Postgres
    session = get_session()
    try:
        episode = Episode(
            episode_id=result["episode_id"],
            run_id=run_id,
            matchup_id=None,  # No matchup for single-model runs
            worker_name=get_worker_name(),
            env=result["env"],
            agent=result["agent"],
            type=result["arena_type"],
            game=result["game"],
            status=result["status"],
            winner=result["winner"],
            num_steps=result["num_steps"],
            error=result["error"],
            summary=result["summary"],
            action=result["action"],
            start_time=result["start_time"],
            end_time=result["end_time"],
        )
        session.add(episode)

        # Create EpisodeStep records for each action
        actions = result.get("action") or []
        for step_index, action_str in enumerate(actions):
            if action_str and isinstance(action_str, str):
                step = EpisodeStep(
                    step_id=str(uuid.uuid4()),
                    episode_id=result["episode_id"],
                    step_index=step_index,
                    player=action_str[0].lower(),
                    action=action_str,
                )
                session.add(step)

        session.commit()
        increment_counter("worker.episode.persist.success", tags={"run_type": run_type})
        LOGGER.info(
            "episode_persisted",
            extra={
                "run_type": run_type,
                "run_id": run_id,
                "episode_id": result["episode_id"],
                "step_count": len(actions),
            },
        )

        _check_run_completion(session, run_id)
    except Exception:
        session.rollback()
        increment_counter("worker.episode.persist.error", tags={"run_type": run_type})
        LOGGER.exception(
            "episode_persist_failed",
            extra={
                "run_type": run_type,
                "run_id": run_id,
                "episode_num": episode_num,
            },
        )
        raise
    finally:
        session.close()

    return {
        "run_id": run_id,
        "episode_num": episode_num,
        "result": result,
    }


def _check_pvp_completion(session, matchup_id: str, run_id: str):
    """Check matchup completion, then check if all matchups in run are done."""
    # 1. Check matchup completion
    matchup = session.query(Matchup).filter_by(matchup_id=matchup_id).first()
    if matchup is None:
        return

    completed_count = session.query(Episode).filter_by(matchup_id=matchup_id).count()

    if completed_count >= matchup.num_episodes:
        failed_count = (
            session.query(Episode)
            .filter(Episode.matchup_id == matchup_id, Episode.status != "SUCCESS")
            .count()
        )
        matchup.status = "FAILED" if failed_count > 0 else "COMPLETED"
        session.commit()
        increment_counter(
            "worker.matchup.status_transition", tags={"status": matchup.status}
        )
        LOGGER.info(
            "matchup_status_updated",
            extra={
                "run_id": run_id,
                "matchup_id": matchup_id,
                "status": matchup.status,
                "completed_count": completed_count,
                "num_episodes": matchup.num_episodes,
                "failed_count": failed_count,
            },
        )

        # 2. Check if all matchups in run are complete
        all_matchups = session.query(Matchup).filter_by(run_id=run_id).all()
        if all(m.status in TERMINAL_MATCHUP_STATUSES for m in all_matchups):
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                run.status = (
                    "FAILED"
                    if any(m.status == "FAILED" for m in all_matchups)
                    else "COMPLETED"
                )
                session.commit()
                increment_counter(
                    "worker.run.status_transition", tags={"status": run.status}
                )
                LOGGER.info(
                    "run_status_updated",
                    extra={
                        "run_id": run_id,
                        "status": run.status,
                        "reason": "all_matchups_terminal",
                    },
                )
    else:
        LOGGER.info(
            "matchup_progress",
            extra={
                "run_id": run_id,
                "matchup_id": matchup_id,
                "completed_count": completed_count,
                "num_episodes": matchup.num_episodes,
            },
        )


def _check_run_completion(session, run_id: str):
    """Check if all episodes for a run are done (for RAND/INPUT runs)."""
    run = session.query(Run).filter_by(run_id=run_id).first()
    if run is None:
        return

    completed_count = session.query(Episode).filter_by(run_id=run_id).count()

    if completed_count >= run.num_episodes:
        failed_count = (
            session.query(Episode)
            .filter(Episode.run_id == run_id, Episode.status != "SUCCESS")
            .count()
        )
        run.status = "FAILED" if failed_count > 0 else "COMPLETED"
        session.commit()
        increment_counter("worker.run.status_transition", tags={"status": run.status})
        LOGGER.info(
            "run_status_updated",
            extra={
                "run_id": run_id,
                "status": run.status,
                "completed_count": completed_count,
                "num_episodes": run.num_episodes,
                "failed_count": failed_count,
            },
        )
    else:
        LOGGER.info(
            "run_progress",
            extra={
                "run_id": run_id,
                "completed_count": completed_count,
                "num_episodes": run.num_episodes,
            },
        )
