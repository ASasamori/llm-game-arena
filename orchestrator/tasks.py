from game_arena.openspiel import harness
from game_arena.openspiel import utils
from orchestrator.models import Episode, Run, get_session
from game_arena.openspiel import model_generation

from collections import deque

MODEL_REGISTRY = {
    "GEMINI": model_generation.GeminiModel(),
    "GPT": model_generation.GPTModel(),
}

def run_episode(run_id: str, episode_id: int, num_games: int, model_name: list[str]):
    """
    Single job that runs one episode.
    Workers execute this function.
    """
    print(f"[TASKS.PY: WORKER] Starting episode {episode_id} for run {run_id}")

    # Check to see how the order is maintained? I think stays since list not tup
    model_list = [MODEL_REGISTRY[model] for model in model_name]
    try_run = utils.HarnessOptions(
        type="INPUT",
        num_moves=num_games,
        models = model_list,
        input=deque([0,1,2,4])
    )
    result: utils.EpisodeResult = harness.harness(try_run)

    print(f"[TASKS.PY: WORKER] Completed episode {episode_id} for run {run_id}")

    # Write episode result to Postgres
    session = get_session()
    try:
        episode = Episode(
            episode_id=result["episode_id"],
            run_id=run_id,
            env=result["env"],
            agent=result["agent"],
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
        session.commit()
        print(f"[TASKS.PY: DB] Saved episode {result['episode_id']} to database")

        _check_run_completion(session, run_id)
    except Exception as e:
        session.rollback()
        print(f"[TASKS.PY: DB] Failed to save episode: {e}")
        raise
    finally:
        session.close()

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "result": result,
    }


def _check_run_completion(session, run_id: str):
    """Check if all episodes for a run are done, update run status if so."""
    run = session.query(Run).filter_by(run_id=run_id).first()
    if run is None:
        return

    completed_count = session.query(Episode).filter_by(run_id=run_id).count()

    if completed_count >= run.num_episodes:
        run.status = "COMPLETED"
        session.commit()
        print(f"[TASKS.PY: DB] Run {run_id} marked COMPLETED ({completed_count}/{run.num_episodes})")
    else:
        print(f"[TASKS.PY: DB] Run {run_id} progress: {completed_count}/{run.num_episodes}")

#      │ 1. Start Redis, start worker, run python -m orchestrator.jobs                        │
#      │ 2. Check: psql -h <pi-ip> -U arena_user -d llm_game_arena -c "SELECT * FROM runs;"   │
#      │ 3. Check: psql -h <pi-ip> -U arena_user -d llm_game_arena -c "SELECT * FROM          │
#      │ episodes;"                                                                           │
#      │ 4. Verify run status flips from PENDING to COMPLETED after all episodes finish   