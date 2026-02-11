from game_arena.openspiel import harness
from game_arena.openspiel import utils
from orchestrator.models import Episode, Run, Matchup, get_session
from game_arena.openspiel import model_generation


MODEL_REGISTRY = {
    "GEMINI": model_generation.GeminiModel(),
    "GPT": model_generation.GPTModel(),
    "CLAUDE": model_generation.ClaudeModel(),
}


def run_episode(matchup_id: str, run_id: str, episode_num: int, num_steps: int):
    """
    Single job that runs one episode.
    Workers execute this function.
    """
    print(f"[TASKS.PY: WORKER] Starting episode {episode_num} for matchup {matchup_id[:8]}")

    # Look up matchup to get model pair
    session = get_session()
    matchup = session.query(Matchup).filter_by(matchup_id=matchup_id).first()
    if matchup is None:
        session.close()
        raise ValueError(f"Matchup {matchup_id} not found")
    model_names = matchup.models  # e.g., ["GEMINI", "GPT"]
    session.close()

    model_list = [MODEL_REGISTRY[model] for model in model_names]
    try_run = utils.HarnessOptions(
        type="PVP", num_steps=num_steps, models=model_list, input=None
    )
    result: utils.EpisodeResult = harness.harness(try_run)

    print(f"[TASKS.PY: WORKER] Completed episode {episode_num} for matchup {matchup_id[:8]} ({model_names})")

    # Write episode result to Postgres
    session = get_session()
    try:
        episode = Episode(
            episode_id=result["episode_id"],
            run_id=run_id,
            matchup_id=matchup_id,
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
        session.commit()
        print(f"[TASKS.PY: DB] Saved episode {result['episode_id']} to database")

        _check_completion(session, matchup_id, run_id)
    except Exception as e:
        session.rollback()
        print(f"[TASKS.PY: DB] Failed to save episode: {e}")
        raise
    finally:
        session.close()

    return {
        "run_id": run_id,
        "matchup_id": matchup_id,
        "episode_num": episode_num,
        "result": result,
    }


def _check_completion(session, matchup_id: str, run_id: str):
    """Check matchup completion, then check if all matchups in run are done."""
    # 1. Check matchup completion
    matchup = session.query(Matchup).filter_by(matchup_id=matchup_id).first()
    if matchup is None:
        return

    completed_count = session.query(Episode).filter_by(matchup_id=matchup_id).count()

    if completed_count >= matchup.num_episodes:
        matchup.status = "COMPLETED"
        session.commit()
        print(
            f"[TASKS.PY: DB] Matchup {matchup_id[:8]} marked COMPLETED ({completed_count}/{matchup.num_episodes})"
        )

        # 2. Check if all matchups in run are complete
        all_matchups = session.query(Matchup).filter_by(run_id=run_id).all()
        if all(m.status == "COMPLETED" for m in all_matchups):
            run = session.query(Run).filter_by(run_id=run_id).first()
            if run:
                run.status = "COMPLETED"
                session.commit()
                print(f"[TASKS.PY: DB] Run {run_id[:8]} marked COMPLETED (all matchups done)")
    else:
        print(
            f"[TASKS.PY: DB] Matchup {matchup_id[:8]} progress: {completed_count}/{matchup.num_episodes}"
        )


#      │ 1. Start Redis, start worker, run python -m orchestrator.jobs                        │
#      │ 2. Check: psql -h <pi-ip> -U arena_user -d llm_game_arena -c "SELECT * FROM runs;"   │
#      │ 3. Check: psql -h <pi-ip> -U arena_user -d llm_game_arena -c "SELECT * FROM          │
#      │ episodes;"                                                                           │
#      │ 4. Verify run status flips from PENDING to COMPLETED after all episodes finish
