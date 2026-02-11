import uuid
from redis import Redis
from rq import Queue

from orchestrator.tasks import run_episode
from orchestrator.models import get_session, Run, Matchup
from itertools import combinations

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("episodes", connection=redis_conn)


# This is where we should define the params for how diff runs should operate
def enqueue_run(num_episodes: int, num_steps: int, model_names: list[str]):
    """
    Creates one run with matchups for each model pair.
    Each matchup gets num_episodes jobs enqueued.
    This function enqueues jobs but does NOT execute them.
    """
    run_id = str(uuid.uuid4())

    session = get_session()
    try:
        # 1. Create the Run
        new_run = Run(run_id=run_id, status="PENDING")
        session.add(new_run)

        # 2. Create Matchups for each model combination
        comb_list = list(combinations(model_names, 2))
        matchup_ids = []

        for models in comb_list:
            matchup_id = str(uuid.uuid4())
            matchup = Matchup(
                matchup_id=matchup_id,
                run_id=run_id,
                models=list(models),
                num_episodes=num_episodes,
                status="PENDING",
            )
            session.add(matchup)
            matchup_ids.append((matchup_id, models))

        session.commit()
        print(f"[JOBS.PY: DB] Created run {run_id} with {len(comb_list)} matchups")

    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create run: {e}")
        raise
    finally:
        session.close()

    # 3. Enqueue episodes for each matchup
    job_ids = []
    total_episodes = len(comb_list) * num_episodes

    print(f"[JOBS.PY: ENQUEUE] Creating {total_episodes} total episodes ({num_episodes} per matchup)")

    for matchup_id, models in matchup_ids:
        for ep_num in range(num_episodes):
            job = queue.enqueue(
                run_episode,
                matchup_id=matchup_id,
                run_id=run_id,
                episode_num=ep_num,
                num_steps=num_steps,
                job_timeout="30m",
            )
            job_ids.append(job.id)
            print(f"[JOBS.PY: ENQUEUE] Queued {models} episode {ep_num} (job {job.id})")

    print(f"[JOBS.PY: ENQUEUE] Successfully queued {total_episodes} episodes for run {run_id}")
    return run_id, job_ids


if __name__ == "__main__":
    # This simulates what your FastAPI endpoint would do
    run_id, job_ids = enqueue_run(num_episodes=2, num_steps=3, model_names=["GEMINI", "GPT", "CLAUDE"])
    print(f"\n[JOBS.PY: MAIN] Run ID: {run_id}")
    print(f"[JOBS.PY: MAIN] Job IDs: {job_ids}")

# TODO:
# Next steps, update API key, create fast API to allow requests to occur and have proper error/return codes
# Maybe change up to see how it would work on the raspberry Pi, add an evaluation metric or analysis to aggregate the metrics,
