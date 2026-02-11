import uuid
from redis import Redis
from rq import Queue

from orchestrator.tasks import run_episode
from orchestrator.models import get_session, Run
from itertools import combinations

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("episodes", connection=redis_conn)


# This is where we should define the params for how diff runs should operate
def enqueue_run(
    num_episodes: int, num_steps: int, model_name: list[str]
):
    """
    Creates one run with multiple episodes.
    This function enqueues jobs but does NOT execute them.
    """
    run_id = str(uuid.uuid4())  # ONE run_id for all episodes

    session = get_session()
    try:
        new_run = Run(run_id=run_id, num_episodes=num_episodes, status="PENDING")
        session.add(new_run)
        session.commit()
        print(f"[JOBS.PY: DB] Created run {run_id} with status PENDING")
    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create run: {e}")
        raise
    finally:
        session.close()

    job_ids = []

    print(f"[JOBS.PY: ENQUEUE] Creating run {run_id} with {num_episodes} episodes")

    # Enqueue each episode as a separate job

    # Change this to use different models, can think about that later
    comb_list = list(combinations(model_name,2))
    for episode_id in range(num_episodes):
        models = comb_list.pop()
        job = queue.enqueue(
            run_episode,
            run_id=run_id,
            episode_id=episode_id,
            num_steps=num_steps,
            model_name=models,
            job_timeout="30m",  # Prevent jobs from hanging forever
        )
        job_ids.append(job.id)
        print(f"[JOBS.PY: ENQUEUE] Queued episode {episode_id} (job {job.id})")

    print(
        f"[JOBS.PY: ENQUEUE] Successfully queued {num_episodes} episodes for run {run_id}"
    )
    return run_id, job_ids


if __name__ == "__main__":
    # This simulates what your FastAPI endpoint would do
    # run_id, job_ids = enqueue_run(num_episodes=2, num_steps=3, model_name=["GEMINI","CLAUDE","GPT"])
    run_id, job_ids = enqueue_run(num_episodes=1, num_steps=3, model_name=["GEMINI","GPT"])
    print(f"\n[JOBS.PY: MAIN] Run ID: {run_id}")
    print(f"[JOBS.PY: `MAIN] Job IDs: {job_ids}")

# TODO:
# Next steps, update API key, create fast API to allow requests to occur and have proper error/return codes
# Maybe change up to see how it would work on the raspberry Pi, add an evaluation metric or analysis to aggregate the metrics,
