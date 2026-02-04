import uuid
from redis import Redis
from rq import Queue

from orchestrator.tasks import run_episode

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("episodes", connection=redis_conn)


def enqueue_run(
    num_episodes: int = 1, num_games: int = 3, model_name: str = "Test Model"
):
    """
    Creates one run with multiple episodes.
    This function enqueues jobs but does NOT execute them.
    """
    run_id = str(uuid.uuid4())  # ONE run_id for all episodes
    job_ids = []

    print(f"[JOBS.PY: ENQUEUE] Creating run {run_id} with {num_episodes} episodes")

    # Enqueue each episode as a separate job
    for episode_id in range(num_episodes):
        job = queue.enqueue(
            run_episode,
            run_id=run_id,
            episode_id=episode_id,
            num_games=num_games,
            model_name=model_name,
            job_timeout="30m",  # Prevent jobs from hanging forever
        )
        job_ids.append(job.id)
        print(f"[JOBS.PY: ENQUEUE] Queued episode {episode_id} (job {job.id})")

    print(f"[JOBS.PY: ENQUEUE] Successfully queued {num_episodes} episodes for run {run_id}")
    return run_id, job_ids


if __name__ == "__main__":
    # This simulates what your FastAPI endpoint would do
    run_id, job_ids = enqueue_run(num_episodes=1, num_games=3, model_name="Test Model")
    print(f"\n[JOBS.PY: MAIN] Run ID: {run_id}")
    print(f"[JOBS.PY: `MAIN] Job IDs: {job_ids}")

# TODO:
# Next steps, update API key, create fast API to allow requests to occur and have proper error/return codes
# Maybe change up to see how it would work on the raspberry Pi, add an evaluation metric or analysis to aggregate the metrics,
# curate the environment to be hard coded steps instead of random steps for now, or rather have input (this would be quite hard)
# MOST IMPORTANT: Store information in the database; not sure where docker would play here?