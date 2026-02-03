import uuid
from redis import Redis
from rq import Queue

from orchestrator.tasks import run_episode

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("episodes", connection=redis_conn)


def enqueue_run(
    num_episodes: int = 5, num_games: int = 3, model_name: str = "Test Model"
):
    """
    Creates one run with multiple episodes.
    This function enqueues jobs but does NOT execute them.
    """
    run_id = str(uuid.uuid4())  # ONE run_id for all episodes
    job_ids = []

    print(f"[ENQUEUE] Creating run {run_id} with {num_episodes} episodes")

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
        print(f"[ENQUEUE] Queued episode {episode_id} (job {job.id})")

    print(f"[ENQUEUE] Successfully queued {num_episodes} episodes for run {run_id}")
    return run_id, job_ids


if __name__ == "__main__":
    # This simulates what your FastAPI endpoint would do
    run_id, job_ids = enqueue_run(num_episodes=5, num_games=3, model_name="Test Model")
    print(f"\n[MAIN] Run ID: {run_id}")
    print(f"[MAIN] Job IDs: {job_ids}")
