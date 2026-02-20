import uuid
from itertools import combinations

from redis import Redis
from rq import Queue

from orchestrator.tasks import run_episode_pvp, run_episode_single
from orchestrator.models import get_session, Run, Matchup

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("episodes", connection=redis_conn)

SUPPORTED_TYPES = ["INPUT", "RAND", "PVP", "PVP_RR"]


def enqueue_pvp_run(num_episodes: int, num_steps: int, model_names: list[str]):
    """
    Creates a PVP run for exactly one model pair.
    Jobs are chained with dependencies so episode order stays deterministic.
    """
    if len(model_names) != 2:
        raise ValueError(
            f"PVP requires exactly 2 models (got {len(model_names)}): {model_names}"
        )

    run_id = str(uuid.uuid4())
    total_episodes = num_episodes

    session = get_session()
    try:
        # Is there a way to get the worker that would call this function?
        new_run = Run(
            run_id=run_id,
            run_type="PVP",
            num_episodes=total_episodes,
            status="PENDING",
        )
        session.add(new_run)

        matchup_id = str(uuid.uuid4())
        matchup = Matchup(
            matchup_id=matchup_id,
            run_id=run_id,
            models=list(model_names),
            num_episodes=num_episodes,
            status="PENDING",
        )
        session.add(matchup)

        session.commit()
        print(f"[JOBS.PY: DB] Created PVP run {run_id} for models {model_names}")

    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create run: {e}")
        raise
    finally:
        session.close()

    job_ids = []
    print(
        f"[JOBS.PY: ENQUEUE] Creating {total_episodes} total episodes for matchup {model_names}"
    )

    previous_job = None
    for ep_num in range(num_episodes):
        job = queue.enqueue(
            run_episode_pvp,
            matchup_id=matchup_id,
            run_id=run_id,
            episode_num=ep_num,
            num_steps=num_steps,
            depends_on=previous_job,
            job_timeout="30m",
        )
        previous_job = job
        job_ids.append(job.id)
        print(
            f"[JOBS.PY: ENQUEUE] Queued PVP {model_names} episode {ep_num} (job {job.id})"
        )

    print(
        f"[JOBS.PY: ENQUEUE] Successfully queued {total_episodes} PVP episodes for run {run_id}"
    )
    return run_id, job_ids


def enqueue_pvp_round_robin_run(
    num_episodes: int, num_steps: int, model_names: list[str]
):
    """
    Creates a round-robin PVP run across all unique model pairs.
    Each pair gets its own matchup; episodes are serialized within each matchup.
    """
    if len(model_names) < 2:
        raise ValueError(
            f"PVP_RR requires at least 2 models (got {len(model_names)}): {model_names}"
        )

    run_id = str(uuid.uuid4())
    pairings = list(combinations(model_names, 2))
    total_episodes = len(pairings) * num_episodes

    session = get_session()
    try:
        new_run = Run(
            run_id=run_id,
            run_type="PVP_RR",
            num_episodes=total_episodes,
            status="PENDING",
        )
        session.add(new_run)

        matchup_data = []
        for pairing in pairings:
            matchup_id = str(uuid.uuid4())
            matchup = Matchup(
                matchup_id=matchup_id,
                run_id=run_id,
                models=list(pairing),
                num_episodes=num_episodes,
                status="PENDING",
            )
            session.add(matchup)
            matchup_data.append((matchup_id, pairing))

        session.commit()
        print(
            f"[JOBS.PY: DB] Created PVP_RR run {run_id} with {len(pairings)} matchups"
        )
    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create round-robin run: {e}")
        raise
    finally:
        session.close()

    job_ids = []
    print(
        f"[JOBS.PY: ENQUEUE] Creating {total_episodes} total episodes "
        f"({num_episodes} per matchup)"
    )
    for matchup_id, pairing in matchup_data:
        previous_job = None
        for ep_num in range(num_episodes):
            job = queue.enqueue(
                run_episode_pvp,
                matchup_id=matchup_id,
                run_id=run_id,
                episode_num=ep_num,
                num_steps=num_steps,
                depends_on=previous_job,
                job_timeout="30m",
            )
            previous_job = job
            job_ids.append(job.id)
            print(
                f"[JOBS.PY: ENQUEUE] Queued PVP_RR {pairing} episode {ep_num} "
                f"(job {job.id})"
            )

    print(
        f"[JOBS.PY: ENQUEUE] Successfully queued {total_episodes} PVP_RR episodes "
        f"for run {run_id}"
    )
    return run_id, job_ids


def enqueue_rand_run(num_episodes: int, num_steps: int, model_name: str):
    """
    Creates a RAND run: single model vs random opponent.
    No matchups - episodes link directly to run.
    """
    run_id = str(uuid.uuid4())

    session = get_session()
    try:
        new_run = Run(
            run_id=run_id,
            run_type="RAND",
            num_episodes=num_episodes,
            status="PENDING",
        )
        session.add(new_run)
        session.commit()
        print(f"[JOBS.PY: DB] Created RAND run {run_id} with {num_episodes} episodes")

    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create run: {e}")
        raise
    finally:
        session.close()

    job_ids = []
    for ep_num in range(num_episodes):
        job = queue.enqueue(
            run_episode_single,
            run_id=run_id,
            episode_num=ep_num,
            num_steps=num_steps,
            model_name=model_name,
            run_type="RAND",
            job_timeout="30m",
        )
        job_ids.append(job.id)
        print(
            f"[JOBS.PY: ENQUEUE] Queued RAND {model_name} episode {ep_num} (job {job.id})"
        )

    print(
        f"[JOBS.PY: ENQUEUE] Successfully queued {num_episodes} RAND episodes for run {run_id}"
    )
    return run_id, job_ids


def enqueue_input_run(num_steps: int, model_name: str, input_actions: list[int]):
    """
    Creates an INPUT run: single model vs predefined input actions.
    No matchups - single episode per run (input is specific to one game).
    """
    run_id = str(uuid.uuid4())

    session = get_session()
    try:
        new_run = Run(
            run_id=run_id,
            run_type="INPUT",
            num_episodes=1,
            status="PENDING",
        )
        session.add(new_run)
        session.commit()
        print(f"[JOBS.PY: DB] Created INPUT run {run_id}")

    except Exception as e:
        session.rollback()
        print(f"[JOBS.PY: DB] Failed to create run: {e}")
        raise
    finally:
        session.close()

    job = queue.enqueue(
        run_episode_single,
        run_id=run_id,
        episode_num=0,
        num_steps=num_steps,
        model_name=model_name,
        run_type="INPUT",
        input_actions=input_actions,
        job_timeout="30m",
    )
    print(f"[JOBS.PY: ENQUEUE] Queued INPUT {model_name} episode (job {job.id})")

    return run_id, [job.id]


def enqueue_run(
    run_type: str,
    num_episodes: int,
    num_steps: int,
    model_names: list[str],
    input_actions: list[int] | None = None,
) -> tuple[str, list[str]]:
    """
    Unified entry point for creating runs of any type.

    Args:
        run_type: "PVP", "PVP_RR", "RAND", or "INPUT"
        num_episodes: Number of episodes per matchup (PVP) or total (RAND). Ignored for INPUT.
        num_steps: Max moves per episode
        model_names: List of model names. PVP uses exactly 2.
        PVP_RR uses all pairwise combinations.
        RAND/INPUT use the first model.
        input_actions: Required for INPUT type - list of action integers for opponent

    Returns:
        Tuple of (run_id, job_ids)
    """
    if run_type not in SUPPORTED_TYPES:
        raise ValueError(f"run_type must be one of {SUPPORTED_TYPES}, got {run_type}")

    if run_type == "PVP":
        return enqueue_pvp_run(num_episodes, num_steps, model_names)
    elif run_type == "PVP_RR":
        return enqueue_pvp_round_robin_run(num_episodes, num_steps, model_names)
    elif run_type == "RAND":
        return enqueue_rand_run(num_episodes, num_steps, model_names[0])
    else:  # INPUT
        if input_actions is None:
            raise ValueError("INPUT run requires input_actions")
        return enqueue_input_run(num_steps, model_names[0], input_actions)


if __name__ == "__main__":
    # Uncomment the run type you want to test:
    run_id = None
    job_ids = None

    # PVP: One matchup with exactly two models
    # run_id, job_ids = enqueue_run(
    #     run_type="PVP",
    #     num_episodes=5,
    #     num_steps=6,
    #     model_names=["GEMINI", "GPT"],
    # )

    # PVP_RR: Round-robin across all model pairs in one run
    # run_id, job_ids = enqueue_run(
    #     run_type="PVP_RR",
    #     num_episodes=5,
    #     num_steps=6,
    #     model_names=["CLAUDE", "GEMINI", "GPT"],
    # )

    # RAND: Single model vs random opponent
    # run_id, job_ids = enqueue_run(
    #     run_type="RAND",
    #     num_episodes=10,
    #     num_steps=3,
    #     model_names=["GPT"],
    # )

    # INPUT: Single model vs predefined moves
    # run_id, job_ids = enqueue_run(
    #     run_type="INPUT",
    #     num_episodes=2,
    #     num_steps=3,
    #     model_names=["GEMINI"],
    #     input_actions=[0, 1, 2, 3],
    # )

    run_id, job_ids = enqueue_run(
        run_type="RAND",
        num_episodes=10,
        num_steps=3,
        model_names=["GPT"],
    )

    print(f"\n[JOBS.PY: MAIN] Run ID: {run_id}")
    print(f"[JOBS.PY: MAIN] Job IDs: {job_ids}")
