from game_arena.openspiel.harness import harness


def run_episode(run_id: str, episode_id: int, num_games: int, model_name: str):
    """
    Single job that runs one episode.
    Workers execute this function.
    """
    print(f"[WORKER] Starting episode {episode_id} for run {run_id}")

    result = harness(num_games, model_name)

    print(f"[WORKER] Completed episode {episode_id} for run {run_id}")

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "result": result,
    }
