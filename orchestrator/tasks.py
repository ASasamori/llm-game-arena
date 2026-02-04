from game_arena.openspiel.harness import harness, EpisodeResult

def run_episode(run_id: str, episode_id: int, num_games: int, model_name: str):
    """
    Single job that runs one episode.
    Workers execute this function.
    """
    print(f"[TASKS.PY: WORKER] Starting episode {episode_id} for run {run_id}")

    result = harness(num_games, model_name)
    # The result will be of the form of EpisodeResult
    
    print(f"[TAKS.PY: WORKER] Summary list = {result["summary"]} \n\n\n")
    print(f"[TAKS.PY: WORKER] Action list = {result["action"]} \n\n\n")
    print(f"[TAKS.PY: WORKER] Completed episode {episode_id} for run {run_id}")

    # TODO: Append to POSTGRES here

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "result": result,
    }
