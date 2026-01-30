from typing import TypedDict

class EpisodeResult(TypedDict):
    episode_id: str
    run_id: str
    env: str
    game: str
    agent: str
    seed: int
    status: str
    winner: str
    num_steps: int
    error: str

def run_ep(ep_id: str, run_id: str, env: str, game: str, agent: str, seed: int) -> EpisodeResult:
    
    # RUN ID should not be part of the epsidoe result, part of the run for now
    return EpisodeResult(
        episode_id=ep_id,
        run_id=run_id,
        env=env,
        game=game,
        agent=agent,
        seed=seed,
        status="PENDING",
        winner="",
        num_steps=0,
        error="",
    )