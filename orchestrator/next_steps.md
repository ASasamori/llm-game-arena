 - Create functionality to call upon environment episodes
    - Have it define specific features: games (not other ones right now), AI models, turns, skills of the player, (add more ideas later)
    - Should return structured data like `class EpsidoeResult`

    ```python
    def run_ep(ep_id: str, run_id: str, env_name: str, agent_config: dict, seed: int) -> EpisodeResult
    ```

After create that:
- FastAPI
- Redis
- Postgres

Create database schema (first real infra)
- 3 tables only for now:
1) Runs
```
run_id (uuid, pk)
status (text)
spec_json (jsonb)
epsidoes_total (int)
epsidoes_completed (int)
episodes_failed (int)
created_at (time)
```
2) Episodes
```
episode_id (uuid, pk)
run_id (fk)
status
winner
num_steps
error
```
3) Episode Steps
```
# Skip for now
```

## Create FastAPI control plane for now (Run Orchestrator)
- Endpoint must return immediately
- Nothing actually runs

## Add local queue
- Use Redis + RQ
queue.py
```python
epsidoe_queue.enque(
    "worker.run_epsidoe_task",
    run_id,
    epsidoe_id,
    env,
    agent
)
```

## Write worker (where to plug into the environment)

worker.py
```python
from env import run_epsidoe

def run_ep(run_id, ep_id, env, agent):
    res = run_ep(
        run_id,
        ep_id,`
        env,
        agent,
        seed
    )
```

And then the workers would run locally:
` rq workers epsidoes`

At this point, would have:
- Async execution
- Multiple workers
- Failure isolation

# Before AWS:
Check:
- POST /runs returns immediately
- Redis queue filled with episode jobs
- Workers consume jobs
- Episodes update DB
- Run eventually reaches "COMPLETED"

If done correctly, this will be runOrchestrator + execution plane

Aggregation + analytics
- Query episodes
- Compute win rate, average steps
- Store metrics on run

Can be created inline after every epsidoe or running background task