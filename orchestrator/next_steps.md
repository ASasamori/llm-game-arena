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

To do better concurrency:
  1. rq worker with the --burst flag + a launcher script — start N    
  workers from one command:                                           
                                                                      
  for i in $(seq 5); do python -m orchestrator.worker & done          
                                                                      
  All 5 run as background processes in one terminal. kill %1 %2 %3 %4 
  %5 or pkill -f orchestrator.worker to stop them.                                                                            
  3. Docker Compose (later) — when you're ready, define redis and     
  worker services with replicas: 5. This is the production-grade      
  version of what you're doing manually.                              
                                                                      
  4. Drop RQ entirely, use concurrent.futures in jobs.py — since      
  you're running everything on one machine anyway, you could skip     
  Redis/RQ and just do:                                               
                                                                      
  from concurrent.futures import ProcessPoolExecutor                  
                                                                      
  with ProcessPoolExecutor(max_workers=5) as pool:                    
      futures = [pool.submit(run_episode, run_id, i, 3, "Test Model") 
  for i in range(5)]                                                  
      results = [f.result() for f in futures]                         
                                                                      
  No Redis, no separate worker processes, no extra terminals. One     
  script, true parallelism.                                           
                                                                      
  Option 4 is the simplest for where you are now. RQ pays off when you
   need jobs submitted over time via an API, persistence across       
  restarts, or workers on different machines. If you're always running
   a fixed batch of episodes from one script, concurrent.futures does 
  the same thing with less infrastructure. Worth considering which    
  direction you want to go before building more on top of RQ.         

	•	Entropy reduction rate
	•	Policy divergence vs baseline
	•	Move consistency across episodes
	•	Symmetry breaking timing


When you'd want Celery over RQ:                
                                                 
  - Multiple task queues with different          
  priorities                                     
  - Task chaining/workflows (run A, then B, then 
  C)                                             
  - Rate limiting per-task                       
  - Canvas (groups, chords, chains)              
  - Massive scale (thousands of workers)   

