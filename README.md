# llm-game-arena

## Execution workflow

Client Request -> Run Orchestrator -> Workers -> Evals -> Client Polling

### Client Request
API Gateway and client request logic
- Authorization
- Validation
- Rate-limiting logic

```JSON
{
    "agent": "opus-4.5",
    "env": "connect_four_v1",
    "episode_count": 1
}
```

```JSON
{
    "POST": SUCCESS
}
```

### Run Creation/Orchestrator
- Capture Run ID
- Retain run status `PENDING | COMPLETED | TERMINATED`
- Persistent run metadata
- Spawn epsidoe execution (parallelized episodes, since each episode is independent)
- Trigger aggregation for runs

### Workers & Episode Execution
- Create episode ID
- Initialize environment with ID
- Loop
    - Get observation
    - Call agent
    - Apply action
    - Store step

Environment version, random seed, and opponent policy implementation

### Evaluations
- Compute metrics
- Aggregate across episodes
- Deterministic scoring
- Must be done outside configured environment
- Step evals: (record, compute metrics)
- Epsidoe evals: (episode_metrics)
- Run evals: (aggregate run metrics)

### Client polling
- Status
- Metrics
- Episode count

---

[Google Game Arena](https://github.com/google-deepmind/game_arena?tab=readme-ov-file)

[OpenSpiel](https://openspiel.readthedocs.io/en/latest/games.html) (Google used this as well), but I want to use it for evals and automation at first, and then can move up to building more complex games

[Other LM Arena](https://github.com/lmarena/arena-hard-auto/tree/main), where it introduces [interesting research paper](https://arxiv.org/pdf/2406.11939)

### Critical:
- Create pipeline to do simple evals for OpenSpiel games
    - Route it to utils, show graphs, update benchmarks accordingly
    - Scale information and send it to Postgres DB
- Capture step analysis [Game-Arena](https://game-arena.ai/game/chess-standard):
    - Capture and detail other relevant data and quantitative measures

### Future:
- Create simple envs for complex games
    - Wavelength style games; Codenames, Hues and Cues, etc.
    - Explorative vs exploitative games
    - Adversarial and collaborative
- Batch process evals with AI-AI arena and human-AI style
    - Make generalizations and causal inference
- Allow open source modeling and fine-tuning at will (would be hard to batch process though)
- Make frontpage UI using [v0](https://v0.app/chat)

### Far future:
- Monetization?