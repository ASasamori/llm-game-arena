# llm-game-arena

## Execution workflow

Client Request -> Run Orchestrator -> Workers -> Evals -> Client Polling

## REST API Gateway (Pipelined Run Submission)

Start the API gateway:

```bash
uv run python -m orchestrator.api_gateway
```

Required/important env vars:

```bash
# Comma-separated API keys accepted by the gateway
export ARENA_API_KEYS="dev-key-1,dev-key-2"

# Redis connection used for queue + API rate limiting
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Per-key limits
export API_RATE_LIMIT_REQUESTS_PER_MINUTE=60
export API_RATE_LIMIT_TOKENS_PER_MINUTE=30000

# Retry/backoff upper bound for model API calls
export MODEL_API_RETRY_MAX_SECONDS=90
```

Submit a run:

```bash
curl -X POST http://localhost:8000/v1/runs \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "run_type": "PVP",
    "env": "connect_four_v1",
    "num_episodes": 5,
    "num_steps": 42,
    "model_names": ["GEMINI", "GPT"]
  }'
```

Poll run status:

```bash
curl http://localhost:8000/v1/runs/<RUN_ID> \
  -H "Authorization: Bearer dev-key-1"
```

Check per-key minute usage counters (requests + tokenized request budget):

```bash
curl http://localhost:8000/v1/usage/me \
  -H "Authorization: Bearer dev-key-1"
```

### API Gateway Behavior (Current)
- Framework: FastAPI (`orchestrator/api_gateway.py`)
- Auth: `Authorization: Bearer <key>` or `x-api-key: <key>` where key is in `ARENA_API_KEYS`
- Run types supported by `POST /v1/runs`:
  - `PVP` (exactly 2 models)
  - `PVP_RR` (2+ models, round robin)
  - `RAND` (single model vs random opponent)
  - `INPUT` (single model with required `input_actions`, `num_episodes=1`)
- Backoff/retry: enqueue retry with exponential backoff on Redis/network transient failures
- Rate limiting:
  - Request budget per API key per minute
  - Tokenized payload budget per API key per minute
  - Usage visible via `GET /v1/usage/me`
- OpenAPI docs:
  - Swagger UI: `http://localhost:8000/docs`
  - OpenAPI schema: `http://localhost:8000/openapi.json`

### Observability (Current)
Observability utilities live in `orchestrator/observability.py`.

What it does:
- Configures structured JSON logs to `stdout`
- Tracks process-local counters/timings in memory

Where output appears:
- API logs: terminal/process running `python -m orchestrator.api_gateway`
- Worker/harness logs: terminal/process running `python -m orchestrator.worker` or `python -m orchestrator.runner`
- Redis terminal only shows Redis server logs (not app observability logs)

Important limitation:
- Metrics in `orchestrator/observability.py` are currently in-memory per process.
- They are not persisted to Postgres and not exported to Prometheus/Grafana yet.
- Process restart resets those counters/timings.

### Observability Export + Dashboard (New)
Export process-local metrics and a dashboard-oriented JSON summary:

```bash
uv run python -m orchestrator.observability_export \
  --metrics-out analytics/observability/metrics.json \
  --dashboard-out analytics/observability/dashboard.json
```

Read-only API endpoints (requires API key auth):
- `GET /v1/observability/metrics`
- `GET /v1/observability/dashboard`

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

### Run analytics CLI
After a PVP run is completed and persisted in Postgres, generate plots from the
`runs`, `matchups`, `episodes`, and `episode_steps` tables:

```bash
python -m orchestrator.analytics --run-id <PVP_RUN_ID>
```

Optional baseline mappings for delta-action plots (`MODEL_NAME_WITH_VERSION=RAND_RUN_ID`):

```bash
python -m orchestrator.analytics \
  --run-id <PVP_RUN_ID> \
  --baseline-run 'GPTModel(gpt-4o-mini)'=<RAND_RUN_ID_FOR_GPT_4O_MINI> \
  --baseline-run 'GeminiModel(gemini-2.5-flash-lite)'=<RAND_RUN_ID_FOR_GEMINI_2_5_FLASH_LITE> \
  --baseline-run 'ClaudeModel(claude-3-haiku-20240307)'=<RAND_RUN_ID_FOR_CLAUDE_HAIKU>
```

Outputs are written to `analytics/<run_id>/` by default:
- `heatmap_<MODEL>.png`
- `entropy_<MODEL>.png`
- `delta_<MODEL>.png` (when baselines are found/provided)

### Deterministic Mock E2E Pipeline (New)
One-command local flow for:
1) API run submission
2) Queue drain to completion
3) Run report export
4) Observability export

```bash
uv run python -m orchestrator.mock_e2e \
  --output-dir analytics/mock_e2e
```

Notes:
- Requires local Redis and Postgres configured via existing env vars.
- Uses `MODEL_API_MOCK_MODE=true` and deterministic `INPUT` flow by default.
- Writes:
  - `analytics/mock_e2e/report.json`
  - `analytics/mock_e2e/observability_metrics.json`
  - `analytics/mock_e2e/observability_dashboard.json`

### Elo ratings CLI
Create/update Elo ratings from completed PVP episodes. Ratings are persisted in:
- `model_ratings` (current aggregate rating rows)
- `rating_events` (immutable per-episode Elo update log)

If needed, create tables first:

```bash
python -m orchestrator.models
```

Incrementally rate one run:

```bash
uv run python -m orchestrator.ratings --run-id <PVP_RUN_ID>
```

Rebuild from all completed PVP data:

```bash
uv run python -m orchestrator.ratings --rebuild-all
```

Optional: export a JSON leaderboard summary:

```bash
uv run python -m orchestrator.ratings --rebuild-all --summary-json ratings_summary.json
```

### Health / Hung Detection CLI
Inspect queue and run health, including STARTED/DEFERRED jobs:

```bash
uv run python -m orchestrator.health --run-id <RUN_ID>
```

Mark stale `PENDING` runs as `FAILED` when no runnable jobs remain:

```bash
uv run python -m orchestrator.health --reconcile-stale --stale-seconds 900
```

Rate-limit/credit retry budget for model API calls is configurable:

```bash
export MODEL_API_RETRY_MAX_SECONDS=90
```

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
