# llm-game-arena

## Execution workflow

Client Request -> Run Orchestrator -> Workers -> Evals -> Client Polling

## REST API Gateway (Pipelined Run Submission)

### Docker Compose local stack

Run the full local stack (Postgres + Redis + API + runner):

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

The compose stack defaults to mock model responses (`MODEL_API_MOCK_MODE=true`),
so you can test end-to-end without external model API keys.

Submit a local test run:

```bash
curl -X POST http://localhost:8000/v1/runs \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "run_type": "PVP",
    "env": "connect_four_v1",
    "num_episodes": 2,
    "num_steps": 42,
    "model_names": ["GEMINI", "GPT"]
  }'
```

Stop and remove containers:

```bash
docker compose down
```

Stop and remove containers plus DB volume:

```bash
docker compose down -v
```

Start the API gateway:

```bash
uv run python -m orchestrator.api_gateway
```

Required/important env vars:

```bash
# Comma-separated API keys accepted by the gateway
export ARENA_API_KEYS="dev-key-1,dev-key-2"

# Database target switch (supports raspberry_pi or gcp)
export DATABASE_TARGET=raspberry_pi

# Option 1 (recommended): full URLs per target
export DATABASE_URL_RASPBERRY_PI="postgresql+psycopg2://<user>:<pass>@100.116.74.50:5432/llm_game_arena"
export DATABASE_URL_GCP="postgresql+psycopg2://<user>:<pass>@<gcp-host>:5432/llm_game_arena"

# Option 2: one direct override URL (takes precedence over target vars)
# export DATABASE_URL="postgresql+psycopg2://<user>:<pass>@<host>:5432/llm_game_arena"

# Redis connection used for queue + API rate limiting
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Per-key limits
export API_RATE_LIMIT_REQUESTS_PER_MINUTE=60
export API_RATE_LIMIT_TOKENS_PER_MINUTE=30000

# Retry/backoff upper bound for model API calls
export MODEL_API_RETRY_MAX_SECONDS=90
```

Quick target switch helper:

```bash
eval "$(scripts/db_target_env.sh raspberry_pi)"
# or:
eval "$(scripts/db_target_env.sh gcp)"
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

List run episodes (paginated):

```bash
curl "http://localhost:8000/v1/runs/<RUN_ID>/episodes?limit=50&offset=0" \
  -H "Authorization: Bearer dev-key-1"
```

Fetch run analytics as JSON for frontend charting:

```bash
curl "http://localhost:8000/v1/runs/<RUN_ID>/analytics?action_space=7" \
  -H "Authorization: Bearer dev-key-1"
```

Fetch leaderboard data:

```bash
curl "http://localhost:8000/v1/leaderboard?game=connect_four&limit=25" \
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

### GCP Deployment Path (Cloud Run)
Use `infra/gcp/README.md` and `infra/gcp/deploy_cloud_run.sh` to deploy:

- API gateway as Cloud Run service
- Worker as Cloud Run Job (`WORKER_BURST=true`)
- Memorystore Redis + Cloud SQL Postgres over VPC connector

To run long-lived workers instead of burst jobs:

```bash
export WORKER_EXECUTION_MODE=service_long_lived
export RUNNER_NUM_WORKERS=2
bash infra/gcp/deploy_cloud_run.sh
```

Run a smoke test from Cloud Shell with `scripts/gcp_test_client.sh`:

```bash
export API_URL="https://<your-api>.a.run.app"
export API_KEY="<api-key>"
bash scripts/gcp_test_client.sh
```

### One-command GCP On/Off
Use `scripts/gcp_stack_toggle.sh` to bring the full managed stack up or down.

Bring stack up:

```bash
bash scripts/gcp_stack_toggle.sh up
```

Turn stack down:

```bash
bash scripts/gcp_stack_toggle.sh down
```

`up` creates/restores Cloud SQL, Redis, VPC connector, and deploys API + worker job.
`down` deletes Cloud Run/Redis/VPC resources and stops Cloud SQL compute.

Run the same smoke test from any other machine (no `gcloud` required):

```bash
export API_URL="https://<your-api>.a.run.app"
export API_KEY="<api-key>"
bash scripts/gcp_test_client.sh
```

### Google Cloud Runbook (No Pi Required)
Once these are provisioned in GCP, the Raspberry Pi is no longer required for runtime execution:

- Cloud Run service: `llm-arena-api` (API Gateway)
- Cloud Run job: `llm-arena-worker` (RQ burst worker)
- Cloud SQL Postgres: `llm-arena-pg`
- Memorystore Redis: `llm-arena-redis`
- Serverless VPC connector: `llm-arena-connector`
- Secret Manager: database URL, API keys, provider keys

#### Cheapest stable option
Managed Cloud SQL + Memorystore + VPC connector can exceed a very small budget quickly.
If you want a long-lived system around `< $10/month`, prefer one small VM and run Docker Compose
(`api + worker + redis + postgres`) on that VM.

Use GCP managed stack when you need managed failover/ops. Use single-VM stack when cost is priority.

#### On-demand worker trigger (no scheduler)
Current flow:

1. `POST /v1/runs` validates and enqueues into Redis
2. API triggers Cloud Run Job `llm-arena-worker` through Cloud Run Jobs API
3. Worker drains queue and persists episode rows into Postgres
4. Client polls run/episodes/analytics endpoints

This keeps workers non-long-lived and avoids minute-based scheduler churn.

Required IAM:
- API service account needs permission to run the worker job (`run.jobs.run`)
- Practical role set: `roles/run.invoker` + `roles/run.developer` on the worker job

#### Cloud Shell Smoke Test
Use these exact exports:

```bash
export API_URL="$(gcloud run services describe llm-arena-api --region us-central1 --format='value(status.url)')"
export API_KEY="$(gcloud secrets versions access latest --secret arena-api-keys | cut -d',' -f1)"
```

Then:

```bash
curl -sS "$API_URL/v1/usage/me" -H "Authorization: Bearer $API_KEY" | jq .
curl -sS -X POST "$API_URL/v1/runs" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"run_type":"PVP","env":"connect_four_v1","num_episodes":1,"num_steps":10,"model_names":["GEMINI","GPT"]}' | jq .
```

Run status check:

```bash
curl -sS "$API_URL/v1/runs/<RUN_ID>" -H "Authorization: Bearer $API_KEY" | jq .
```

#### Common failure modes

- `404` HTML page from Google for `/healthz`
  - You are hitting the wrong URL/region/service.
  - Re-resolve URL:
    ```bash
    gcloud run services list --region us-central1 --format='table(metadata.name,status.url)'
    gcloud run services describe llm-arena-api --region us-central1 --format='value(status.url)'
    ```
- `curl` hangs on `POST /v1/runs` or returns `503`
  - First enforce timeout so it fails fast:
    ```bash
    curl -i --connect-timeout 5 --max-time 20 "$API_URL/v1/usage/me" \
      -H "Authorization: Bearer $API_KEY"
    ```
  - If `/healthz` is OK but `/v1/runs` hangs, inspect API logs:
    ```bash
    gcloud run services logs read llm-arena-api --region us-central1 --limit=200
    ```
  - Typical cause: `REDIS_HOST` points to a non-reachable address (for GCP it must be Memorystore private IP via VPC connector).
  - Another common cause is an API service revision started without the explicit API entrypoint command. Re-run deployment with the updated `infra/gcp/deploy_cloud_run.sh` (it now sets `--command .venv/bin/python --args -m,orchestrator.api_gateway` for the API service).
  - Another common cause is uninitialized database schema in a fresh Cloud SQL instance. Startup now auto-initializes tables (`DB_AUTO_INIT_SCHEMA=true`).

#### Multi-game in API
`POST /v1/runs` supports:
- `env: "connect_four_v1"`
- `env: "chess_v1"`

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

Simple stylized client response endpoint:

```bash
curl -X POST http://localhost:8000/v1/client/stylize \
  -H "Authorization: Bearer dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Arena Run Snapshot",
    "summary": "Run accepted and processing",
    "highlights": ["target=raspberry_pi", "workers=long_lived", "queue=episodes"],
    "tone": "technical"
  }'
```

Or use the helper script:

```bash
export API_URL=http://localhost:8000
export API_KEY=dev-key-1
bash scripts/stylized_client_request.sh
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

### Testcases (New)
Run unit tests for DB switching + stylized client endpoint:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Run a local build test run using mock model behavior and local API auth key:

```bash
export MODEL_API_MOCK_MODE=true
export MODEL_API_MOCK_CASE=default
export ARENA_API_KEYS=local-dev-key
uv run python -m orchestrator.mock_e2e --api-key local-dev-key --output-dir analytics/mock_e2e
```

### Guardrails + Failure Taxonomy (New)
Shared failure taxonomy is implemented in `orchestrator/failures.py` and propagated to:
- API unhandled exception responses
- Worker episode execution errors
- Harness fallback/error metrics

Taxonomy categories include:
- `validation`, `auth`, `rate_limit`, `model_api`, `parser`, `invalid_action`, `queue`, `db`, `timeout`, `internal`

Model response guardrails:
- `MODEL_API_MAX_RESPONSE_CHARS` (default `8000`)
- empty response rejection
- deterministic injected mock failure case: `MODEL_API_MOCK_CASE=raise_error`

Harness guardrails:
- deterministic fallback RNG via `HARNESS_RANDOM_SEED`
- legal-action precheck before apply
- categorized fallback metrics under `harness.failure.total`

### Parallelization Tuning (New)
Job scheduling and worker concurrency are now configurable:

```bash
export RUNNER_NUM_WORKERS=4
export ORCH_JOB_TIMEOUT_SECONDS=1800
export ORCH_SERIALIZE_PVP_EPISODES=true
export ORCH_SERIALIZE_PVP_RR_EPISODES=false
```

Notes:
- `ORCH_SERIALIZE_PVP_EPISODES=true` preserves strict determinism inside one PVP matchup.
- Setting either serialize flag to `false` allows higher throughput by removing intra-matchup dependency chaining.

### Failure Evidence Demo
Run a local harness-only demo that proves:
- successful episodes
- recoverable fallbacks
- hard failures that propagate with taxonomy

```bash
python -m orchestrator.failure_demo
```

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
