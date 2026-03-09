# Google Cloud Deployment Handoff (LLM-Game-Arena)

This document explains what has already been created in Google Cloud, how your cloud runtime maps to the codebase, what to verify, and what remains to ship a production product.

## 1) What was created in GCP

Project used:
- `gen-lang-client-0155805792` (`LLM-Game-Arena`)

Provisioned resources:
- Cloud SQL (Postgres): `llm-arena-pg`
  - Region: `us-central1`
  - Engine: `POSTGRES_16`
  - Tier: `db-custom-1-3840`
  - Private IP: `192.168.0.3`
  - Database: `llm_game_arena`
  - User: `arena_user`
- Memorystore Redis: `llm-arena-redis`
  - Region: `us-central1`
  - Tier: `BASIC`
  - Size: `1GB`
  - Host: `192.168.1.3`
  - Port: `6379`
- Serverless VPC Access Connector: `llm-arena-connector`
  - CIDR: `10.8.0.0/28`
  - State: `READY`
- Private Service Access reserved range:
  - `google-managed-services-default` (`192.168.0.0/16`)
- Secret Manager secrets:
  - `arena-database-url`
  - `arena-api-keys`
  - `arena-openai-key`
  - `arena-gemini-key`
  - `arena-anthropic-key`
- Cloud Run service:
  - `llm-arena-api`
- Cloud Run job:
  - `llm-arena-worker`
- Cloud Scheduler job:
  - `llm-arena-worker-minute`
  - POSTs every minute to the Cloud Run Jobs API endpoint to run `llm-arena-worker`.

Enabled APIs:
- `run.googleapis.com`
- `sqladmin.googleapis.com`
- `redis.googleapis.com`
- `vpcaccess.googleapis.com`
- `secretmanager.googleapis.com`
- `servicenetworking.googleapis.com`
- `artifactregistry.googleapis.com`
- `cloudbuild.googleapis.com`
- `cloudscheduler.googleapis.com`

## 2) What code was added/updated for cloud operations

### Backend API enhancements
In `orchestrator/api_gateway.py`:
- Added frontend-ready read endpoints:
  - `GET /v1/runs/{run_id}/episodes` (paginated episode list)
  - `GET /v1/runs/{run_id}/analytics` (JSON analytics for chart rendering)
  - `GET /v1/leaderboard` (ratings table for leaderboard)
- Existing endpoints retained:
  - `POST /v1/runs`
  - `GET /v1/runs/{run_id}`
  - `GET /v1/usage/me`
  - `GET /healthz`
  - observability endpoints

### Worker cloud-job mode
In `orchestrator/worker.py`:
- Added env-driven burst mode:
  - `WORKER_BURST=true` for Cloud Run Job execution
- Added configurable log level:
  - `WORKER_LOG_LEVEL`

This is critical because Cloud Run jobs should process queue work in burst mode and exit.

### GCP deployment and tests
Added:
- `infra/gcp/deploy_cloud_run.sh`
  - Builds image
  - Deploys API Cloud Run service
  - Creates/updates worker Cloud Run job
- `infra/gcp/README.md`
  - End-to-end setup and runbook
- `scripts/gcp_test_client.sh`
  - Smoke test: create run, execute worker, poll status, fetch analytics/leaderboard

### README updates
`README.md` now includes:
- new API endpoints for frontend data
- cloud runbook guidance
- Cloud Scheduler HTTP target details
- Cloud Shell smoke test commands

## 3) How architecture works now (Pi no longer required)

Runtime path in cloud:
1. Client calls Cloud Run API (`llm-arena-api`)
2. API writes run metadata to Cloud SQL and enqueues episode jobs in Redis
3. Cloud Scheduler triggers Cloud Run worker job every minute
4. Worker drains queue (burst), executes OpenSpiel episodes, persists results in Cloud SQL
5. Client polls status, episodes, analytics, leaderboard from API

This fully replaces the old requirement for always-on local/Pi worker+redis orchestration.

## 4) What to verify now in Console

- Cloud Run service `llm-arena-api`: latest revision healthy
- Cloud Run job `llm-arena-worker`: executes successfully
- Cloud Scheduler job `llm-arena-worker-minute`: enabled and succeeding
- Cloud SQL: instance healthy, backups enabled
- Redis: ready
- Secret Manager: latest versions present

## 5) Security + IAM hardening for product

Current setup works, but production should be tightened:

1. Create dedicated service accounts:
- `llm-arena-api-sa`
- `llm-arena-worker-sa`
- `llm-arena-scheduler-sa`

2. Use least-privilege roles:
- API SA:
  - Secret accessor (specific secrets only)
  - Cloud SQL Client (if using connector/auth proxy path)
- Worker SA:
  - Secret accessor (provider + DB)
  - Redis network access via VPC connector
- Scheduler SA:
  - Permission to run Cloud Run job (prefer scoped job-level permissions)

3. Restrict API ingress if needed:
- keep public only if product requires public access
- otherwise use auth gateways / IAP / identity-aware edge

4. Rotate secrets regularly:
- provider keys and API keys
- DB password rotation with coordinated secret updates

## 6) Observability/logging recommendations

Current logs are JSON and sufficient for debugging, but product needs:
- log-based metrics for:
  - API 5xx rate
  - queue lag
  - worker job failures
- Cloud Monitoring dashboard:
  - request latency p50/p95/p99
  - run completion latency
  - worker executions success/failure
- alerting policies:
  - API error spikes
  - worker execution failures
  - scheduler missed runs

## 7) Cost control recommendations

Primary baseline costs:
- Cloud SQL (always-on)
- Redis Memorystore (always-on)

Add immediately:
- Billing budget alerts (50/75/90/100%)
- Separate budget for this project
- periodic review of SQL tier / Redis size

## 8) Product-readiness checklist (remaining)

1. Frontend app:
- landing page + run creation UI
- status polling UI
- charts from `/v1/runs/{id}/analytics`
- leaderboard table from `/v1/leaderboard`

2. API hardening:
- structured API key management per customer
- per-customer quotas and usage endpoints
- stronger validation and better error contracts

3. Run policy:
- decide handling of `TRUNCATED` episodes
- SLA for run completion and retries

4. Testing:
- automated integration tests against cloud staging
- load tests for queue and API throughput

5. Data governance:
- retention policy for episodes/actions
- PII/security review if exposing user-submitted prompts/metadata

## 9) Important note about Security Command Center message

If Console says organization is required for Security Command Center, that is expected for many standalone projects.
You can still run safely without SCC by using:
- IAM least privilege
- Secret Manager
- Cloud Audit Logs
- Monitoring + alerting
- billing budgets

## 10) Exact Cloud Shell smoke flow

```bash
export API_URL="$(gcloud run services describe llm-arena-api --region us-central1 --format='value(status.url)')"
export API_KEY="$(gcloud secrets versions access latest --secret arena-api-keys | cut -d',' -f1)"

curl -sS "$API_URL/healthz"
curl -sS -X POST "$API_URL/v1/runs" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"run_type":"PVP","env":"connect_four_v1","num_episodes":1,"num_steps":10,"model_names":["GEMINI","GPT"]}'

gcloud run jobs execute llm-arena-worker --region us-central1 --wait
```

For chess, change `env` to `chess_v1`.

