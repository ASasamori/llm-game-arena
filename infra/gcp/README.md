# GCP Deployment (Cloud Run + Cloud SQL + Memorystore)

This setup keeps your current architecture:

- API Gateway: Cloud Run service (`orchestrator.api_gateway`)
- Worker: Cloud Run Job (`orchestrator.worker` in burst mode)
- Queue: Memorystore for Redis
- Database: Cloud SQL for PostgreSQL

## 1) Prerequisites

- `gcloud` authenticated to your project
- Billing enabled
- APIs enabled:
  - `run.googleapis.com`
  - `sqladmin.googleapis.com`
  - `redis.googleapis.com`
  - `vpcaccess.googleapis.com`
  - `secretmanager.googleapis.com`
  - `artifactregistry.googleapis.com`
  - `cloudbuild.googleapis.com`

## 2) Build and deploy

Use the helper script:

```bash
bash infra/gcp/deploy_cloud_run.sh
```

Required env vars:

```bash
export GCP_PROJECT_ID="<project-id>"
export GCP_REGION="us-central1"
export ARTIFACT_REPO="llm-arena"
export IMAGE_NAME="llm-arena"
export CLOUDRUN_API_SERVICE="llm-arena-api"
export CLOUDRUN_WORKER_JOB="llm-arena-worker"
export VPC_CONNECTOR="llm-arena-connector"
export REDIS_HOST="<memorystore-ip>"
export REDIS_PORT="6379"

# Secret names in Secret Manager (latest version used)
export DATABASE_URL_SECRET="arena-database-url" # pragma: allowlist secret -- Secret Manager resource name only
export API_KEYS_SECRET="arena-api-keys" # pragma: allowlist secret -- Secret Manager resource name only
export OPENAI_API_KEY_SECRET="arena-openai-key" # pragma: allowlist secret -- Secret Manager resource name only
export GEMINI_API_KEY_SECRET="arena-gemini-key" # pragma: allowlist secret -- Secret Manager resource name only
export ANTHROPIC_API_KEY_SECRET="arena-anthropic-key" # pragma: allowlist secret -- Secret Manager resource name only
```

Optional:

```bash
export IMAGE_TAG="v0.1.0"
export WORKER_TASK_TIMEOUT_SECONDS="1800"
export WORKER_EXECUTION_MODE="job_burst" # or "service_long_lived"
export RUNNER_NUM_WORKERS="2"            # used when WORKER_EXECUTION_MODE=service_long_lived
export CLOUDRUN_WORKER_SERVICE="llm-arena-worker-svc" # optional override
```

## 3) Worker execution model

Default (`WORKER_EXECUTION_MODE=job_burst`): worker is a Cloud Run Job with:

- `WORKER_MODE=burst`
- Runs one queue drain pass and exits

In this mode, API run creation triggers worker execution on-demand (no scheduler required).
`POST /v1/runs` returns immediately, enqueues jobs to Redis, then requests:

`https://run.googleapis.com/v2/projects/$GCP_PROJECT_ID/locations/$GCP_REGION/jobs/$CLOUDRUN_WORKER_JOB:run`

Run manually (optional):

```bash
gcloud run jobs execute "$CLOUDRUN_WORKER_JOB" --region "$GCP_REGION" --wait
```

Long-lived option (`WORKER_EXECUTION_MODE=service_long_lived`):

- Deploys a dedicated Cloud Run Service running `python -m orchestrator.runner`
- Uses `RUNNER_NUM_WORKERS` threads in-process
- Keeps workers online continuously via `min-instances`

Required IAM for API service account:

- `roles/run.invoker` on job `$CLOUDRUN_WORKER_JOB`
- `roles/run.developer` (or equivalent permission to `run.jobs.run`)

## 4) Smoke test from Cloud Shell

```bash
export API_URL="$(gcloud run services describe "$CLOUDRUN_API_SERVICE" --region "$GCP_REGION" --format='value(status.url)')"
export API_KEY="$(gcloud secrets versions access latest --secret "$API_KEYS_SECRET" | cut -d',' -f1)"

bash scripts/gcp_test_client.sh
```

You can also run `scripts/gcp_test_client.sh` without setting `API_URL` as long as `gcloud` is installed and authenticated; it auto-resolves:

```bash
export CLOUDRUN_API_SERVICE="llm-arena-api"
export GCP_REGION="us-central1"
export API_KEY="$(gcloud secrets versions access latest --secret "$API_KEYS_SECRET" | cut -d',' -f1)"
bash scripts/gcp_test_client.sh
```

For non-GCP machines (no `gcloud`), set `API_URL` and `API_KEY` manually and run the same script.

The script will:

1. Create a run
2. Trigger the worker job if configured
3. Poll run status
4. Fetch episodes, analytics JSON, and leaderboard JSON
