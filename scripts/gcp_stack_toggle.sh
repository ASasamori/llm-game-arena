#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "up" && "$MODE" != "down" ]]; then
  echo "Usage: $0 <up|down>"
  exit 1
fi

# Core project/deploy settings
GCP_PROJECT_ID="${GCP_PROJECT_ID:-gen-lang-client-0155805792}"
GCP_REGION="${GCP_REGION:-us-central1}"
ARTIFACT_REPO="${ARTIFACT_REPO:-llm-arena}"
IMAGE_NAME="${IMAGE_NAME:-llm-arena}"
CLOUDRUN_API_SERVICE="${CLOUDRUN_API_SERVICE:-llm-arena-api}"
CLOUDRUN_WORKER_JOB="${CLOUDRUN_WORKER_JOB:-llm-arena-worker}"

# Network + managed dependencies
VPC_CONNECTOR="${VPC_CONNECTOR:-llm-arena-connector}"
VPC_NETWORK="${VPC_NETWORK:-default}"
VPC_CONNECTOR_RANGE="${VPC_CONNECTOR_RANGE:-10.8.0.0/28}"
VPC_CONNECTOR_MACHINE_TYPE="${VPC_CONNECTOR_MACHINE_TYPE:-e2-micro}"
VPC_CONNECTOR_MIN_INSTANCES="${VPC_CONNECTOR_MIN_INSTANCES:-2}"
VPC_CONNECTOR_MAX_INSTANCES="${VPC_CONNECTOR_MAX_INSTANCES:-2}"

REDIS_INSTANCE="${REDIS_INSTANCE:-llm-arena-redis}"
REDIS_SIZE_GB="${REDIS_SIZE_GB:-1}"
REDIS_TIER="${REDIS_TIER:-basic}"
REDIS_PORT="${REDIS_PORT:-6379}"

SQL_INSTANCE="${SQL_INSTANCE:-llm-arena-pg}"
SQL_VERSION="${SQL_VERSION:-POSTGRES_16}"
SQL_TIER="${SQL_TIER:-db-custom-1-3840}"
SQL_DISK_GB="${SQL_DISK_GB:-10}"

# Secret names expected by deploy script
DATABASE_URL_SECRET="${DATABASE_URL_SECRET:-arena-database-url}"
API_KEYS_SECRET="${API_KEYS_SECRET:-arena-api-keys}"
OPENAI_API_KEY_SECRET="${OPENAI_API_KEY_SECRET:-arena-openai-key}"
GEMINI_API_KEY_SECRET="${GEMINI_API_KEY_SECRET:-arena-gemini-key}"
ANTHROPIC_API_KEY_SECRET="${ANTHROPIC_API_KEY_SECRET:-arena-anthropic-key}"

log() {
  printf "[gcp-toggle] %s\n" "$1"
}

require_secret() {
  local secret_name="$1"
  if ! gcloud secrets describe "$secret_name" >/dev/null 2>&1; then
    echo "Missing required secret: $secret_name"
    echo "Create it first: gcloud secrets create $secret_name"
    exit 1
  fi
}

gcloud config set project "$GCP_PROJECT_ID" >/dev/null

if [[ "$MODE" == "up" ]]; then
  log "Enabling required APIs"
  gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    vpcaccess.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com >/dev/null

  log "Checking required secrets"
  require_secret "$DATABASE_URL_SECRET"
  require_secret "$API_KEYS_SECRET"
  require_secret "$OPENAI_API_KEY_SECRET"
  require_secret "$GEMINI_API_KEY_SECRET"
  require_secret "$ANTHROPIC_API_KEY_SECRET"

  log "Creating/restoring Cloud SQL instance ($SQL_INSTANCE)"
  if gcloud sql instances describe "$SQL_INSTANCE" >/dev/null 2>&1; then
    gcloud sql instances patch "$SQL_INSTANCE" --activation-policy=ALWAYS --quiet
  else
    gcloud sql instances create "$SQL_INSTANCE" \
      --database-version="$SQL_VERSION" \
      --tier="$SQL_TIER" \
      --region="$GCP_REGION" \
      --storage-size="$SQL_DISK_GB" \
      --network="$VPC_NETWORK"
  fi

  log "Creating/restoring Memorystore Redis ($REDIS_INSTANCE)"
  if ! gcloud redis instances describe "$REDIS_INSTANCE" --region "$GCP_REGION" >/dev/null 2>&1; then
    gcloud redis instances create "$REDIS_INSTANCE" \
      --region "$GCP_REGION" \
      --size "$REDIS_SIZE_GB" \
      --tier "$REDIS_TIER" \
      --redis-version redis_7_0 \
      --network "$VPC_NETWORK"
  fi
  REDIS_HOST="$(gcloud redis instances describe "$REDIS_INSTANCE" --region "$GCP_REGION" --format='value(host)')"

  log "Creating/restoring VPC connector ($VPC_CONNECTOR)"
  if ! gcloud compute networks vpc-access connectors describe "$VPC_CONNECTOR" --region "$GCP_REGION" >/dev/null 2>&1; then
    gcloud compute networks vpc-access connectors create "$VPC_CONNECTOR" \
      --region "$GCP_REGION" \
      --network "$VPC_NETWORK" \
      --range "$VPC_CONNECTOR_RANGE" \
      --machine-type "$VPC_CONNECTOR_MACHINE_TYPE" \
      --min-instances "$VPC_CONNECTOR_MIN_INSTANCES" \
      --max-instances "$VPC_CONNECTOR_MAX_INSTANCES"
  fi

  log "Deploying API + worker job"
  export GCP_PROJECT_ID
  export GCP_REGION
  export ARTIFACT_REPO
  export IMAGE_NAME
  export CLOUDRUN_API_SERVICE
  export CLOUDRUN_WORKER_JOB
  export VPC_CONNECTOR
  export REDIS_HOST
  export REDIS_PORT
  export DATABASE_URL_SECRET
  export API_KEYS_SECRET
  export OPENAI_API_KEY_SECRET
  export GEMINI_API_KEY_SECRET
  export ANTHROPIC_API_KEY_SECRET
  export WORKER_EXECUTION_MODE="${WORKER_EXECUTION_MODE:-job_burst}"

  bash infra/gcp/deploy_cloud_run.sh

  API_URL="$(gcloud run services describe "$CLOUDRUN_API_SERVICE" --region "$GCP_REGION" --format='value(status.url)')"
  log "Stack is up."
  log "API URL: $API_URL"
  log "No Cloud Scheduler required; API triggers worker job on run creation."
  exit 0
fi

log "Turning stack down"

log "Deleting Cloud Run service and jobs"
gcloud run services delete "$CLOUDRUN_API_SERVICE" --region "$GCP_REGION" --quiet || true
gcloud run jobs delete "$CLOUDRUN_WORKER_JOB" --region "$GCP_REGION" --quiet || true
gcloud run jobs delete llm-arena-db-init --region "$GCP_REGION" --quiet || true

log "Deleting Memorystore and VPC connector"
gcloud redis instances delete "$REDIS_INSTANCE" --region "$GCP_REGION" --quiet || true
gcloud compute networks vpc-access connectors delete "$VPC_CONNECTOR" --region "$GCP_REGION" --quiet || true

log "Stopping Cloud SQL compute"
if gcloud sql instances describe "$SQL_INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances patch "$SQL_INSTANCE" --activation-policy=NEVER --quiet
fi

log "Stack is down (Cloud SQL may still incur storage charges)."
