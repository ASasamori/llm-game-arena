#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:?Set GCP_REGION}"
: "${ARTIFACT_REPO:?Set ARTIFACT_REPO}"
: "${IMAGE_NAME:?Set IMAGE_NAME}"
: "${CLOUDRUN_API_SERVICE:?Set CLOUDRUN_API_SERVICE}"
: "${CLOUDRUN_WORKER_JOB:?Set CLOUDRUN_WORKER_JOB}"
: "${VPC_CONNECTOR:?Set VPC_CONNECTOR}"
: "${REDIS_HOST:?Set REDIS_HOST}"
: "${REDIS_PORT:?Set REDIS_PORT}"

: "${DATABASE_URL_SECRET:?Set DATABASE_URL_SECRET}"
: "${API_KEYS_SECRET:?Set API_KEYS_SECRET}"
: "${OPENAI_API_KEY_SECRET:?Set OPENAI_API_KEY_SECRET}"
: "${GEMINI_API_KEY_SECRET:?Set GEMINI_API_KEY_SECRET}"
: "${ANTHROPIC_API_KEY_SECRET:?Set ANTHROPIC_API_KEY_SECRET}"

IMAGE_TAG="${IMAGE_TAG:-latest}"
WORKER_TASK_TIMEOUT_SECONDS="${WORKER_TASK_TIMEOUT_SECONDS:-1800}"
WORKER_EXECUTION_MODE="${WORKER_EXECUTION_MODE:-job_burst}"
RUNNER_NUM_WORKERS="${RUNNER_NUM_WORKERS:-2}"
WORKER_SERVICE_MIN_INSTANCES="${WORKER_SERVICE_MIN_INSTANCES:-1}"
WORKER_SERVICE_MAX_INSTANCES="${WORKER_SERVICE_MAX_INSTANCES:-1}"
if [[ "${WORKER_EXECUTION_MODE}" == "service_long_lived" ]]; then
  CLOUDRUN_WORKER_SERVICE="${CLOUDRUN_WORKER_SERVICE:-${CLOUDRUN_WORKER_JOB}-svc}"
fi

gcloud config set project "$GCP_PROJECT_ID" >/dev/null

gcloud artifacts repositories create "$ARTIFACT_REPO" \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --description="LLM arena images" 2>/dev/null || true

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud builds submit --tag "$IMAGE_URI" .

gcloud run deploy "$CLOUDRUN_API_SERVICE" \
  --image "$IMAGE_URI" \
  --region "$GCP_REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --vpc-connector "$VPC_CONNECTOR" \
  --vpc-egress all-traffic \
  --set-env-vars "API_HOST=0.0.0.0,API_PORT=8000,REDIS_HOST=$REDIS_HOST,REDIS_PORT=$REDIS_PORT" \
  --set-secrets "DATABASE_URL=${DATABASE_URL_SECRET}:latest,ARENA_API_KEYS=${API_KEYS_SECRET}:latest,OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY_SECRET}:latest"

if [[ "${WORKER_EXECUTION_MODE}" == "service_long_lived" ]]; then
  gcloud run deploy "$CLOUDRUN_WORKER_SERVICE" \
    --image "$IMAGE_URI" \
    --region "$GCP_REGION" \
    --platform managed \
    --no-allow-unauthenticated \
    --vpc-connector "$VPC_CONNECTOR" \
    --vpc-egress all-traffic \
    --min-instances "$WORKER_SERVICE_MIN_INSTANCES" \
    --max-instances "$WORKER_SERVICE_MAX_INSTANCES" \
    --set-env-vars "REDIS_HOST=$REDIS_HOST,REDIS_PORT=$REDIS_PORT,RUNNER_NUM_WORKERS=$RUNNER_NUM_WORKERS,RUNNER_QUEUE_NAME=episodes" \
    --set-secrets "DATABASE_URL=${DATABASE_URL_SECRET}:latest,OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY_SECRET}:latest" \
    --command ".venv/bin/python" \
    --args "-m,orchestrator.runner"
else
  gcloud run jobs describe "$CLOUDRUN_WORKER_JOB" --region "$GCP_REGION" >/dev/null 2>&1 && {
    gcloud run jobs update "$CLOUDRUN_WORKER_JOB" \
      --image "$IMAGE_URI" \
      --region "$GCP_REGION" \
      --task-timeout "${WORKER_TASK_TIMEOUT_SECONDS}s" \
      --vpc-connector "$VPC_CONNECTOR" \
      --vpc-egress all-traffic \
      --set-env-vars "REDIS_HOST=$REDIS_HOST,REDIS_PORT=$REDIS_PORT,WORKER_MODE=burst,WORKER_LOG_LEVEL=INFO" \
      --set-secrets "DATABASE_URL=${DATABASE_URL_SECRET}:latest,OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY_SECRET}:latest" \
      --command ".venv/bin/python" \
      --args "-m,orchestrator.worker"
  } || {
    gcloud run jobs create "$CLOUDRUN_WORKER_JOB" \
      --image "$IMAGE_URI" \
      --region "$GCP_REGION" \
      --max-retries 0 \
      --tasks 1 \
      --task-timeout "${WORKER_TASK_TIMEOUT_SECONDS}s" \
      --vpc-connector "$VPC_CONNECTOR" \
      --vpc-egress all-traffic \
      --set-env-vars "REDIS_HOST=$REDIS_HOST,REDIS_PORT=$REDIS_PORT,WORKER_MODE=burst,WORKER_LOG_LEVEL=INFO" \
      --set-secrets "DATABASE_URL=${DATABASE_URL_SECRET}:latest,OPENAI_API_KEY=${OPENAI_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY_SECRET}:latest" \
      --command ".venv/bin/python" \
      --args "-m,orchestrator.worker"
  }
fi

echo "Deployed:"
echo "  API service: $CLOUDRUN_API_SERVICE"
if [[ "${WORKER_EXECUTION_MODE}" == "service_long_lived" ]]; then
  echo "  Worker svc:  $CLOUDRUN_WORKER_SERVICE"
else
  echo "  Worker job:  $CLOUDRUN_WORKER_JOB"
fi
echo "  Image:       $IMAGE_URI"
