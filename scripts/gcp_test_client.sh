#!/usr/bin/env bash
set -euo pipefail

: "${API_KEY:?Set API_KEY}"

CLOUDRUN_API_SERVICE="${CLOUDRUN_API_SERVICE:-llm-arena-api}"
GCP_REGION="${GCP_REGION:-us-central1}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-20}"

if [[ -z "${API_URL:-}" ]]; then
  if command -v gcloud >/dev/null 2>&1; then
    API_URL="$(gcloud run services describe "${CLOUDRUN_API_SERVICE}" --region "${GCP_REGION}" --format='value(status.url)')"
  else
    echo "[TEST] API_URL is required when gcloud is unavailable."
    exit 1
  fi
fi
API_URL="${API_URL%/}"

if ! command -v jq >/dev/null 2>&1; then
  echo "[TEST] jq is required for parsing JSON responses."
  exit 1
fi

RUN_TYPE="${RUN_TYPE:-PVP}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_STEPS="${NUM_STEPS:-null}"
MODEL_NAMES_JSON="${MODEL_NAMES_JSON:-[\"GEMINI\",\"GPT\"]}"
ENV_NAME="${ENV_NAME:-connect_four_v1}"
POLL_SECONDS="${POLL_SECONDS:-2}"
MAX_POLLS="${MAX_POLLS:-30}"

AUTH_HEADER="Authorization: Bearer ${API_KEY}"

echo "[TEST] API_URL=${API_URL}"
echo "[TEST] Authenticated preflight (/v1/usage/me)..."
HEALTH_BODY=$(curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  -H "${AUTH_HEADER}" \
  "${API_URL}/v1/usage/me")
HEALTH_STATUS=$(echo "${HEALTH_BODY}" | jq -r '.request_count // empty' 2>/dev/null || true)
if [[ -z "${HEALTH_STATUS}" ]]; then
  echo "[TEST] Preflight failed; expected JSON from /v1/usage/me."
  echo "[TEST] Raw response:"
  echo "${HEALTH_BODY}"
  exit 1
fi

CREATE_PAYLOAD=$(cat <<JSON
{"run_type":"${RUN_TYPE}","env":"${ENV_NAME}","num_episodes":${NUM_EPISODES},"num_steps":${NUM_STEPS},"model_names":${MODEL_NAMES_JSON}}
JSON
)

echo "[TEST] Creating run at ${API_URL}/v1/runs"
CREATE_JSON=$(curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  -X POST "${API_URL}/v1/runs" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "${CREATE_PAYLOAD}")

RUN_ID=$(echo "$CREATE_JSON" | jq -r '.run_id // empty')
if [[ -z "${RUN_ID}" ]]; then
  echo "[TEST] Failed to create run:"
  echo "$CREATE_JSON"
  exit 1
fi

echo "[TEST] run_id=${RUN_ID}"

if [[ -n "${CLOUDRUN_WORKER_JOB:-}" && -n "${GCP_REGION:-}" ]]; then
  echo "[TEST] Executing worker job ${CLOUDRUN_WORKER_JOB}"
  gcloud run jobs execute "${CLOUDRUN_WORKER_JOB}" --region "${GCP_REGION}" --wait >/dev/null
fi

STATUS="PENDING"
for ((i=1; i<=MAX_POLLS; i++)); do
  STATUS_JSON=$(curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
    "${API_URL}/v1/runs/${RUN_ID}" -H "${AUTH_HEADER}")
  STATUS=$(echo "$STATUS_JSON" | jq -r '.status // "UNKNOWN"')
  EPISODES_SAVED=$(echo "$STATUS_JSON" | jq -r '.episodes_saved // 0')
  echo "[TEST] poll=${i} status=${STATUS} episodes_saved=${EPISODES_SAVED}"
  if [[ "${STATUS}" == "COMPLETED" || "${STATUS}" == "FAILED" || "${STATUS}" == "SUCCESS" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[TEST] Fetching episodes:"
curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  "${API_URL}/v1/runs/${RUN_ID}/episodes?limit=50&offset=0" -H "${AUTH_HEADER}" | jq .

echo "[TEST] Fetching analytics:"
curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  "${API_URL}/v1/runs/${RUN_ID}/analytics?action_space=7" -H "${AUTH_HEADER}" | jq .

echo "[TEST] Fetching leaderboard:"
curl -sS --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  "${API_URL}/v1/leaderboard?game=connect_four&limit=10" -H "${AUTH_HEADER}" | jq .
