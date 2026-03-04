#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL, e.g. https://llm-arena-api-xxxxx.a.run.app}"
: "${API_KEY:?Set API_KEY}"

RUN_TYPE="${RUN_TYPE:-PVP}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_STEPS="${NUM_STEPS:-null}"
MODEL_NAMES_JSON="${MODEL_NAMES_JSON:-[\"GEMINI\",\"GPT\"]}"
ENV_NAME="${ENV_NAME:-connect_four_v1}"
POLL_SECONDS="${POLL_SECONDS:-2}"
MAX_POLLS="${MAX_POLLS:-30}"

AUTH_HEADER="Authorization: Bearer ${API_KEY}"

CREATE_PAYLOAD=$(cat <<JSON
{"run_type":"${RUN_TYPE}","env":"${ENV_NAME}","num_episodes":${NUM_EPISODES},"num_steps":${NUM_STEPS},"model_names":${MODEL_NAMES_JSON}}
JSON
)

echo "[TEST] Creating run at ${API_URL}/v1/runs"
CREATE_JSON=$(curl -sS -X POST "${API_URL}/v1/runs" \
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
  STATUS_JSON=$(curl -sS "${API_URL}/v1/runs/${RUN_ID}" -H "${AUTH_HEADER}")
  STATUS=$(echo "$STATUS_JSON" | jq -r '.status // "UNKNOWN"')
  EPISODES_SAVED=$(echo "$STATUS_JSON" | jq -r '.episodes_saved // 0')
  echo "[TEST] poll=${i} status=${STATUS} episodes_saved=${EPISODES_SAVED}"
  if [[ "${STATUS}" == "COMPLETED" || "${STATUS}" == "FAILED" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[TEST] Fetching episodes:"
curl -sS "${API_URL}/v1/runs/${RUN_ID}/episodes?limit=50&offset=0" -H "${AUTH_HEADER}" | jq .

echo "[TEST] Fetching analytics:"
curl -sS "${API_URL}/v1/runs/${RUN_ID}/analytics?action_space=7" -H "${AUTH_HEADER}" | jq .

echo "[TEST] Fetching leaderboard:"
curl -sS "${API_URL}/v1/leaderboard?game=connect_four&limit=10" -H "${AUTH_HEADER}" | jq .
