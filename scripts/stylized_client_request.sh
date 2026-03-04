#!/usr/bin/env bash
set -euo pipefail

: "${API_URL:?Set API_URL, e.g. http://localhost:8000}"
: "${API_KEY:?Set API_KEY}"

TITLE="${TITLE:-Arena Run Snapshot}"
SUMMARY="${SUMMARY:-Queue accepted your run and workers are processing episodes.}"
HIGHLIGHT_1="${HIGHLIGHT_1:-Run status: PENDING}"
HIGHLIGHT_2="${HIGHLIGHT_2:-Database target: raspberry_pi}"
HIGHLIGHT_3="${HIGHLIGHT_3:-Worker mode: long_lived}"
TONE="${TONE:-technical}"

AUTH_HEADER="Authorization: Bearer ${API_KEY}"

PAYLOAD=$(cat <<JSON
{
  "title": "${TITLE}",
  "summary": "${SUMMARY}",
  "highlights": ["${HIGHLIGHT_1}", "${HIGHLIGHT_2}", "${HIGHLIGHT_3}"],
  "tone": "${TONE}"
}
JSON
)

echo "[CLIENT] POST ${API_URL}/v1/client/stylize"
curl -sS -X POST "${API_URL}/v1/client/stylize" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" | jq .
