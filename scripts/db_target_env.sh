#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: eval \"\$(scripts/db_target_env.sh <raspberry_pi|gcp>)\"" >&2
  exit 1
fi

case "${TARGET}" in
  raspberry_pi|pi|rpi)
    cat <<'EOF'
export DATABASE_TARGET=raspberry_pi
# Preferred: full URL (takes precedence if set)
# export DATABASE_URL_RASPBERRY_PI="postgresql+psycopg2://<user>:<pass>@100.116.74.50:5432/llm_game_arena"
export RASPBERRY_PI_PG_HOST=100.116.74.50
export RASPBERRY_PI_PG_PORT=5432
export RASPBERRY_PI_PG_DATABASE=llm_game_arena
EOF
    ;;
  gcp|google_cloud|google-cloud)
    cat <<'EOF'
export DATABASE_TARGET=gcp
# Preferred: full URL (takes precedence if set)
# export DATABASE_URL_GCP="postgresql+psycopg2://<user>:<pass>@<gcp-host>:5432/llm_game_arena"
# Optional build-from-parts fallback:
# export GCP_PG_HOST="<private-ip-or-hostname>"
# export GCP_PG_PORT=5432
# export GCP_PG_DATABASE=llm_game_arena
EOF
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Supported: raspberry_pi, gcp" >&2
    exit 1
    ;;
esac
