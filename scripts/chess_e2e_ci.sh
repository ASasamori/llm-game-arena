#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-logs/ci_e2e}"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export ARENA_API_KEYS="${ARENA_API_KEYS:-local-dev-key}"
export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export API_HOST="${API_HOST:-127.0.0.1}"
export API_PORT="${API_PORT:-8000}"
export MODEL_API_MOCK_MODE="${MODEL_API_MOCK_MODE:-true}"
export MODEL_API_MOCK_CASE="${MODEL_API_MOCK_CASE:-chess_e4}"
export WORKER_MODE="${WORKER_MODE:-service_long_lived}"
export WORKER_LOG_LEVEL="${WORKER_LOG_LEVEL:-INFO}"

API_LOG="${LOG_DIR}/api.log"
WORKER_LOG="${LOG_DIR}/worker.log"
RUN_FILE="${LOG_DIR}/run_id.txt"

API_PID=""
WORKER_PID=""

cleanup() {
  set +e
  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" >/dev/null 2>&1; then
    kill "${WORKER_PID}" >/dev/null 2>&1
    wait "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[CI-E2E] Reinitializing Postgres schema..."
python - <<'PY'
from orchestrator.models import Base, engine
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print("[CI-E2E] DB schema dropped and recreated")
PY

echo "[CI-E2E] Starting API gateway..."
python -m orchestrator.api_gateway >"${API_LOG}" 2>&1 &
API_PID=$!

echo "[CI-E2E] Starting worker..."
python -m orchestrator.worker >"${WORKER_LOG}" 2>&1 &
WORKER_PID=$!

echo "[CI-E2E] Waiting for API readiness..."
python - <<'PY'
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

host = os.environ.get("API_HOST", "127.0.0.1")
port = os.environ.get("API_PORT", "8000")
url = f"http://{host}:{port}/healthz"

deadline = time.time() + 60
while time.time() < deadline:
    try:
        with urlopen(url, timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("status") == "ok":
                print("[CI-E2E] API is healthy")
                raise SystemExit(0)
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        pass
    time.sleep(1)

raise SystemExit("API did not become healthy within 60 seconds")
PY

echo "[CI-E2E] Submitting chess run..."
python - <<'PY'
import json
import os
from urllib.request import Request, urlopen

host = os.environ.get("API_HOST", "127.0.0.1")
port = os.environ.get("API_PORT", "8000")
api_key = os.environ.get("ARENA_API_KEYS", "local-dev-key").split(",")[0].strip()
url = f"http://{host}:{port}/v1/runs"
payload = {
    "run_type": "RAND",
    "env": "chess_v1",
    "num_episodes": 1,
    "num_steps": None,
    "model_names": ["GPT"],
}
req = Request(
    url=url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urlopen(req, timeout=10) as resp:
    body = json.loads(resp.read().decode("utf-8"))
run_id = body["run_id"]
print(f"[CI-E2E] Submitted run_id={run_id}")
with open(os.path.join("logs", "ci_e2e", "run_id.txt"), "w", encoding="utf-8") as f:
    f.write(run_id)
PY

export RUN_ID="$(cat "${RUN_FILE}")"
echo "[CI-E2E] Polling run status for ${RUN_ID}..."
python - <<'PY'
import json
import os
import time
from urllib.request import Request, urlopen

host = os.environ.get("API_HOST", "127.0.0.1")
port = os.environ.get("API_PORT", "8000")
run_id = os.environ["RUN_ID"]
api_key = os.environ.get("ARENA_API_KEYS", "local-dev-key").split(",")[0].strip()
url = f"http://{host}:{port}/v1/runs/{run_id}"

deadline = time.time() + 90
last_status = None
while time.time() < deadline:
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    last_status = body["status"]
    if last_status in {"COMPLETED", "FAILED"}:
        print(f"[CI-E2E] Run terminal status={last_status}")
        if last_status == "FAILED":
            raise SystemExit("Run ended in FAILED status")
        raise SystemExit(0)
    time.sleep(1)

raise SystemExit(f"Run did not finish in time; last_status={last_status}")
PY

echo "[CI-E2E] Verifying persisted episode game..."
python - <<'PY'
import os
from orchestrator.models import Episode, get_session

run_id = os.environ["RUN_ID"]
session = get_session()
try:
    episodes = session.query(Episode).filter(Episode.run_id == run_id).all()
    if not episodes:
        raise SystemExit("No episodes were persisted for run")
    for ep in episodes:
        if ep.game != "chess":
            raise SystemExit(f"Expected game=chess, got game={ep.game}")
        if ep.status == "FAILED":
            raise SystemExit(f"Episode {ep.episode_id} failed")
    print(f"[CI-E2E] Verified {len(episodes)} episode(s) with game=chess")
finally:
    session.close()
PY

echo "[CI-E2E] Checking worker log for hard errors..."
if grep -E "Traceback|\\bERROR\\b|episode_execution_failed|episode_persist_failed|Worker crashed" "${WORKER_LOG}"; then
  echo "[CI-E2E] Worker log contains error signatures"
  exit 1
fi

echo "[CI-E2E] Chess E2E completed successfully without worker log errors."
