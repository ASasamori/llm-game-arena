"""
Deterministic mock E2E pipeline:
API request -> run completion -> report + observability export.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from redis import Redis
from rq import Queue, SimpleWorker
from rq.registry import DeferredJobRegistry, StartedJobRegistry

os.environ.setdefault("MODEL_API_MOCK_MODE", "true")
os.environ.setdefault("MODEL_API_MOCK_CASE", "default")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from orchestrator.api_gateway import app
from orchestrator.models import Episode, Run, get_session
from orchestrator.observability import export_dashboard_view, export_metrics_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic mock E2E pipeline.")
    parser.add_argument(
        "--api-key", default="local-dev-key", help="API key for local gateway auth."
    )
    parser.add_argument(
        "--redis-host", default=os.environ.get("REDIS_HOST", "localhost")
    )
    parser.add_argument(
        "--redis-port", type=int, default=int(os.environ.get("REDIS_PORT", "6379"))
    )
    parser.add_argument("--queue-name", default="episodes")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", default="analytics/mock_e2e")
    return parser.parse_args()


def _build_deterministic_request() -> dict[str, Any]:
    # INPUT mode executes one model turn then one fixed input turn repeatedly.
    # With model mock case "default" (column 3), this sequence reaches a deterministic terminal win.
    return {
        "run_type": "INPUT",
        "env": "connect_four_v1",
        "num_episodes": 1,
        "num_steps": None,
        "model_names": ["GPT"],
        "input_actions": [0, 0, 0],
    }


def _queue_is_idle(queue: Queue) -> bool:
    deferred = DeferredJobRegistry(queue=queue)
    started = StartedJobRegistry(queue=queue)
    return queue.count == 0 and deferred.count == 0 and started.count == 0


def _drain_queue_once(queue: Queue, redis_conn: Redis) -> None:
    worker = SimpleWorker([queue], connection=redis_conn)
    worker.work(burst=True, logging_level="WARNING")


def _fetch_run_report(run_id: str) -> dict[str, Any]:
    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        episodes = (
            session.query(Episode)
            .filter(Episode.run_id == run_id)
            .order_by(Episode.start_time.asc())
            .all()
        )
        if run is None:
            raise RuntimeError(f"Run {run_id} not found in DB.")
        return {
            "run": {
                "run_id": run.run_id,
                "run_type": run.run_type,
                "status": run.status,
                "num_episodes": run.num_episodes,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            },
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "status": episode.status,
                    "winner": episode.winner,
                    "num_steps": episode.num_steps,
                    "agent": episode.agent,
                    "error": episode.error,
                    "action_count": len(episode.action)
                    if isinstance(episode.action, list)
                    else 0,
                    "actions": episode.action,
                    "start_time": episode.start_time.isoformat()
                    if episode.start_time
                    else None,
                    "end_time": episode.end_time.isoformat()
                    if episode.end_time
                    else None,
                }
                for episode in episodes
            ],
        }
    finally:
        session.close()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    redis_conn = Redis(host=args.redis_host, port=args.redis_port)
    queue = Queue(args.queue_name, connection=redis_conn)

    payload = _build_deterministic_request()
    headers = {"Authorization": f"Bearer {args.api_key}"}
    status_history: list[dict[str, Any]] = []

    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json=payload, headers=headers)
        if create_response.status_code != 202:
            raise RuntimeError(
                f"Run creation failed ({create_response.status_code}): {create_response.text}"
            )
        create_json = create_response.json()
        run_id = create_json["run_id"]

        deadline = time.time() + args.timeout_seconds
        final_status = "PENDING"
        while time.time() < deadline:
            if not _queue_is_idle(queue):
                _drain_queue_once(queue, redis_conn)

            status_response = client.get(f"/v1/runs/{run_id}", headers=headers)
            if status_response.status_code != 200:
                raise RuntimeError(
                    f"Status polling failed ({status_response.status_code}): {status_response.text}"
                )
            status_json = status_response.json()
            final_status = status_json["status"]
            status_history.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "status": final_status,
                    "episodes_saved": status_json.get("episodes_saved", 0),
                }
            )
            if final_status in {"COMPLETED", "FAILED"} and _queue_is_idle(queue):
                break
            time.sleep(args.poll_interval_seconds)

        if final_status not in {"COMPLETED", "FAILED"}:
            raise TimeoutError(
                f"Run {run_id} did not reach terminal status within {args.timeout_seconds}s."
            )

        db_report = _fetch_run_report(run_id)
        report = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "mock_mode": os.environ.get("MODEL_API_MOCK_MODE"),
            "mock_case": os.environ.get("MODEL_API_MOCK_CASE"),
            "request_payload": payload,
            "create_response": create_json,
            "status_history": status_history,
            "final_status": final_status,
            "db_report": db_report,
        }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = export_metrics_snapshot(
        str(output_dir / "observability_metrics.json")
    )
    dashboard_path = export_dashboard_view(
        str(output_dir / "observability_dashboard.json")
    )

    print(f"Run ID: {report['create_response']['run_id']}")
    print(f"Final status: {report['final_status']}")
    print(f"Report: {report_path.resolve()}")
    print(f"Observability metrics: {Path(metrics_path).resolve()}")
    print(f"Observability dashboard: {Path(dashboard_path).resolve()}")


if __name__ == "__main__":
    main()
