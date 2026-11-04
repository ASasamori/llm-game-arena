"""Exercise the real API and four runner workers using credential-free mock models.

Provide DATABASE_URL, REDIS_HOST, and REDIS_PORT for disposable test services.
The script adds fresh runs and never drops tables, deletes data, or flushes Redis.
An occupied episodes queue is rejected before starting workers. Logs and replay
verification evidence are written under $LOG_DIR/generic (logs/ci_e2e by default).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Each entry supplies the expected total, including multiplayer combinations.
CASES: tuple[tuple[dict[str, Any], int], ...] = (
    (
        {
            "run_type": "PVP_RR",
            "env": "matching_pennies_3p",
            "num_episodes": 12,
            "num_steps": 100,
            "model_names": ["MOCK:a", "MOCK:b", "MOCK:c", "MOCK:d"],
            "seed": 123,
        },
        48,
    ),
    (
        {
            "run_type": "RAND",
            "env": "kuhn_poker(players=3)",
            "num_episodes": 12,
            "num_steps": 100,
            "model_names": ["MOCK:a"],
            "seed": 456,
        },
        12,
    ),
    (
        {
            "run_type": "PVP",
            "env": "catch",
            "num_episodes": 4,
            "num_steps": 100,
            "model_names": ["MOCK:a"],
            "seed": 789,
        },
        4,
    ),
    (
        {
            "run_type": "PVP",
            "env": "mfg_crowd_modelling(horizon=2)",
            "num_episodes": 4,
            "num_steps": 100,
            "model_names": ["MOCK:a"],
            "seed": 101,
            "mean_field_distribution": "uniform",
        },
        4,
    ),
    (
        {
            "run_type": "PVP",
            "env": "tic_tac_toe",
            "num_episodes": 12,
            "num_steps": 100,
            "model_names": ["MOCK:a", "MOCK:b"],
            "seed": 222,
        },
        12,
    ),
)


def configure_environment(output_dir: Path) -> tuple[str, str]:
    for name in ("DATABASE_URL", "REDIS_HOST", "REDIS_PORT"):
        if not os.environ.get(name):
            raise ValueError(f"{name} must explicitly point to the test services")
    redis_port = int(os.environ["REDIS_PORT"])
    if not 1 <= redis_port <= 65535:
        raise ValueError("REDIS_PORT must be between 1 and 65535")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        api_port = listener.getsockname()[1]
    api_key = f"generic-ci-{uuid.uuid4().hex}"
    # Export before importing any application modules or starting subprocesses.
    # Explicit dummy keys prevent dotenv from loading real provider credentials.
    os.environ.update(
        PYTHONUNBUFFERED="1",
        ARENA_API_KEYS=api_key,
        API_HOST="127.0.0.1",
        API_PORT=str(api_port),
        API_RATE_LIMIT_REQUESTS_PER_MINUTE="60000",
        API_RATE_LIMIT_TOKENS_PER_MINUTE="1000000",
        MODEL_API_MOCK_MODE="true",
        MODEL_API_MOCK_CASE="default",
        OPENAI_API_KEY="mock-not-a-real-key",
        GEMINI_API_KEY="mock-not-a-real-key",
        GOOGLE_API_KEY="mock-not-a-real-key",
        ANTHROPIC_API_KEY="mock-not-a-real-key",
        WORKER_TRIGGER_MODE="none",
        RUNNER_NUM_WORKERS="4",
        RUNNER_QUEUE_NAME="episodes",
        RUNNER_LOG_DIR=str(output_dir / "workers"),
        DB_AUTO_INIT_SCHEMA="true",
        SQL_ECHO="false",
        ORCH_MAX_JOBS_PER_RUN="1000",
    )
    return f"http://127.0.0.1:{api_port}", api_key


def prepare_services() -> None:
    # These imports must follow configure_environment: runtime modules bind
    # environment-driven settings when first imported.
    from redis import Redis
    from rq import Queue
    from rq.registry import StartedJobRegistry
    from orchestrator.models import ensure_tables_created

    connection = Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    connection.ping()
    queue = Queue("episodes", connection=connection)
    if queue.count or StartedJobRegistry(queue=queue).get_job_ids():
        raise RuntimeError(
            "Generic E2E requires an idle, disposable episodes queue; existing jobs were left untouched"
        )
    # Initialize once before concurrent API/runner startup; no schema drop.
    ensure_tables_created()


def request_api(
    base_url: str,
    api_key: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 10,
) -> dict[str, Any]:
    request = Request(
        base_url + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def assert_processes_alive(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(
                f"Service exited with code {process.returncode}; inspect the API/runner logs"
            )


def wait_for_api(
    base_url: str, api_key: str, processes: list[subprocess.Popen], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert_processes_alive(processes)
        try:
            if (
                request_api(base_url, api_key, "/healthz", timeout=2).get("status")
                == "ok"
            ):
                return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"API did not become ready within {timeout}s")


def wait_for_runs(
    base_url: str,
    api_key: str,
    run_ids: list[str],
    processes: list[subprocess.Popen],
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    snapshots: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        assert_processes_alive(processes)
        snapshots = [
            request_api(base_url, api_key, f"/v1/runs/{run_id}") for run_id in run_ids
        ]
        if all(item["status"] in {"COMPLETED", "FAILED"} for item in snapshots):
            if any(item["status"] != "COMPLETED" for item in snapshots):
                raise AssertionError(f"Runs ended unsuccessfully: {snapshots}")
            return snapshots
        time.sleep(0.25)
    raise TimeoutError(f"Runs did not finish within {timeout}s: {snapshots}")


def verify_runs(
    base_url: str, api_key: str, run_ids: list[str]
) -> list[dict[str, Any]]:
    from game_arena.openspiel.replay import replay_episode
    from orchestrator.models import (
        Episode,
        EpisodeTrace,
        EpisodeTransition,
        Matchup,
        MatchupProgress,
        RunProgress,
        get_session,
    )
    from orchestrator.tasks import run_episode_pvp

    report: list[dict[str, Any]] = []
    with get_session() as session:
        for run_id, (payload, expected) in zip(run_ids, CASES, strict=True):
            episodes = (
                session.query(Episode)
                .filter_by(run_id=run_id)
                .order_by(Episode.episode_id)
                .all()
            )
            progress = session.get(RunProgress, run_id)
            if (
                len(episodes) != expected
                or progress is None
                or progress.completed != expected
                or progress.failed != 0
            ):
                raise AssertionError(f"Incorrect episode/progress totals for {run_id}")
            for matchup in session.query(Matchup).filter_by(run_id=run_id):
                matchup_progress = session.get(MatchupProgress, matchup.matchup_id)
                if (
                    matchup_progress is None
                    or matchup_progress.completed != matchup.num_episodes
                    or matchup_progress.failed != 0
                ):
                    raise AssertionError(
                        f"Incorrect matchup counters for {matchup.matchup_id}"
                    )
            verified = 0
            for episode in episodes:
                trace = session.get(EpisodeTrace, episode.episode_id)
                if trace is None:
                    raise AssertionError(f"Missing trace for {episode.episode_id}")
                record = dict(trace.payload)
                record["steps"] = [
                    row.payload
                    for row in session.query(EpisodeTransition)
                    .filter_by(episode_id=episode.episode_id)
                    .order_by(EpisodeTransition.step_index)
                ]
                if not replay_episode(record)["verified"]:
                    raise AssertionError(f"Replay failed for {episode.episode_id}")
                verified += 1
            first = episodes[0]
            page = request_api(
                base_url, api_key, f"/v1/episodes/{first.episode_id}/trace?limit=1"
            )
            if len(page["transitions"]) != min(1, page["total"]):
                raise AssertionError(
                    "Trace pagination did not preserve the transition count"
                )
            if first.matchup_id:
                repeated = run_episode_pvp(
                    first.matchup_id, run_id, 0, 100, episode_id=first.episode_id
                )
                if not repeated.get("reused"):
                    raise AssertionError("Duplicate delivery was executed again")
                session.expire_all()
                after = session.get(RunProgress, run_id)
                if after is None or after.completed != expected or after.failed != 0:
                    raise AssertionError(
                        "Duplicate delivery changed completion counters"
                    )
            metrics = request_api(base_url, api_key, f"/v1/runs/{run_id}/metrics")
            if metrics["status_counts"] != {"SUCCESS": expected}:
                raise AssertionError(
                    f"Incorrect aggregate metrics for {run_id}: {metrics}"
                )
            decision_report = request_api(
                base_url,
                api_key,
                f"/v1/runs/{run_id}/report?episode_limit=1&step_limit=1",
            )
            if (
                decision_report["run"]["status"] != "COMPLETED"
                or decision_report["episodes"]["total"] != expected
                or len(decision_report["episodes"]["items"]) != 1
                or len(decision_report["transitions"]["items"]) != 1
            ):
                raise AssertionError(
                    "Decision report did not preserve bounded run evidence"
                )
            report_links = decision_report["links"]
            text_request = Request(
                report_links["text_report_url"],
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urlopen(text_request, timeout=10) as response:
                terminal_report = response.read().decode("utf-8")
                if response.headers.get_content_type() != "text/plain":
                    raise AssertionError("Terminal report has the wrong content type")
            if run_id not in terminal_report or not terminal_report.isascii():
                raise AssertionError(
                    "Terminal report is missing its run identity or is not ASCII"
                )
            with urlopen(report_links["report_url"], timeout=10) as response:
                shell = response.read().decode("utf-8")
                if (
                    response.headers.get_content_type() != "text/html"
                    or api_key in shell
                    or not response.headers.get("Content-Security-Policy")
                ):
                    raise AssertionError(
                        "Browser report shell is not safe to open without a key"
                    )
            report.append(
                {
                    "run_id": run_id,
                    "game": payload["env"],
                    "episodes": expected,
                    "replay_verified": verified,
                    "status": "COMPLETED",
                    "duplicate_checked": first.matchup_id is not None,
                    "decision_report_verified": True,
                }
            )
    return report


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--run-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.startup_timeout <= 0 or args.run_timeout <= 0:
        parser.error("timeouts must be positive")
    output_dir = (Path(os.environ.get("LOG_DIR", "logs/ci_e2e")) / "generic").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url, api_key = configure_environment(output_dir)
    prepare_services()
    processes: list[subprocess.Popen] = []
    with ExitStack() as stack:
        try:
            for module, name in (
                ("orchestrator.api_gateway", "api"),
                ("orchestrator.runner", "runner"),
            ):
                log = stack.enter_context(
                    (output_dir / f"{name}.log").open("w", encoding="utf-8")
                )
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-m", module],
                        cwd=PROJECT_ROOT,
                        env=os.environ.copy(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            wait_for_api(base_url, api_key, processes, args.startup_timeout)
            run_ids = [
                request_api(base_url, api_key, "/v1/runs", payload)["run_id"]
                for payload, _ in CASES
            ]
            wait_for_runs(base_url, api_key, run_ids, processes, args.run_timeout)
            runs = verify_runs(base_url, api_key, run_ids)
            report = {
                "schema_version": 1,
                "verified_at": datetime.now(UTC).isoformat(),
                "workers": 4,
                "episodes": sum(item["episodes"] for item in runs),
                "runs": runs,
            }
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(report))
            print(f"[GENERIC-E2E] Evidence: {output_dir}")
        finally:
            stop_processes(processes)


if __name__ == "__main__":
    main()
