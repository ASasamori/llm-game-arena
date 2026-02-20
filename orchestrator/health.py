"""
Run/queue health checks for orchestration.

Examples:
1) Inspect all recent runs:
   uv run python -m orchestrator.health

2) Inspect one run in detail:
   uv run python -m orchestrator.health --run-id <RUN_ID>

3) Mark stale pending runs as FAILED (safe reconcile):
   uv run python -m orchestrator.health --reconcile-stale --stale-seconds 900
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

from redis import Redis
from rq import Queue
from rq.registry import (
    DeferredJobRegistry,
    FailedJobRegistry,
    FinishedJobRegistry,
    StartedJobRegistry,
)
from sqlalchemy import func

from orchestrator.models import Episode, Matchup, Run, get_session


@dataclass
class QueueSnapshot:
    queued: int
    started: int
    deferred: int
    failed: int
    finished: int


def _queue_snapshot(queue: Queue) -> QueueSnapshot:
    return QueueSnapshot(
        queued=len(queue.job_ids),
        started=len(StartedJobRegistry(queue=queue).get_job_ids()),
        deferred=len(DeferredJobRegistry(queue=queue).get_job_ids()),
        failed=len(FailedJobRegistry(queue=queue).get_job_ids()),
        finished=len(FinishedJobRegistry(queue=queue).get_job_ids()),
    )


def _run_jobs(queue: Queue, run_id: str) -> dict[str, list]:
    regs = {
        "queued": queue.jobs,
        "started": [
            queue.fetch_job(jid)
            for jid in StartedJobRegistry(queue=queue).get_job_ids()
        ],
        "deferred": [
            queue.fetch_job(jid)
            for jid in DeferredJobRegistry(queue=queue).get_job_ids()
        ],
        "failed": [
            queue.fetch_job(jid) for jid in FailedJobRegistry(queue=queue).get_job_ids()
        ],
    }

    filtered: dict[str, list] = {
        "queued": [],
        "started": [],
        "deferred": [],
        "failed": [],
    }
    for key, jobs in regs.items():
        for job in jobs:
            if job is None:
                continue
            if (job.kwargs or {}).get("run_id") == run_id:
                filtered[key].append(job)
    return filtered


def _print_run_status(queue: Queue, run_id: str, stale_seconds: int) -> None:
    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            print(f"[HEALTH] Run {run_id} not found")
            return

        episode_count = (
            session.query(func.count(Episode.episode_id))
            .filter(Episode.run_id == run_id)
            .scalar()
        )
        matchup_count = (
            session.query(func.count(Matchup.matchup_id))
            .filter(Matchup.run_id == run_id)
            .scalar()
        )

        print(
            f"[HEALTH] run={run.run_id} type={run.run_type} status={run.status} "
            f"episodes_saved={episode_count}/{run.num_episodes} matchups={matchup_count}"
        )

        if run.run_type in {"PVP", "PVP_RR"}:
            matchups = session.query(Matchup).filter(Matchup.run_id == run_id).all()
            for m in matchups:
                m_count = (
                    session.query(func.count(Episode.episode_id))
                    .filter(Episode.matchup_id == m.matchup_id)
                    .scalar()
                )
                print(
                    f"  matchup={m.matchup_id[:8]} status={m.status} "
                    f"episodes_saved={m_count}/{m.num_episodes} models={m.models}"
                )
    finally:
        session.close()

    job_map = _run_jobs(queue, run_id)
    print(
        f"[HEALTH] jobs queued={len(job_map['queued'])} started={len(job_map['started'])} "
        f"deferred={len(job_map['deferred'])} failed={len(job_map['failed'])}"
    )

    now_utc = datetime.now(timezone.utc)
    for job in job_map["started"]:
        started_at = job.started_at
        elapsed = int((now_utc - started_at).total_seconds()) if started_at else -1
        stale_mark = " STALE" if elapsed >= stale_seconds else ""
        print(
            f"  started job={job.id} elapsed={elapsed}s{stale_mark} kwargs={job.kwargs}"
        )


def _reconcile_stale_runs(queue: Queue, stale_seconds: int) -> None:
    session = get_session()
    try:
        pending_runs = (
            session.query(Run)
            .filter(
                Run.status == "PENDING",
                Run.run_type.in_(["PVP", "PVP_RR", "RAND", "INPUT"]),
            )
            .all()
        )

        now_utc = datetime.now(timezone.utc)
        updated = 0

        for run in pending_runs:
            jobs = _run_jobs(queue, run.run_id)
            episode_count = (
                session.query(func.count(Episode.episode_id))
                .filter(Episode.run_id == run.run_id)
                .scalar()
            )

            # Still making progress or currently runnable.
            if jobs["queued"] or jobs["deferred"]:
                continue

            # If there are active started jobs, only mark stale when all are too old.
            if jobs["started"]:
                all_stale = True
                for job in jobs["started"]:
                    if job.started_at is None:
                        all_stale = False
                        break
                    elapsed = (now_utc - job.started_at).total_seconds()
                    if elapsed < stale_seconds:
                        all_stale = False
                        break
                if not all_stale:
                    continue

            # No runnable jobs left and run isn't complete.
            if episode_count < run.num_episodes:
                run.status = "FAILED"
                updated += 1
                print(
                    f"[HEALTH] Reconciled stale run to FAILED: run={run.run_id} "
                    f"episodes_saved={episode_count}/{run.num_episodes}"
                )

        if updated:
            session.commit()
        else:
            session.rollback()
            print("[HEALTH] No stale runs needed reconciliation")
    finally:
        session.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect/reconcile orchestration health"
    )
    parser.add_argument("--run-id", default=None, help="Optional run_id to inspect")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--queue", default="episodes")
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument(
        "--reconcile-stale",
        action="store_true",
        help="Mark stale pending runs as FAILED when no runnable jobs remain",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    redis_conn = Redis(host=args.redis_host, port=args.redis_port)
    queue = Queue(args.queue, connection=redis_conn)

    snap = _queue_snapshot(queue)
    print(
        f"[HEALTH] queue={args.queue} queued={snap.queued} started={snap.started} "
        f"deferred={snap.deferred} failed={snap.failed} finished={snap.finished}"
    )

    if args.run_id:
        _print_run_status(queue, args.run_id, args.stale_seconds)

    if args.reconcile_stale:
        _reconcile_stale_runs(queue, args.stale_seconds)


if __name__ == "__main__":
    main()
