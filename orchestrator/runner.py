"""
Runner script that spawns multiple RQ workers using ThreadPoolExecutor.

Usage:
    python -m orchestrator_rq.runner           # Start 5 workers (default)
    python -m orchestrator_rq.runner --workers 3  # Start 3 workers

This replaces manually opening N terminal tabs for workers.
Now you only need 3 terminals total:
    1. redis-server
    2. python -m orchestrator_rq.runner (spawns all workers)
    3. python -m orchestrator_rq.jobs (enqueue runs)

Logs are written to logs/worker_0.log, logs/worker_1.log, etc.
"""

import redis
import signal
import logging
import argparse
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rq import SimpleWorker, Queue


class NoOpDeathPenalty:
    """Dummy death penalty that doesn't use signals (for threads)."""

    def __init__(self, timeout, exception, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        return False


class ThreadSafeWorker(SimpleWorker):
    """SimpleWorker that doesn't use signals (for use in threads)."""
    death_penalty_class = NoOpDeathPenalty

    def _install_signal_handlers(self):
        pass


# Shared state
shutdown_event = threading.Event()
workers = []
workers_lock = threading.Lock()


def setup_logger(worker_id: int, log_dir: Path) -> logging.Logger:
    """Create a logger that writes to a dedicated file for this worker."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_dir / f"worker_{worker_id}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

    return logger


def run_worker(worker_id: int, log_dir: Path, queue_name: str):
    """Run a single RQ worker in a thread."""
    logger = setup_logger(worker_id, log_dir)
    logger.info(f"Worker {worker_id} starting")
    worker = None

    try:
        redis_conn = redis.Redis(host="localhost", port=6379)
        queue = Queue(queue_name, connection=redis_conn)
        worker_name = f"worker_{worker_id}_{uuid.uuid4().hex[:8]}"
        worker = ThreadSafeWorker([queue], connection=redis_conn, name=worker_name)

        with workers_lock:
            workers.append(worker)

        logger.info(f"Listening on '{queue_name}' queue")

        # Process jobs until shutdown (work() blocks until stopped via signal)
        worker.work(logging_level="INFO")

        logger.info(f"Worker {worker_id} stopped")

    except Exception as e:
        logger.exception(f"Worker {worker_id} error: {e}")
    finally:
        # Deregister worker from Redis
        if worker:
            try:
                worker.register_death()
            except Exception:
                pass


shutdown_count = 0


def signal_handler(signum, frame):
    """Handle shutdown signals. First Ctrl+C = warm, second = force exit."""
    global shutdown_count
    shutdown_count += 1
    shutdown_event.set()

    if shutdown_count == 1:
        print(f"\n[RUNNER] Shutting down (Ctrl+C again to force)...")
        with workers_lock:
            for worker in workers:
                try:
                    worker.request_stop(signum, frame)
                except Exception:
                    pass
    else:
        print(f"\n[RUNNER] Force exit")
        # Deregister workers from Redis before exiting
        with workers_lock:
            for worker in workers:
                try:
                    worker.register_death()
                except Exception:
                    pass
        import os
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Run multiple RQ workers")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Number of workers (default: 5)")
    parser.add_argument("--log-dir", "-l", type=Path, default=Path("logs"), help="Log directory")
    parser.add_argument("--queue", "-q", type=str, default="episodes", help="Queue name")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[RUNNER] Starting {args.workers} workers on '{args.queue}' queue")
    print(f"[RUNNER] Logs: {args.log_dir.absolute()}/worker_N.log")
    print(f"[RUNNER] Press Ctrl+C to stop\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_worker, i, args.log_dir, args.queue)
            for i in range(args.workers)
        ]

        # Wait for all workers to complete
        for future in futures:
            future.result()

    print("[RUNNER] All workers stopped")


if __name__ == "__main__":
    main()
