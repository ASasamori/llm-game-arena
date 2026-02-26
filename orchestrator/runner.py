"""
Multi-threaded RQ SimpleWorker runner.

One Python process, N workers in threads, graceful Ctrl+C shutdown.
This is Phase 1 local infrastructure - disappears in production.
"""

import io
import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
import redis
from rq import SimpleWorker, Queue, Worker
from rq.registry import StartedJobRegistry
from rq.timeouts import TimerDeathPenalty
from termcolor import colored


# Configuration
NUM_WORKERS = int(os.environ.get("RUNNER_NUM_WORKERS", "1"))
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
QUEUE_NAME = os.environ.get("RUNNER_QUEUE_NAME", "episodes")
LOG_DIR = os.environ.get("RUNNER_LOG_DIR", "logs")
STALE_JOB_WARN_SECONDS = int(os.environ.get("RUNNER_STALE_JOB_WARN_SECONDS", "900"))
if NUM_WORKERS < 1:
    NUM_WORKERS = 1

# Thread-local storage for log files
_thread_log_files: dict[int | None, io.TextIOWrapper] = {}
_thread_log_lock = threading.Lock()


class ThreadFilter(logging.Filter):
    """Only allow log records from a specific thread."""

    def __init__(self, thread_id: int | None):
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record):
        return record.thread == self.thread_id


# TODO: Rework so that whenever this is called or when workers SHOULD die
# they delete the log files.
class ThreadLocalWriter:
    """
    A stdout/stderr replacement that routes output to thread-specific log files.
    Falls back to the original stream for the main thread.
    """

    def __init__(self, original_stream, stream_name: str = "stdout"):
        self._original = original_stream
        self._stream_name = stream_name
        self._main_thread_id = threading.main_thread().ident

    def write(self, text):
        thread_id = threading.current_thread().ident

        # Main thread writes to original stream
        if thread_id == self._main_thread_id:
            self._original.write(text)
            self._original.flush()
            return

        # Worker threads write to their log file
        with _thread_log_lock:
            log_file = _thread_log_files.get(thread_id)

        if log_file and not log_file.closed:
            try:
                log_file.write(text)
                log_file.flush()
            except (ValueError, IOError):
                pass  # File was closed
        else:
            # Fallback to original if no log file registered
            self._original.write(text)
            self._original.flush()

    def flush(self):
        thread_id = threading.current_thread().ident

        if thread_id == self._main_thread_id:
            self._original.flush()
            return

        with _thread_log_lock:
            log_file = _thread_log_files.get(thread_id)

        if log_file and not log_file.closed:
            try:
                log_file.flush()
            except (ValueError, IOError):
                pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        thread_id = threading.current_thread().ident
        if thread_id == self._main_thread_id:
            return self._original.isatty()
        return False


def _runner_msg(message: str, color: str = "white") -> str:
    return colored(f"[RUNNER] {message}", color)


class ThreadedWorker(SimpleWorker):
    """
    SimpleWorker subclass safe for running in threads.
    - Skips signal handler installation (only main thread can do that)
    - Uses timer-based timeouts instead of SIGALRM
    """

    death_penalty_class = TimerDeathPenalty

    def _install_signal_handlers(self):
        pass  # No-op: main thread handles signals


def worker_thread(worker: ThreadedWorker, worker_id: str, log_file_path: str):
    """Thread target: run the worker's blocking loop with redirected output."""
    thread_id = threading.current_thread().ident

    # Open log file for this thread
    log_file = open(log_file_path, "w", buffering=1)  # Line buffered

    # Register this thread's log file
    with _thread_log_lock:
        _thread_log_files[thread_id] = log_file

    # Create a handler that only accepts logs from THIS thread
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    handler.addFilter(ThreadFilter(thread_id))  # Only logs from this thread

    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        print(f"={'=' * 60}")
        print(f"Worker: {worker_id[:8]}")
        print(f"Queue: {QUEUE_NAME}")
        print(f"{'=' * 60}")

        worker.work(burst=False, logging_level="INFO")
    except Exception as e:
        print(f"Worker crashed: {e}")
    finally:
        print(f"{'=' * 60}")
        print(f"Worker {worker_id[:8]} stopped")
        print(f"{'=' * 60}")

        # Cleanup
        root_logger.removeHandler(handler)
        handler.close()

        with _thread_log_lock:
            _thread_log_files.pop(thread_id, None)

        log_file.close()


def requeue_orphaned_started_jobs(queue: Queue):
    """
    Requeue STARTED jobs owned by workers that are no longer active.
    This recovers jobs left behind after crashes/restarts.
    """
    started_registry = StartedJobRegistry(queue=queue)
    started_job_ids = started_registry.get_job_ids()
    if not started_job_ids:
        return

    now_utc = datetime.now(timezone.utc)
    active_worker_names = {
        worker.name
        for worker in Worker.all(connection=queue.connection)
        if worker.last_heartbeat is not None
        and (now_utc - worker.last_heartbeat).total_seconds() <= 120
    }
    requeued = 0
    skipped = 0
    print(
        _runner_msg(
            f"Found {len(started_job_ids)} STARTED jobs. Checking for orphaned jobs...",
            "cyan",
        )
    )
    for job_id in started_job_ids:
        job = queue.fetch_job(job_id)
        if job is None:
            continue
        if job.worker_name in active_worker_names:
            skipped += 1
            continue
        try:
            started_registry.requeue(job_id, at_front=True)
            requeued += 1
        except Exception as e:
            print(_runner_msg(f"Failed to requeue orphaned job {job_id}: {e}", "red"))
    print(
        _runner_msg(
            f"Requeued {requeued} orphaned STARTED jobs (active and skipped: {skipped}).",
            "green",
        )
    )


def report_stale_started_jobs(queue: Queue, stale_seconds: int) -> None:
    """
    Emit warning logs for STARTED jobs that have exceeded a runtime threshold.
    This does not requeue/kill jobs; it is an operator signal for potentially hung work.
    """
    started_registry = StartedJobRegistry(queue=queue)
    started_job_ids = started_registry.get_job_ids()
    if not started_job_ids:
        return

    now_utc = datetime.now(timezone.utc)
    stale_jobs = []
    for job_id in started_job_ids:
        job = queue.fetch_job(job_id)
        if job is None or job.started_at is None:
            continue
        elapsed = (now_utc - job.started_at).total_seconds()
        if elapsed >= stale_seconds:
            stale_jobs.append((job, int(elapsed)))

    if not stale_jobs:
        return

    print(
        _runner_msg(
            f"Detected {len(stale_jobs)} long-running STARTED job(s) "
            f"(>{stale_seconds}s).",
            "yellow",
        )
    )
    for job, elapsed in stale_jobs:
        run_id = (job.kwargs or {}).get("run_id")
        matchup_id = (job.kwargs or {}).get("matchup_id")
        print(
            _runner_msg(
                f"stale job={job.id} elapsed={elapsed}s run_id={run_id} "
                f"matchup_id={matchup_id} func={job.func_name}",
                "yellow",
            )
        )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Install thread-local stdout/stderr redirectors
    sys.stdout = ThreadLocalWriter(sys.__stdout__, "stdout")
    sys.stderr = ThreadLocalWriter(sys.__stderr__, "stderr")

    # Configure root logger to DEBUG so handlers receive everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove default handlers that would print to terminal
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    queue = Queue(QUEUE_NAME, connection=redis_conn)
    requeue_orphaned_started_jobs(queue)

    workers = []
    threads = []
    worker_ids = []

    for _ in range(NUM_WORKERS):
        worker_id = str(uuid.uuid4())
        worker_ids.append(worker_id)
        log_file_path = os.path.join(LOG_DIR, f"{worker_id}_worker.log")

        conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        worker = ThreadedWorker(
            [queue],
            connection=conn,
            name=f"worker-{worker_id}",
        )
        workers.append(worker)

        t = threading.Thread(
            target=worker_thread, args=(worker, worker_id, log_file_path), daemon=True
        )
        threads.append(t)

    # Start all threads
    for t in threads:
        t.start()

    print(_runner_msg(f"Started {NUM_WORKERS} workers", "green"))
    print(_runner_msg(f"Listening on '{QUEUE_NAME}' queue", "green"))
    print(_runner_msg("Worker logs (real-time):", "cyan"))
    for wid in worker_ids:
        print(f"         - {LOG_DIR}/{wid}_worker.log")
    print(_runner_msg(f"Watch all logs: tail -f {LOG_DIR}/*_worker.log", "cyan"))
    print(_runner_msg("Press Ctrl+C to stop", "yellow"))
    print("")

    # Signal handler for graceful shutdown
    shutdown_requested = threading.Event()

    def shutdown_handler(_signum, _frame):
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        print("\n" + _runner_msg("Shutdown signal received...", "yellow"))
        for w in workers:
            w._stop_requested = True
        print(_runner_msg("Waiting for workers to finish current jobs...", "yellow"))

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Block until shutdown
    try:
        last_recovery_ts = 0
        while not shutdown_requested.is_set():
            shutdown_requested.wait(timeout=1.0)
            now = int(time.time())
            if now - last_recovery_ts >= 60:
                requeue_orphaned_started_jobs(queue)
                report_stale_started_jobs(queue, STALE_JOB_WARN_SECONDS)
                last_recovery_ts = now
    except KeyboardInterrupt:
        pass

    for t in threads:
        t.join(timeout=1.0)

    alive = [t for t in threads if t.is_alive()]
    if alive:
        print(_runner_msg(f"{len(alive)} workers still busy, forcing exit...", "red"))
    else:
        print(_runner_msg("All workers stopped cleanly.", "green"))

    sys.exit(0)


if __name__ == "__main__":
    main()
