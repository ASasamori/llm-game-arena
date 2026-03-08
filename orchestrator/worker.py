import redis
import os
from rq import SimpleWorker, Queue
from termcolor import colored
from orchestrator.models import ensure_tables_created

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
WORKER_MODE = os.environ.get("WORKER_MODE", "").strip().lower()
WORKER_BURST = os.environ.get("WORKER_BURST", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if WORKER_MODE in {"burst", "one_shot", "one-shot"}:
    WORKER_BURST = True
elif WORKER_MODE in {"long_lived", "long-lived", "service"}:
    WORKER_BURST = False

WORKER_LOG_LEVEL = os.environ.get("WORKER_LOG_LEVEL", "INFO").upper()

redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

if __name__ == "__main__":
    ensure_tables_created()
    queue = Queue("episodes", connection=redis_conn)
    worker = SimpleWorker([queue], connection=redis_conn)

    print(colored("[WORKER] Starting worker...", "green"))
    print(colored("[WORKER] Listening on 'episodes' queue", "cyan"))
    print(colored("[WORKER] Press Ctrl+C to stop", "yellow") + "\n")
    print(
        colored(
            f"[WORKER] mode={'burst' if WORKER_BURST else 'long_lived'} "
            f"burst={WORKER_BURST} log_level={WORKER_LOG_LEVEL}",
            "cyan",
        )
    )

    worker.work(burst=WORKER_BURST, logging_level=WORKER_LOG_LEVEL)
