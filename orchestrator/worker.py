import redis
import os
from rq import SimpleWorker, Queue
from termcolor import colored

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

if __name__ == "__main__":
    queue = Queue("episodes", connection=redis_conn)
    worker = SimpleWorker([queue], connection=redis_conn)

    print(colored("[WORKER] Starting worker...", "green"))
    print(colored("[WORKER] Listening on 'episodes' queue", "cyan"))
    print(colored("[WORKER] Press Ctrl+C to stop", "yellow") + "\n")

    worker.work()
