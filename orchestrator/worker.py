import redis
from rq import SimpleWorker, Queue
from termcolor import colored

redis_conn = redis.Redis(host="localhost", port=6379)

if __name__ == "__main__":
    queue = Queue("episodes", connection=redis_conn)
    worker = SimpleWorker([queue], connection=redis_conn)

    print(colored("[WORKER] Starting worker...", "green"))
    print(colored("[WORKER] Listening on 'episodes' queue", "cyan"))
    print(colored("[WORKER] Press Ctrl+C to stop", "yellow") + "\n")

    worker.work()
