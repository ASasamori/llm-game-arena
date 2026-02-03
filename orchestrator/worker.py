import redis
from rq import SimpleWorker, Queue

redis_conn = redis.Redis(host="localhost", port=6379)

if __name__ == "__main__":
    queue = Queue("episodes", connection=redis_conn)
    worker = SimpleWorker([queue], connection=redis_conn)

    print("[WORKER] Starting worker...")
    print("[WORKER] Listening on 'episodes' queue")
    print("[WORKER] Press Ctrl+C to stop\n")

    worker.work()
