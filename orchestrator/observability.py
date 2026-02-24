from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for structured service logs."""

    def __init__(self, *, service: str):
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


def configure_json_logging(*, service: str, level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service=service))
    root.addHandler(handler)


_METRICS_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = defaultdict(int)
_TIMINGS_MS: dict[str, list[float]] = defaultdict(list)


def _metric_key(name: str, tags: dict[str, Any] | None = None) -> str:
    if not tags:
        return name
    serialized = ",".join(f"{k}={tags[k]}" for k in sorted(tags))
    return f"{name}|{serialized}"


def increment_counter(name: str, value: int = 1, tags: dict[str, Any] | None = None) -> None:
    key = _metric_key(name, tags)
    with _METRICS_LOCK:
        _COUNTERS[key] += value


def observe_timing_ms(
    name: str, duration_ms: float, tags: dict[str, Any] | None = None
) -> None:
    key = _metric_key(name, tags)
    with _METRICS_LOCK:
        _TIMINGS_MS[key].append(duration_ms)


def time_now_ms() -> float:
    return time.perf_counter() * 1000.0
