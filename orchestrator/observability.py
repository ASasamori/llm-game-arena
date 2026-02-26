from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from pathlib import Path


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


def increment_counter(
    name: str, value: int = 1, tags: dict[str, Any] | None = None
) -> None:
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


def reset_metrics() -> None:
    with _METRICS_LOCK:
        _COUNTERS.clear()
        _TIMINGS_MS.clear()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    )


def get_metrics_snapshot() -> dict[str, Any]:
    with _METRICS_LOCK:
        counters = dict(_COUNTERS)
        timings = {key: list(values) for key, values in _TIMINGS_MS.items()}

    timing_stats: dict[str, dict[str, float | int]] = {}
    for key, values in timings.items():
        if not values:
            continue
        ordered = sorted(values)
        timing_stats[key] = {
            "count": len(values),
            "min_ms": round(ordered[0], 3),
            "max_ms": round(ordered[-1], 3),
            "avg_ms": round(sum(values) / len(values), 3),
            "p50_ms": round(_percentile(ordered, 50), 3),
            "p95_ms": round(_percentile(ordered, 95), 3),
        }

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "counter_count": len(counters),
        "timing_series_count": len(timings),
        "counters": counters,
        "timings": timing_stats,
    }


def build_dashboard_view(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = snapshot or get_metrics_snapshot()
    counters = snap.get("counters", {})
    timings = snap.get("timings", {})

    top_counters = sorted(
        (
            {"name": name, "value": value}
            for name, value in counters.items()
            if isinstance(value, int)
        ),
        key=lambda row: row["value"],
        reverse=True,
    )[:15]
    hottest_timings = sorted(
        (
            {"name": name, **stats}
            for name, stats in timings.items()
            if isinstance(stats, dict)
        ),
        key=lambda row: float(row.get("p95_ms", 0.0)),
        reverse=True,
    )[:10]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "counter_count": snap.get("counter_count", 0),
            "timing_series_count": snap.get("timing_series_count", 0),
            "total_counter_events": int(
                sum(v for v in counters.values() if isinstance(v, int))
            ),
        },
        "top_counters": top_counters,
        "hottest_timings": hottest_timings,
    }


def export_metrics_snapshot(output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = get_metrics_snapshot()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def export_dashboard_view(
    output_path: str, snapshot: dict[str, Any] | None = None
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard = build_dashboard_view(snapshot=snapshot)
    path.write_text(json.dumps(dashboard, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
