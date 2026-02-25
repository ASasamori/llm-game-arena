"""
CLI for exporting process-local observability artifacts.

Example:
  uv run python -m orchestrator.observability_export \
    --metrics-out analytics/observability/metrics.json \
    --dashboard-out analytics/observability/dashboard.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestrator.observability import (
    export_dashboard_view,
    export_metrics_snapshot,
    get_metrics_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export observability snapshots.")
    parser.add_argument(
        "--metrics-out",
        default="analytics/observability/metrics.json",
        help="Output path for the metrics snapshot JSON.",
    )
    parser.add_argument(
        "--dashboard-out",
        default="analytics/observability/dashboard.json",
        help="Output path for the dashboard JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    snapshot = get_metrics_snapshot()
    metrics_path = export_metrics_snapshot(args.metrics_out)
    dashboard_path = export_dashboard_view(args.dashboard_out, snapshot=snapshot)
    print(f"Exported metrics snapshot: {Path(metrics_path).resolve()}")
    print(f"Exported dashboard view:   {Path(dashboard_path).resolve()}")


if __name__ == "__main__":
    main()
