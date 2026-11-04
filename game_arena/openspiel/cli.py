"""Safe offline entry point: planning is the default; execution is mock-only."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from game_arena.openspiel.datasets import DatasetStore, canonical_json, export_dataset
from game_arena.openspiel.experiments import (
    ExperimentPlan,
    PlanConfig,
    execute_mock_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible OpenSpiel arena planning, mock execution, replay, and export"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    catalog = subparsers.add_parser(
        "catalog", help="Discover installed games or inspect a parameterized game"
    )
    catalog.add_argument("--inspect", metavar="GAME")
    plan = subparsers.add_parser(
        "plan",
        help="Create a dry-run plan; --execute-mock explicitly enables local mock execution",
    )
    plan.add_argument(
        "--games",
        nargs="+",
        required=True,
        help="OpenSpiel names/specifications; quote parameterized specifications",
    )
    plan.add_argument(
        "--models",
        nargs="+",
        default=["mock"],
        help="Mock competitor labels; one label enables selfplay for any player count",
    )
    plan.add_argument(
        "--episodes", type=int, default=1, help="Episodes per game/matchup"
    )
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--max-steps", type=int, default=1000)
    plan.add_argument("--no-seat-rotation", action="store_true")
    plan.add_argument(
        "--mean-field-distribution",
        choices=["uniform"],
        help="Explicit exogenous population policy for mean field games",
    )
    plan.add_argument("--mean-field-population", type=int, default=0)
    plan.add_argument(
        "--output",
        type=Path,
        help="Immutable dataset directory; existing matching episodes are resumed",
    )
    plan.add_argument(
        "--execute-mock",
        action="store_true",
        help="Execute only local MockModel policies, without provider clients or API keys",
    )
    plan.add_argument("--workers", type=int, default=1)
    plan.add_argument("--max-pending", type=int)
    plan.add_argument("--shard-index", type=int, default=0)
    plan.add_argument("--shard-count", type=int, default=1)
    replay = subparsers.add_parser(
        "replay",
        help="Replay an episode from recorded transitions without calling a model",
    )
    replay.add_argument("--dataset", type=Path, required=True)
    replay.add_argument("--episode", required=True)
    export = subparsers.add_parser(
        "export", help="Stream normalized episodes, steps, and decisions JSONL"
    )
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--uncompressed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result: Any
        if args.command == "catalog":
            from game_arena.openspiel.catalog import inspect_game, list_games

            result = inspect_game(args.inspect) if args.inspect else list_games()
        elif args.command == "plan":
            if args.execute_mock and args.output is None:
                parser.error("--execute-mock requires --output")
            if (
                args.workers < 1
                or args.shard_count < 1
                or not 0 <= args.shard_index < args.shard_count
            ):
                parser.error(
                    "workers must be positive and shard-index must be in [0, shard-count)"
                )
            config = PlanConfig(
                games=tuple(args.games),
                episodes_per_matchup=args.episodes,
                model_names=tuple(args.models),
                seed=args.seed,
                max_steps=args.max_steps,
                rotate_seats=not args.no_seat_rotation,
                mean_field_distribution=args.mean_field_distribution,
                mean_field_population=args.mean_field_population,
            )
            plan = ExperimentPlan(config)
            result = {
                "execution": "mock" if args.execute_mock else "dry_run",
                "experiment_id": plan.experiment_id,
                "total_episodes": plan.episode_count,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "preview": [
                    job.as_dict()
                    for job in islice(
                        plan.jobs(
                            shard_index=args.shard_index, shard_count=args.shard_count
                        ),
                        3,
                    )
                ],
            }
            if args.execute_mock:
                result["result"] = execute_mock_plan(
                    plan,
                    args.output,
                    workers=args.workers,
                    max_pending=args.max_pending,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                )
            elif args.output is not None:
                DatasetStore(args.output).initialize(plan.manifest)
            if args.output is not None:
                result["dataset"] = str(args.output)
        elif args.command == "replay":
            from game_arena.openspiel.replay import replay_episode

            result = replay_episode(
                DatasetStore(args.dataset).read_episode(args.episode)
            )
        else:
            result = export_dataset(
                args.dataset, args.output, compressed=not args.uncompressed
            )
        print(canonical_json(result))
        if args.command == "plan" and args.execute_mock and result["result"]["failed"]:
            return 1
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(
            canonical_json({"error": str(error), "type": type(error).__name__}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
