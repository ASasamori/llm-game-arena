"""Arena entry point; original OpenSpiel commands remain compatible."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Game arena environments, experiments, replay, and database export"
    )
    parser.add_argument(
        "command",
        choices=["environments", "export-db", "catalog", "plan", "replay", "export"],
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    if args.command == "environments":
        from game_arena.environments.cli import main as command

        return command(remaining)
    if args.command == "export-db":
        from orchestrator.exports import main as command

        return command(remaining)
    from game_arena.openspiel.cli import main as command

    return command([args.command, *remaining])


raise SystemExit(main())
