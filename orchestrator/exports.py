"""Consistent, streaming database exports for private collection and migration.

Each ZIP contains one gzip JSONL file per table, normalized applied decisions,
separate pending attempts, and a checksum manifest. No database URL, environment
credentials or API keys are added.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from uuid import UUID
import zipfile

from sqlalchemy import MetaData, create_engine, select, text
from sqlalchemy.orm import Session

from game_arena.openspiel.datasets import episode_pending_attempt_rows

ARENA_TABLES = (
    "runs",
    "run_configurations",
    "run_dispatches",
    "run_progress",
    "matchups",
    "matchup_progress",
    "episodes",
    "episode_steps",
    "episode_traces",
    "episode_transitions",
    "model_ratings",
    "rating_events",
)


class ExportNotFound(ValueError):
    pass


def _encode(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Unsupported database value: {type(value).__name__}")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(session: Session) -> str | None:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        session.execute(text("SET TRANSACTION READ ONLY"))
        return session.execute(text("SELECT pg_export_snapshot()")).scalar_one()
    if dialect == "sqlite":
        session.execute(text("BEGIN"))
        return None
    raise ValueError("Exports currently support PostgreSQL and SQLite")


def export_database(
    session: Session,
    output: Path,
    *,
    run_id: str | None = None,
    postgres_dump: bool = False,
    pg_dump_binary: str = "pg_dump",
) -> dict[str, Any]:
    """Export a fresh session's consistent snapshot without changing database rows.

    Existing files are never overwritten. The caller owns session cleanup. Rows
    and decisions are streamed; total dataset size does not determine RAM use.
    """
    if run_id is not None:
        UUID(run_id)
    if output.exists():
        raise FileExistsError(f"Export already exists: {output}")
    if postgres_dump and run_id is not None:
        raise ValueError("PostgreSQL backups require a whole-database export")
    if postgres_dump and session.get_bind().dialect.name != "postgresql":
        raise ValueError("PostgreSQL backup requires PostgreSQL")
    snapshot = _snapshot(session)
    metadata = MetaData()
    metadata.reflect(bind=session.connection(), only=ARENA_TABLES)
    runs, episodes = metadata.tables["runs"], metadata.tables["episodes"]
    if (
        run_id is not None
        and session.execute(
            select(runs.c.run_id).where(runs.c.run_id == run_id)
        ).first()
        is None
    ):
        raise ExportNotFound("Run not found")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "arena-database-export-v1",
        "schema_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "database_dialect": session.get_bind().dialect.name,
        "tables": {},
        "files": {},
        "contains_private_player_observations": True,
        "restoration": "JSONL files contain complete database rows; arena-postgres.dump, when requested, is a pg_restore archive.",
    }
    episode_ids = select(episodes.c.episode_id).where(episodes.c.run_id == run_id)
    with tempfile.TemporaryDirectory(
        prefix="arena-export-", dir=output.parent
    ) as staging:
        root = Path(staging)
        for table in metadata.sorted_tables:
            if table.name not in ARENA_TABLES:
                continue
            query = select(table)
            if run_id is not None:
                if "run_id" in table.c:
                    query = query.where(table.c.run_id == run_id)
                elif "episode_id" in table.c:
                    query = query.where(table.c.episode_id.in_(episode_ids))
                elif "matchup_id" in table.c:
                    matchups = metadata.tables["matchups"]
                    query = query.where(
                        table.c.matchup_id.in_(
                            select(matchups.c.matchup_id).where(
                                matchups.c.run_id == run_id
                            )
                        )
                    )
                else:
                    # Global model ratings are not a single run's evidence.
                    continue
            query = query.order_by(*table.primary_key.columns)
            count = 0
            with gzip.open(
                root / (table.name + ".jsonl.gz"), "wt", encoding="utf-8"
            ) as stream:
                for row in session.execute(
                    query.execution_options(yield_per=250)
                ).mappings():
                    stream.write(
                        json.dumps(
                            dict(row),
                            default=_encode,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    count += 1
            manifest["tables"][table.name] = count
        transitions = metadata.tables["episode_transitions"]
        query = select(transitions).order_by(
            transitions.c.episode_id, transitions.c.step_index
        )
        if run_id is not None:
            query = query.where(transitions.c.episode_id.in_(episode_ids))
        count = 0
        with gzip.open(root / "decisions.jsonl.gz", "wt", encoding="utf-8") as stream:
            for transition in session.execute(
                query.execution_options(yield_per=250)
            ).mappings():
                for decision in transition["payload"].get("decisions", []):
                    stream.write(
                        json.dumps(
                            {
                                "episode_id": transition["episode_id"],
                                "step_index": transition["step_index"],
                                "decision": decision,
                            },
                            default=_encode,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    count += 1
        manifest["decisions"] = count
        traces = metadata.tables["episode_traces"]
        query = (
            select(
                traces.c.episode_id,
                episodes.c.run_id,
                traces.c.payload["pending_transition"].label("pending_transition"),
                traces.c.payload["manifest"]["resume_history"].label("resume_history"),
            )
            .join(episodes, episodes.c.episode_id == traces.c.episode_id)
            .order_by(traces.c.episode_id)
        )
        if run_id is not None:
            query = query.where(episodes.c.run_id == run_id)
        count = 0
        with gzip.open(
            root / "pending_attempts.jsonl.gz", "wt", encoding="utf-8"
        ) as stream:
            for trace in session.execute(
                query.execution_options(yield_per=250)
            ).mappings():
                for attempt in episode_pending_attempt_rows(
                    {
                        "pending_transition": trace["pending_transition"],
                        "manifest": {"resume_history": trace["resume_history"]},
                    },
                    episode_id=trace["episode_id"],
                    run_id=trace["run_id"],
                ):
                    stream.write(
                        json.dumps(
                            attempt,
                            default=_encode,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    count += 1
        manifest["pending_attempts"] = count
        if postgres_dump:
            engine = session.get_bind()
            _, connection_parameters = engine.dialect.create_connect_args(engine.url)
            environment_parameters = {
                "host": "PGHOST",
                "hostaddr": "PGHOSTADDR",
                "port": "PGPORT",
                "user": "PGUSER",
                "password": "PGPASSWORD",
                "dbname": "PGDATABASE",
                "connect_timeout": "PGCONNECT_TIMEOUT",
                "application_name": "PGAPPNAME",
                "options": "PGOPTIONS",
                "sslmode": "PGSSLMODE",
                "sslrootcert": "PGSSLROOTCERT",
                "sslcert": "PGSSLCERT",
                "sslkey": "PGSSLKEY",
                "sslcrl": "PGSSLCRL",
                "sslcrldir": "PGSSLCRLDIR",
                "sslcompression": "PGSSLCOMPRESSION",
                "sslsni": "PGSSLSNI",
                "gssencmode": "PGGSSENCMODE",
                "target_session_attrs": "PGTARGETSESSIONATTRS",
                "channel_binding": "PGCHANNELBINDING",
                "passfile": "PGPASSFILE",
                "service": "PGSERVICE",
                "servicefile": "PGSERVICEFILE",
                "client_encoding": "PGCLIENTENCODING",
            }
            unsupported = set(connection_parameters) - set(environment_parameters)
            if unsupported:
                raise ValueError(
                    "pg_dump cannot preserve these connection options: "
                    + ", ".join(sorted(unsupported))
                )
            env = {
                **os.environ,
                **{
                    environment_parameters[key]: str(value)
                    for key, value in connection_parameters.items()
                },
            }
            args = [
                pg_dump_binary,
                "--format=custom",
                "--no-owner",
                "--no-privileges",
                "--snapshot=" + str(snapshot),
                "--file=" + str(root / "arena-postgres.dump"),
            ]
            result = subprocess.run(
                args, env=env, capture_output=True, timeout=300, check=False
            )
            if result.returncode:
                # Do not return subprocess diagnostics that could contain credentials.
                raise RuntimeError(
                    "pg_dump failed; check database access and pg_dump version"
                )
        for path in sorted(root.iterdir()):
            path.chmod(0o600)
            manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        archive = root / "export.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.iterdir()):
                if path != archive:
                    bundle.write(path, arcname=path.name)
        archive.chmod(0o600)
        # Hard-link publication fails atomically if another writer created output.
        os.link(archive, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--database-url-env",
        default="DATABASE_URL",
        help="Environment variable holding the database URL",
    )
    parser.add_argument("--postgres-dump", action="store_true")
    parser.add_argument("--pg-dump-binary", default="pg_dump")
    args = parser.parse_args(argv)
    try:
        database_url = os.environ.get(args.database_url_env)
        if not database_url:
            raise ValueError(
                f"Set {args.database_url_env} explicitly; export never guesses a database target"
            )
        engine = create_engine(database_url)
        try:
            with Session(engine) as session:
                manifest = export_database(
                    session,
                    args.output,
                    run_id=args.run_id,
                    postgres_dump=args.postgres_dump,
                    pg_dump_binary=args.pg_dump_binary,
                )
        finally:
            engine.dispose()
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "tables": manifest["tables"],
                    "decisions": manifest["decisions"],
                    "pending_attempts": manifest["pending_attempts"],
                }
            )
        )
        return 0
    except Exception as exc:
        # SQLAlchemy connection errors can contain connection parameters.
        message = (
            str(exc)
            if isinstance(exc, (ValueError, FileExistsError))
            else type(exc).__name__
        )
        print(json.dumps({"error": message}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
