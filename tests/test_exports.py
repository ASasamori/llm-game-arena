"""Complete, portable collection exports from real mock traces and SQLite."""

from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit
import uuid
import zipfile

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("ARENA_API_KEYS", "local-dev-key")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker

from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.replay import replay_episode
from game_arena.openspiel.utils import EpisodeResult, HarnessOptions
from orchestrator import api_gateway, exports, report_routes, tasks
from orchestrator.models import Base, Episode, ModelRating, Run, RunConfiguration


def read_bundle(data: bytes) -> tuple[dict, dict[str, list[dict]], dict[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        if bundle.testzip() is not None:
            raise AssertionError("Corrupt export ZIP")
        files = {name: bundle.read(name) for name in bundle.namelist()}
    manifest = json.loads(files["manifest.json"])
    rows = {
        name.removesuffix(".jsonl.gz"): [
            json.loads(line) for line in gzip.decompress(value).splitlines()
        ]
        for name, value in files.items()
        if name.endswith(".jsonl.gz")
    }
    return manifest, rows, files


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="arena-export-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.database_url = "sqlite+pysqlite:///" + str(
            self.directory / "collection.sqlite"
        )
        self.engine = create_engine(
            self.database_url, connect_args={"check_same_thread": False}
        )

        @event.listens_for(self.engine, "connect")
        def foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.addCleanup(self.engine.dispose)
        for item in (
            patch.object(api_gateway, "get_session", self.sessions),
            patch.object(tasks, "get_session", self.sessions),
            patch.object(api_gateway, "_enforce_rate_limit"),
            patch.object(api_gateway, "API_KEYS", {"export-test-key"}),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.client = TestClient(api_gateway.app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer export-test-key"}

    def create_run(
        self, episodes: int = 1, game: str = "tic_tac_toe"
    ) -> tuple[str, list[EpisodeResult]]:
        run_id = str(uuid.uuid4())
        with self.sessions() as session:
            session.add(
                Run(
                    run_id=run_id,
                    run_type="PVP",
                    num_episodes=episodes,
                    status="PENDING",
                )
            )
            session.flush()
            session.add(
                RunConfiguration(
                    run_id=run_id,
                    payload={
                        "game": game,
                        "model_names": ["MOCK:alpha", "MOCK:beta"],
                        "seed": 73,
                    },
                )
            )
            session.commit()
        records = []
        for index in range(episodes):
            record = harness(
                HarnessOptions(
                    type="PVP",
                    models=[MockModel("alpha"), MockModel("beta")],
                    num_steps=20,
                    input=None,
                    game=game,
                    seed=73 + index,
                    failure_policy="raise",
                )
            )
            tasks._persist_result(record, run_id, None)
            records.append(record)
        return run_id, records

    def save_export(self, name: str = "collection.zip", run_id: str | None = None):
        output = self.directory / name
        with self.sessions() as session:
            manifest = exports.export_database(session, output, run_id=run_id)
        decoded = read_bundle(output.read_bytes())
        self.assertEqual(manifest, decoded[0])
        return output, decoded

    def test_full_database_export_preserves_all_tables_transitions_and_decisions(
        self,
    ) -> None:
        first_id, first_records = self.create_run(3)
        second_id, second_records = self.create_run()
        with self.sessions() as session:
            session.add(
                ModelRating(
                    rating_id=str(uuid.uuid4()),
                    model_name="global-model",
                    game="tic_tac_toe",
                )
            )
            session.commit()
        records = first_records + second_records
        output, (manifest, rows, files) = self.save_export()
        self.assertEqual(set(manifest["tables"]), set(Base.metadata.tables))
        self.assertEqual(manifest["tables"]["runs"], 2)
        self.assertEqual(manifest["tables"]["episodes"], 4)
        self.assertEqual(manifest["tables"]["model_ratings"], 1)
        self.assertEqual({row["run_id"] for row in rows["runs"]}, {first_id, second_id})
        total_steps = sum(len(record.get("steps", [])) for record in records)
        total_decisions = sum(
            len(step["decisions"])
            for record in records
            for step in record.get("steps", [])
        )
        self.assertGreater(total_steps, 20)
        self.assertEqual(len(rows["episode_transitions"]), total_steps)
        self.assertEqual(len(rows["decisions"]), total_decisions)
        self.assertEqual(manifest["decisions"], total_decisions)
        self.assertTrue(manifest["contains_private_player_observations"])
        self.assertEqual(manifest["database_dialect"], "sqlite")
        self.assertIsNone(manifest["run_id"])
        for name, expected in manifest["files"].items():
            self.assertEqual(len(files[name]), expected["bytes"])
            self.assertEqual(
                hashlib.sha256(files[name]).hexdigest(), expected["sha256"]
            )
        for table, count in manifest["tables"].items():
            self.assertEqual(len(rows[table]), count)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        for row in rows["decisions"]:
            decision = row["decision"]
            self.assertIn("observation", decision)
            self.assertIn("legal_actions", decision)
            self.assertIn("Final Answer:", decision["response"])
            self.assertIn("legal_actions", decision["prompt"])
            self.assertTrue(decision["response_metadata"]["mock"])
        for name, value in files.items():
            uncompressed = gzip.decompress(value) if name.endswith(".gz") else value
            self.assertNotIn(b"export-test-key", uncompressed)
            self.assertNotIn(self.database_url.encode(), uncompressed)

    def test_single_run_export_has_no_pagination_or_cross_run_rows(self) -> None:
        run_id, records = self.create_run(3)
        other_id, other_records = self.create_run()
        with self.sessions() as session:
            session.add(
                ModelRating(
                    rating_id=str(uuid.uuid4()),
                    model_name="global-only",
                    game="tic_tac_toe",
                )
            )
            session.commit()
        _, (manifest, rows, _) = self.save_export(run_id=run_id)
        expected_ids = {record["episode_id"] for record in records}
        self.assertEqual(manifest["run_id"], run_id)
        self.assertNotIn("model_ratings", rows)
        self.assertEqual({row["episode_id"] for row in rows["episodes"]}, expected_ids)
        self.assertGreater(len(rows["episode_transitions"]), 20)
        self.assertEqual(
            len(rows["episode_transitions"]),
            sum(len(record.get("steps", [])) for record in records),
        )
        self.assertEqual(
            {row["episode_id"] for row in rows["episode_transitions"]}, expected_ids
        )
        self.assertEqual({row["episode_id"] for row in rows["decisions"]}, expected_ids)
        serialized = json.dumps(rows)
        self.assertNotIn(other_id, serialized)
        self.assertNotIn(other_records[0]["episode_id"], serialized)
        self.assertEqual(rows["run_configurations"][0]["payload"]["seed"], 73)
        # Full evidence is sufficient to replay every episode without model calls.
        for trace in rows["episode_traces"]:
            record = dict(trace["payload"])
            record["steps"] = [
                row["payload"]
                for row in rows["episode_transitions"]
                if row["episode_id"] == trace["episode_id"]
            ]
            self.assertTrue(replay_episode(record)["verified"])

    def test_empty_database_is_valid_and_existing_output_is_never_overwritten(
        self,
    ) -> None:
        output, (manifest, rows, _) = self.save_export()
        self.assertTrue(all(count == 0 for count in manifest["tables"].values()))
        self.assertEqual(rows["decisions"], [])
        original = output.read_bytes()
        with self.sessions() as session, self.assertRaises(FileExistsError):
            exports.export_database(session, output)
        self.assertEqual(output.read_bytes(), original)

    def test_private_observations_are_preserved_and_explicitly_identified(self) -> None:
        run_id, records = self.create_run(game="kuhn_poker")
        _, (manifest, rows, _) = self.save_export(run_id=run_id)
        self.assertTrue(manifest["contains_private_player_observations"])
        expected = [
            decision
            for step in records[0].get("steps", [])
            for decision in step["decisions"]
        ]
        exported = [row["decision"] for row in rows["decisions"]]
        self.assertEqual(exported, expected)
        self.assertEqual({decision["player"] for decision in exported}, {0, 1})
        self.assertTrue(
            all(
                decision["observation_type"] == "information_state_string"
                for decision in exported
            )
        )
        self.assertGreater(len(rows["episode_transitions"]), len(exported))

    def test_jsonl_export_excludes_unrelated_tables_autoloaded_by_foreign_keys(
        self,
    ) -> None:
        run_id, _ = self.create_run()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE private_users (user_id INTEGER PRIMARY KEY, "
                    "private_email TEXT NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE runs ADD COLUMN owner_id INTEGER "
                    "REFERENCES private_users(user_id)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO private_users (user_id, private_email) "
                    "VALUES (1, 'UNRELATED_PRIVATE_PII@example.invalid')"
                )
            )
            connection.execute(
                text("UPDATE runs SET owner_id=1 WHERE run_id=:run_id"),
                {"run_id": run_id},
            )
        _, (manifest, rows, files) = self.save_export()
        self.assertNotIn("private_users", manifest["tables"])
        self.assertNotIn("private_users", rows)
        self.assertNotIn("private_users.jsonl.gz", files)
        self.assertEqual(rows["runs"][0]["owner_id"], 1)
        self.assertEqual(manifest["tables"]["episodes"], 1)
        for name, value in files.items():
            uncompressed = gzip.decompress(value) if name.endswith(".gz") else value
            self.assertNotIn(b"UNRELATED_PRIVATE_PII", uncompressed)

    def test_unknown_and_invalid_run_ids_publish_nothing_and_do_not_change_database(
        self,
    ) -> None:
        run_id, _ = self.create_run()
        for identity, exception in (
            (str(uuid.uuid4()), exports.ExportNotFound),
            ("bad-id", ValueError),
        ):
            output = self.directory / (identity + ".zip")
            with self.sessions() as session, self.assertRaises(exception):
                exports.export_database(session, output, run_id=identity)
            self.assertFalse(output.exists())
        with self.sessions() as session:
            self.assertIsNotNone(session.get(Run, run_id))
            self.assertEqual(len(list(session.scalars(select(Episode)))), 1)
            session.add(
                Run(
                    run_id=str(uuid.uuid4()),
                    run_type="PVP",
                    num_episodes=1,
                    status="PENDING",
                )
            )
            session.commit()

    def test_failed_export_cleans_staging_and_publishes_nothing(self) -> None:
        self.create_run()
        output = self.directory / "failed.zip"
        with (
            self.sessions() as session,
            patch.object(
                exports, "_digest", side_effect=RuntimeError("injected failure")
            ),
            self.assertRaises(RuntimeError),
        ):
            exports.export_database(session, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.directory.glob("arena-export-*")), [])

    def test_postgres_only_backup_option_is_rejected_for_sqlite(self) -> None:
        output = self.directory / "invalid.zip"
        with (
            self.sessions() as session,
            self.assertRaisesRegex(ValueError, "PostgreSQL"),
        ):
            exports.export_database(session, output, postgres_dump=True)
        with (
            self.sessions() as session,
            self.assertRaisesRegex(ValueError, "whole-database"),
        ):
            exports.export_database(
                session, output, postgres_dump=True, run_id=str(uuid.uuid4())
            )
        self.assertFalse(output.exists())

    def test_authenticated_export_endpoint_downloads_complete_zip(self) -> None:
        run_id, records = self.create_run(3)
        self.create_run()
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            response = self.client.get(f"/v1/runs/{run_id}/export", headers=headers)
            self.assertEqual(response.status_code, 401)
        directories = []
        original_mkdtemp = tempfile.mkdtemp

        def capture(*args, **kwargs):
            directory = original_mkdtemp(*args, **kwargs)
            directories.append(Path(directory))
            return directory

        with patch.object(report_routes.tempfile, "mkdtemp", side_effect=capture):
            response = self.client.get(
                f"/v1/runs/{run_id}/export", headers=self.headers
            )
        self.assertEqual(response.status_code, 200, response.text[:100])
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn(f"arena-{run_id}.zip", response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        manifest, rows, _ = read_bundle(response.content)
        self.assertEqual(manifest["tables"]["episodes"], 3)
        self.assertEqual(
            {row["episode_id"] for row in rows["episodes"]},
            {record["episode_id"] for record in records},
        )
        self.assertGreater(manifest["tables"]["episode_transitions"], 20)
        self.assertTrue(directories)
        self.assertTrue(all(not path.exists() for path in directories))

    def test_export_endpoint_handles_unknown_and_malformed_ids(self) -> None:
        self.assertEqual(
            self.client.get(
                "/v1/runs/not-a-uuid/export", headers=self.headers
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.get(
                f"/v1/runs/{uuid.uuid4()}/export", headers=self.headers
            ).status_code,
            404,
        )

    def test_creation_status_and_report_expose_authenticated_export_link(self) -> None:
        run_id, _ = self.create_run()
        with (
            patch.object(
                api_gateway,
                "_enqueue_with_backoff",
                return_value=(run_id, ["mock-job"]),
            ),
            patch.object(
                api_gateway,
                "_trigger_worker_if_configured",
                return_value=(None, None, None),
            ),
        ):
            created = self.client.post(
                "/v1/runs",
                headers=self.headers,
                json={
                    "run_type": "PVP",
                    "env": "tic_tac_toe",
                    "model_names": ["MOCK:a", "MOCK:b"],
                    "num_episodes": 1,
                    "num_steps": 20,
                },
            )
        self.assertEqual(created.status_code, 202, created.text)
        status = self.client.get(f"/v1/runs/{run_id}", headers=self.headers)
        report = self.client.get(f"/v1/runs/{run_id}/report", headers=self.headers)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(report.status_code, 200)
        for payload in (created.json(), status.json(), report.json()["links"]):
            url = urlsplit(payload["export_url"])
            self.assertEqual(url.scheme, "http")
            self.assertEqual(url.netloc, "testserver")
            self.assertEqual(url.path, f"/v1/runs/{run_id}/export")
            self.assertEqual(url.query, "")
            self.assertNotIn("export-test-key", payload["export_url"])

    def test_cli_uses_explicit_environment_database_url(self) -> None:
        self.create_run()
        output = self.directory / "cli.zip"
        result = io.StringIO()
        with (
            patch.dict(os.environ, {"ARENA_EXPORT_TEST_URL": self.database_url}),
            redirect_stdout(result),
        ):
            code = exports.main(
                ["--output", str(output), "--database-url-env", "ARENA_EXPORT_TEST_URL"]
            )
        self.assertEqual(code, 0, result.getvalue())
        self.assertEqual(json.loads(result.getvalue())["tables"]["episodes"], 1)
        self.assertEqual(read_bundle(output.read_bytes())[0]["tables"]["runs"], 1)

    def test_cli_missing_explicit_database_variable_fails_without_connection(
        self,
    ) -> None:
        output = self.directory / "missing.zip"
        result = io.StringIO()
        with (
            patch.dict(os.environ),
            patch.object(
                exports,
                "create_engine",
                side_effect=AssertionError("unexpected connection"),
            ),
            redirect_stdout(result),
        ):
            os.environ.pop("ARENA_EXPORT_MISSING_URL", None)
            code = exports.main(
                [
                    "--output",
                    str(output),
                    "--database-url-env",
                    "ARENA_EXPORT_MISSING_URL",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("Set ARENA_EXPORT_MISSING_URL explicitly", result.getvalue())
        self.assertFalse(output.exists())

    def test_cli_connection_failure_does_not_print_credentials(self) -> None:
        result = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "ARENA_EXPORT_TEST_URL": "postgresql://user:PRIVATE_PASSWORD@host/db"  # pragma: allowlist secret -- synthetic test credential
                },
            ),
            patch.object(
                exports,
                "create_engine",
                side_effect=RuntimeError("Connection failed: PRIVATE_PASSWORD"),
            ),
            redirect_stdout(result),
        ):
            code = exports.main(
                [
                    "--output",
                    str(self.directory / "failed.zip"),
                    "--database-url-env",
                    "ARENA_EXPORT_TEST_URL",
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(result.getvalue()), {"error": "RuntimeError"})
        self.assertNotIn("PRIVATE_PASSWORD", result.getvalue())

    def test_help_subprocess_needs_no_database_configuration_or_dotenv(self) -> None:
        working = self.directory / "empty-working-directory"
        working.mkdir()
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            # Do not allow the developer's repository .env to mask an import-time
            # dependency on the default database configuration.
            "PYTHON_DOTENV_DISABLED": "1",
        }
        result = subprocess.run(
            [sys.executable, "-m", "orchestrator.exports", "--help"],
            cwd=working,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--database-url-env", result.stdout)
        self.assertIn("--postgres-dump", result.stdout)
        self.assertNotIn("DATABASE_URL", environment)
        self.assertFalse((working / ".env").exists())

    def test_cli_subprocess_supports_only_custom_database_url_in_clean_environment(
        self,
    ) -> None:
        self.create_run()
        working = self.directory / "clean-custom-cli"
        working.mkdir()
        output = working / "custom.zip"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "PYTHON_DOTENV_DISABLED": "1",
            "CUSTOM_URL": self.database_url,
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "orchestrator.exports",
                "--output",
                str(output),
                "--database-url-env",
                "CUSTOM_URL",
            ],
            cwd=working,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["tables"]["episodes"], 1)
        self.assertEqual(read_bundle(output.read_bytes())[0]["tables"]["runs"], 1)
        self.assertNotIn("DATABASE_URL", environment)
        self.assertFalse((working / ".env").exists())

    def test_pg_dump_uses_effective_socket_and_tls_options_without_credentials_in_argv(
        self,
    ) -> None:
        self.create_run()
        socket_directory = "/private/tmp/arena-export-socket"
        certificate = "/private/tmp/arena export root.pem"
        pg_engine = create_engine(
            "postgresql+psycopg2://dump_user:PRIVATE_PASSWORD@ignored-host:55432/arena_collection"  # pragma: allowlist secret -- synthetic pg_dump test credential
            "?host=/private/tmp/arena-export-socket&sslmode=verify-full"
            "&sslrootcert=/private/tmp/arena%20export%20root.pem"
        )
        self.addCleanup(pg_engine.dispose)
        calls = []

        def fake_dump(args, **kwargs):
            calls.append((args, kwargs))
            archive_path = next(
                arg.removeprefix("--file=") for arg in args if arg.startswith("--file=")
            )
            Path(archive_path).write_bytes(b"PGDMP-MOCK-ARCHIVE")
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

        output = self.directory / "with-postgres-backup.zip"
        with self.sessions() as sqlite_session:
            # Keep real schema reflection, row streaming and ZIP generation on
            # SQLite. Only the PostgreSQL snapshot and pg_dump process are mocked.
            session = MagicMock()
            session.get_bind.return_value = pg_engine
            session.connection.side_effect = sqlite_session.connection
            session.execute.side_effect = sqlite_session.execute
            with (
                patch.object(exports, "_snapshot", return_value="00000004-00000023-1"),
                patch.object(exports.subprocess, "run", side_effect=fake_dump),
                patch.dict(
                    os.environ,
                    {
                        "PGHOST": "WRONG_HOST",
                        "PGSSLMODE": "disable",
                        "PGPASSWORD": "WRONG_PASSWORD",  # pragma: allowlist secret -- synthetic test credential or sentinel only
                    },
                ),
            ):
                exports.export_database(
                    session, output, postgres_dump=True, pg_dump_binary="pg_dump-test"
                )
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[0], "pg_dump-test")
        self.assertIn("--snapshot=00000004-00000023-1", args)
        environment = kwargs["env"]
        self.assertEqual(environment["PGHOST"], socket_directory)
        self.assertEqual(environment["PGPORT"], "55432")
        self.assertEqual(environment["PGUSER"], "dump_user")
        self.assertEqual(environment["PGDATABASE"], "arena_collection")
        self.assertEqual(environment["PGPASSWORD"], "PRIVATE_PASSWORD")
        self.assertEqual(environment["PGSSLMODE"], "verify-full")
        self.assertEqual(environment["PGSSLROOTCERT"], certificate)
        command = " ".join(args)
        for value in (
            "PRIVATE_PASSWORD",
            "dump_user",
            "arena_collection",
            socket_directory,
            "ignored-host",
            "postgresql://",
        ):
            self.assertNotIn(value, command)
        self.assertTrue(kwargs["capture_output"])
        manifest, rows, files = read_bundle(output.read_bytes())
        self.assertEqual(manifest["database_dialect"], "postgresql")
        self.assertEqual(manifest["tables"]["episodes"], 1)
        self.assertEqual(len(rows["decisions"]), manifest["decisions"])
        self.assertEqual(files["arena-postgres.dump"], b"PGDMP-MOCK-ARCHIVE")
        for name, value in files.items():
            uncompressed = gzip.decompress(value) if name.endswith(".gz") else value
            self.assertNotIn(b"PRIVATE_PASSWORD", uncompressed)


if __name__ == "__main__":
    unittest.main()
