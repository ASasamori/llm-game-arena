"""Failed and unapplied model output survives collection and dataset exports."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch
import uuid
import zipfile

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from game_arena.openspiel.datasets import (
    DatasetStore,
    episode_pending_attempt_rows,
    export_dataset,
    pending_attempt_rows,
)
from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.utils import EpisodeResult, HarnessOptions
from orchestrator.exports import export_database
from orchestrator.models import Base, Run
from orchestrator import tasks


def records() -> tuple[dict[str, Any], dict[str, Any]]:
    """Real mock choices, with an explicit unapplied/failed transport fixture."""
    completed: dict[str, Any] = dict(
        harness(
            HarnessOptions(
                type="PVP",
                models=[MockModel("left"), MockModel("right")],
                num_steps=10,
                input=None,
                game="matrix_rps",
                seed=37,
                failure_policy="raise",
            )
        )
    )
    selected, failed = deepcopy(completed["steps"][0]["decisions"])
    selected["usage"] = {"input_tokens": 17, "output_tokens": 6}
    selected["response_metadata"]["fixture"] = "selected, not applied"
    failed.update(
        response="Raw invalid response <literal>\nno selected action",
        usage={"input_tokens": 19, "output_tokens": 7},
        action=None,
        action_string=None,
        failure={
            "stage": "parse",
            "type": "ParseError",
            "message": "No legal action",
            "controlled": False,
        },
    )
    failed.pop("selection", None)
    pending = {
        "index": 0,
        "kind": "simultaneous",
        "applied": False,
        "state_hash_before": completed["steps"][0]["state_hash_before"],
        "decision_count": 1,
        "decisions": [selected],
        "failed_attempt": failed,
    }
    partial = deepcopy(completed)
    partial.update(
        episode_id=str(uuid.uuid4()),
        steps=[],
        num_steps=0,
        terminal=False,
        status="FAILED",
        decision_count=0,
        error="Fixture failure preserves a pending joint turn",
        pending_transition=pending,
        run_id=str(uuid.uuid4()),
    )
    completed["run_id"] = str(uuid.uuid4())
    completed["manifest"]["resume_history"] = [
        {"prefix_transitions": 0, "pending_transition": deepcopy(pending)}
    ]
    return partial, completed


def read_jsonl(path: Path, compressed: bool) -> list[dict[str, Any]]:
    data = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    return [json.loads(line) for line in data.splitlines()]


class PendingExportTests(unittest.TestCase):
    def test_offline_exports_preserve_current_and_historic_attempts_separately(
        self,
    ) -> None:
        for compressed in (False, True):
            with (
                self.subTest(compressed=compressed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source, output = root / "source", root / "export"
                store = DatasetStore(source)
                store.initialize({"experiment_id": "pending-export-fixture"})
                partial, completed = records()
                original = deepcopy(partial)
                for record in (partial, completed):
                    store.write_episode(record, job={"fixture": True})
                counts = export_dataset(source, output, compressed=compressed)
                self.assertEqual(
                    counts,
                    {"episodes": 2, "steps": 1, "decisions": 2, "pending_attempts": 4},
                )
                suffix = ".jsonl.gz" if compressed else ".jsonl"
                attempts = read_jsonl(
                    output / ("pending_attempts" + suffix), compressed
                )
                decisions = read_jsonl(output / ("decisions" + suffix), compressed)
                self.assertEqual(len(decisions), 2)
                self.assertTrue(
                    all(
                        row["episode_id"] == completed["episode_id"]
                        for row in decisions
                    )
                )
                current = [row for row in attempts if not row["historical"]]
                historical = [row for row in attempts if row["historical"]]
                self.assertEqual(
                    [row["attempt_status"] for row in current],
                    ["selected_unapplied", "failed"],
                )
                self.assertEqual(
                    [row["decision"] for row in current],
                    [
                        partial["pending_transition"]["decisions"][0],
                        partial["pending_transition"]["failed_attempt"],
                    ],
                )
                self.assertEqual(
                    [row["decision"] for row in historical],
                    [row["decision"] for row in current],
                )
                self.assertTrue(all(row["resume_index"] == 0 for row in historical))
                self.assertTrue(all(row["applied"] is False for row in attempts))
                self.assertEqual(
                    {row["run_id"] for row in attempts},
                    {partial["run_id"], completed["run_id"]},
                )
                self.assertEqual(
                    {row["experiment_id"] for row in attempts},
                    {"pending-export-fixture"},
                )
                self.assertEqual(
                    json.loads((output / "manifest.json").read_text())["counts"], counts
                )
                for name in ("episodes", "steps", "decisions", "pending_attempts"):
                    self.assertEqual(
                        (output / (name + suffix)).stat().st_mode & 0o777, 0o600
                    )
                self.assertEqual(partial, original)

    def test_database_zip_keeps_pending_raw_evidence_and_correct_run_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = create_engine("sqlite+pysqlite:///:memory:")
            self.addCleanup(engine.dispose)
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine)
            partial, completed = records()
            with patch.object(tasks, "get_session", sessions):
                for record in (partial, completed):
                    with sessions() as session:
                        session.add(
                            Run(
                                run_id=record["run_id"],
                                run_type="PVP",
                                num_episodes=1,
                                status="PENDING",
                            )
                        )
                        session.commit()
                    tasks._persist_result(
                        cast(EpisodeResult, record), record["run_id"], None
                    )
            for run_id, pending_count, applied_count in (
                (None, 4, 2),
                (partial["run_id"], 2, 0),
                (completed["run_id"], 2, 2),
            ):
                output = Path(directory) / (str(run_id) + ".zip")
                with sessions() as session:
                    manifest = export_database(session, output, run_id=run_id)
                self.assertEqual(manifest["pending_attempts"], pending_count)
                self.assertEqual(manifest["decisions"], applied_count)
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                with zipfile.ZipFile(output) as archive:
                    data = archive.read("pending_attempts.jsonl.gz")
                    expected = manifest["files"]["pending_attempts.jsonl.gz"]
                    self.assertEqual(expected["bytes"], len(data))
                    self.assertEqual(
                        expected["sha256"], hashlib.sha256(data).hexdigest()
                    )
                    attempts = [
                        json.loads(line) for line in gzip.decompress(data).splitlines()
                    ]
                    self.assertEqual(len(attempts), pending_count)
                    if run_id is not None:
                        self.assertEqual({row["run_id"] for row in attempts}, {run_id})
                    self.assertTrue(all(row["applied"] is False for row in attempts))
                    failed = next(
                        row for row in attempts if row["attempt_status"] == "failed"
                    )
                    self.assertEqual(
                        failed["decision"],
                        partial["pending_transition"]["failed_attempt"],
                    )
                    # The full original record is preserved alongside normalized rows.
                    traces = [
                        json.loads(line)
                        for line in gzip.decompress(
                            archive.read("episode_traces.jsonl.gz")
                        ).splitlines()
                    ]
                    if run_id == partial["run_id"]:
                        self.assertEqual(
                            traces[0]["payload"]["pending_transition"],
                            partial["pending_transition"],
                        )

    def test_old_datasets_keep_applied_count_meanings_and_empty_attempt_stream(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DatasetStore(root / "source")
            store.initialize({"experiment_id": "old-fixture"})
            _, completed = records()
            completed["manifest"].pop("resume_history")
            store.write_episode(completed, job={})
            counts = export_dataset(store.directory, root / "output", compressed=False)
            self.assertEqual(counts, {"episodes": 1, "steps": 1, "decisions": 2})
            self.assertEqual((root / "output/pending_attempts.jsonl").read_text(), "")
            self.assertEqual(
                json.loads((root / "output/manifest.json").read_text())["counts"][
                    "pending_attempts"
                ],
                0,
            )

    def test_invalid_pending_flags_cannot_be_exported_as_unapplied_actions(
        self,
    ) -> None:
        partial, _ = records()
        for replacement in (
            {"applied": True},
            {"index": -1},
            {"index": False},
            {"decisions": ["not a record"]},
        ):
            pending = partial["pending_transition"] | replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                list(pending_attempt_rows(pending, episode_id=partial["episode_id"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DatasetStore(root / "source")
            store.initialize({})
            partial["pending_transition"]["applied"] = True
            store.write_episode(partial, job={})
            output = root / "output"
            output.mkdir()
            prior = output / "decisions.jsonl"
            prior.write_text("prior export stays intact\n")
            with self.assertRaises(ValueError):
                export_dataset(store.directory, output, compressed=False)
            self.assertEqual(prior.read_text(), "prior export stays intact\n")
            self.assertFalse((output / "pending_attempts.jsonl").exists())

    def test_historical_attempt_source_is_explicit_and_does_not_mutate_record(
        self,
    ) -> None:
        partial, completed = records()
        before = deepcopy(completed)
        attempts = list(
            episode_pending_attempt_rows(
                completed,
                episode_id=completed["episode_id"],
                run_id=completed["run_id"],
            )
        )
        self.assertEqual(len(attempts), 2)
        self.assertEqual({row["resume_index"] for row in attempts}, {0})
        self.assertTrue(all(row["historical"] for row in attempts))
        self.assertEqual(completed, before)
        self.assertEqual(
            attempts[1]["decision"]["prompt"],
            partial["pending_transition"]["failed_attempt"]["prompt"],
        )


if __name__ == "__main__":
    unittest.main()
