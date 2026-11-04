"""Atomic episode artifacts and streaming, normalized JSONL export.

Each committed episode consists of a small summary and a separate transition
stream. The summary is published last, so interrupted writes are never mistaken
for completed episodes. No run-wide collection is held in memory.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, TextIO

SCHEMA_VERSION = 1


def json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _atomic_text(path: Path, lines: Iterable[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            tmp_name = stream.name
            for line in lines:
                stream.write(line)
                digest.update(line.encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    return digest.hexdigest()


class DatasetStore:
    """One directory per immutable experiment; safe to use from worker processes.

    Disjoint shards may share a directory. Running the same shard concurrently is
    intentionally unsupported; workers own distinct stable episode IDs.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def initialize(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "manifest.json"
        normalized = json.loads(canonical_json(manifest))
        temporary: str | None = None
        try:
            # Publish a fully flushed inode exclusively; another shard must never
            # see a partially written manifest.
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.directory, delete=False
            ) as stream:
                temporary = stream.name
                stream.write(canonical_json(normalized) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != normalized:
                    raise ValueError(
                        "Dataset manifest differs from this plan or build; choose a new output directory"
                    )
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return normalized

    def manifest(self) -> dict[str, Any]:
        return json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )

    def _summary_path(self, episode_id: str) -> Path:
        # A stable hex/UUID identifier only; never interpret arbitrary paths.
        if not episode_id or any(
            char not in "0123456789abcdef-" for char in episode_id
        ):
            raise ValueError("Invalid episode ID")
        return self.directory / "episodes" / episode_id[:2] / f"{episode_id}.json"

    def write_episode(self, record: dict[str, Any], *, job: dict[str, Any]) -> Path:
        episode_id = str(record["episode_id"])
        summary_path = self._summary_path(episode_id)
        step_path = summary_path.with_suffix(".steps.jsonl")
        steps = record.get("steps", [])
        digest = _atomic_text(
            step_path, (canonical_json(step) + "\n" for step in steps)
        )
        payload = {key: value for key, value in record.items() if key != "steps"}
        payload.update(
            {
                "dataset_schema_version": SCHEMA_VERSION,
                "job": job,
                "steps_file": str(step_path.relative_to(self.directory)),
                "steps_sha256": digest,
                "transition_count": len(steps),
            }
        )
        envelope = {"sha256": fingerprint(payload), "episode": payload}
        _atomic_text(summary_path, [canonical_json(envelope) + "\n"])
        return summary_path

    def read_episode(
        self,
        episode_id: str,
        *,
        expected_job: dict[str, Any] | None = None,
        include_steps: bool = True,
    ) -> dict[str, Any]:
        path = self._summary_path(episode_id)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["episode"]
        if envelope["sha256"] != fingerprint(payload):
            raise ValueError(f"Episode checksum mismatch: {episode_id}")
        if payload.get("episode_id") != episode_id:
            raise ValueError(f"Episode identity mismatch: {episode_id}")
        if expected_job is not None and payload.get("job") != json.loads(
            canonical_json(expected_job)
        ):
            raise ValueError(f"Episode job mismatch: {episode_id}")
        expected_path = path.with_suffix(".steps.jsonl")
        if payload.get("steps_file") != str(expected_path.relative_to(self.directory)):
            raise ValueError(f"Invalid transition path: {episode_id}")
        digest = hashlib.sha256()
        count = 0
        steps: list[dict[str, Any]] = []
        with expected_path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                step = json.loads(line)
                if step.get("index") != count:
                    raise ValueError(f"Transition order mismatch: {episode_id}")
                if include_steps:
                    steps.append(step)
                count += 1
        if (
            digest.hexdigest() != payload["steps_sha256"]
            or count != payload["transition_count"]
        ):
            raise ValueError(f"Transition checksum mismatch: {episode_id}")
        result = dict(payload)
        if include_steps:
            result["steps"] = steps
        return result

    def is_complete(self, episode_id: str, *, expected_job: dict[str, Any]) -> bool:
        if not self._summary_path(episode_id).exists():
            return False
        # Failed records are committed evidence, but are retried on resume.
        return self.read_episode(
            episode_id, expected_job=expected_job, include_steps=False
        ).get("status") in {"SUCCESS", "TRUNCATED"}

    def iter_episode_ids(self) -> Iterator[str]:
        # scandir avoids materializing a run-sized glob/sorted file list.
        root = self.directory / "episodes"
        if not root.exists():
            return
        with os.scandir(root) as buckets:
            for bucket in buckets:
                if not bucket.is_dir():
                    continue
                with os.scandir(bucket.path) as files:
                    for file in files:
                        if file.name.endswith(".json"):
                            yield file.name.removesuffix(".json")

    def iter_steps(self, episode_id: str) -> Iterator[dict[str, Any]]:
        path = self._summary_path(episode_id).with_suffix(".steps.jsonl")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _export_stream(path: Path, compressed: bool) -> TextIO:
    if compressed:
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def export_dataset(
    source: str | Path, destination: str | Path, *, compressed: bool = True
) -> dict[str, int]:
    """Stream validated artifacts into episode, transition, and decision datasets.

    Export files are replaced only when all records validate. Decision records
    can contain hidden player observations and model responses; treat exports as
    evaluator data rather than public prompts for other players.
    """
    from contextlib import ExitStack

    store = DatasetStore(source)
    manifest = store.manifest()
    destination = Path(destination)
    if destination.resolve() == store.directory.resolve():
        raise ValueError("Export destination must differ from the source dataset")
    destination.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz" if compressed else ".jsonl"
    counts = {"episodes": 0, "steps": 0, "decisions": 0}
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as temporary:
        temporary_path = Path(temporary)
        with ExitStack() as stack:
            streams = {
                kind: stack.enter_context(
                    _export_stream(temporary_path / f"{kind}{suffix}", compressed)
                )
                for kind in counts
            }
            for episode_id in store.iter_episode_ids():
                record = store.read_episode(episode_id, include_steps=False)
                streams["episodes"].write(canonical_json(record) + "\n")
                counts["episodes"] += 1
                for step in store.iter_steps(episode_id):
                    transition = {"episode_id": episode_id, **step}
                    decisions = transition.pop("decisions", [])
                    streams["steps"].write(canonical_json(transition) + "\n")
                    counts["steps"] += 1
                    for decision_index, decision in enumerate(decisions):
                        streams["decisions"].write(
                            canonical_json(
                                {
                                    **decision,
                                    "episode_id": episode_id,
                                    "step_index": step["index"],
                                    "decision_index": decision_index,
                                }
                            )
                            + "\n"
                        )
                        counts["decisions"] += 1
        for kind in counts:
            os.replace(
                temporary_path / f"{kind}{suffix}", destination / f"{kind}{suffix}"
            )
        _atomic_text(
            destination / "manifest.json",
            [
                canonical_json(
                    {
                        "dataset_schema_version": SCHEMA_VERSION,
                        "source_manifest": manifest,
                        "counts": counts,
                        "compressed": compressed,
                    }
                )
                + "\n"
            ],
        )
    return counts
