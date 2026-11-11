"""Offline secret detection on the current Git file snapshot, with redacted output.

The detector scans an isolated copy of tracked and nonignored untracked files.
It never verifies candidates against providers or prints detector raw output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


PROJECT = Path(__file__).resolve().parents[1]
PROVENANCE_FILE = "game_arena/openspiel/data/upstream_games.json"
PROVENANCE_HASH_LENGTHS = {
    "source_sha256": 64,
    "registration_blob_sha1": 40,
    "revision": 40,
}
PROVENANCE_HASH_LINE = re.compile(
    r'(?P<before>[ \t]*"(?P<key>source_sha256|registration_blob_sha1|revision)"\s*:\s*")'
    r'(?P<digest>[a-fA-F0-9]+)(?P<after>",?[ \t]*(?:\r?\n)?)'
)


class ScanError(RuntimeError):
    """The security check could not complete and must fail closed."""


def snapshot_files(root: Path) -> list[str]:
    try:
        listing = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScanError("Could not inventory the Git snapshot") from exc
    if listing.returncode:
        raise ScanError("Git snapshot inventory failed")
    return sorted(
        {
            name.decode("utf-8", errors="surrogateescape")
            for name in listing.stdout.split(b"\0")
            if name
        }
    )


def copy_snapshot(root: Path, destination: Path, files: Sequence[str]) -> set[str]:
    copied = set()
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ScanError("Git snapshot contained an unsafe path")
        if any(part == ".env" for part in relative.parts) or relative.parts[0] in {
            "analytics",
            "logs",
        }:
            raise ScanError(
                "Private environment or generated-data path is tracked; remove it from the Git snapshot"
            )
        source = root
        for part in relative.parts:
            source = source / part
            if source.is_symlink():
                raise ScanError(
                    "Git snapshot contains a symlink; scanner will not follow it"
                )
        if not source.exists():
            # Deleted tracked paths are absent from the current working snapshot.
            continue
        if not source.is_file():
            raise ScanError("Git snapshot contains an unsupported file type")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, target, follow_symlinks=False)
        except OSError as exc:
            raise ScanError("Could not copy the Git snapshot") from exc
        copied.add(relative.as_posix())
    return copied


def mask_reviewed_provenance_hashes(snapshot: Path) -> int:
    """Mask only public source-control/content hashes in the copied provenance JSON.

    These field names and digest widths were reviewed as noncredentials. Keep
    every other byte and line intact, including entropy-bearing api_key fields.
    The original repository file is never rewritten, and entropy plugins remain
    active for this file and all others.
    """
    path = snapshot / PROVENANCE_FILE
    if not path.is_file():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
        if not isinstance(json.loads(text), dict):
            return 0
    except (UnicodeError, ValueError):
        return 0
    count = 0
    lines = []
    for line in text.splitlines(keepends=True):
        match = PROVENANCE_HASH_LINE.fullmatch(line)
        if match and len(match["digest"]) == PROVENANCE_HASH_LENGTHS[match["key"]]:
            line = match["before"] + "0" * len(match["digest"]) + match["after"]
            count += 1
        lines.append(line)
    path.write_text("".join(lines), encoding="utf-8")
    return count


def scan_repository(root: Path = PROJECT) -> dict[str, Any]:
    files = snapshot_files(root)
    with tempfile.TemporaryDirectory(prefix="arena-secret-scan-") as directory:
        snapshot = Path(directory)
        copied = copy_snapshot(root, snapshot, files)
        reviewed_hashes = mask_reviewed_provenance_hashes(snapshot)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "detect_secrets",
                    "scan",
                    "--all-files",
                    "--no-verify",
                    ".",
                ],
                cwd=snapshot,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ScanError("Secret detector could not execute") from exc
        if completed.returncode:
            raise ScanError(
                "Secret detector failed or is not installed; run uv sync --group ci"
            )
        try:
            raw = json.loads(completed.stdout)
            results = raw["results"]
            if not isinstance(results, dict):
                raise ValueError("results must be a mapping")
            findings = []
            for filename, candidates in results.items():
                if filename not in copied or not isinstance(candidates, list):
                    raise ValueError("unexpected result path")
                for candidate in candidates:
                    rule, line = candidate["type"], candidate["line_number"]
                    if (
                        not isinstance(rule, str)
                        or not isinstance(line, int)
                        or line < 1
                    ):
                        raise ValueError("invalid finding metadata")
                    findings.append({"path": filename, "rule": rule, "line": line})
        except (KeyError, TypeError, ValueError) as exc:
            raise ScanError("Secret detector returned an invalid report") from exc
    return {
        "schema_version": 1,
        "scanner": "detect-secrets",
        "offline": True,
        "files_scanned": len(copied),
        "reviewed_provenance_hashes": reviewed_hashes,
        "successful": not findings,
        "findings": sorted(
            findings, key=lambda row: (row["path"], row["line"], row["rule"])
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write JSON containing paths/rules/line numbers only",
    )
    args = parser.parse_args(argv)
    try:
        report = scan_repository(args.root.resolve())
    except ScanError as exc:
        print(f"Secret scan could not complete: {exc}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for finding in report["findings"]:
        # JSON escaping also prevents control characters in filenames reaching
        # terminal output as terminal instructions.
        print(json.dumps(finding, sort_keys=True))
    print(
        f"Offline secret scan: {len(report['findings'])} findings in {report['files_scanned']} files"
    )
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
