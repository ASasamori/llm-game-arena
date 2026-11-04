"""Versioned, secret-free provenance for runs and exported datasets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PROMPT_VERSION = "openspiel-player-view-v1"
PARSER_VERSION = "legal-action-id-v1"


@lru_cache(maxsize=1)
def runtime_manifest() -> dict[str, Any]:
    """Fingerprint working source too: HEAD alone cannot identify an uncommitted build."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in (root / "game_arena", root / "orchestrator"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    lockfile = root / "uv.lock"
    dependency_versions: dict[str, str] = {}
    for package in ("openai", "anthropic", "google-genai", "numpy"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "open_spiel_version": importlib.metadata.version("open-spiel"),
        "python_version": platform.python_version(),
        "source_revision": revision,
        "source_sha256": digest.hexdigest(),
        "dependency_versions": dependency_versions,
        "dependency_lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest()
        if lockfile.exists()
        else None,
    }


def derive_seed(seed: int, *parts: object) -> int:
    """Independent RNG streams that do not depend on process or completion order."""
    import json

    payload = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
