"""Frozen, source-pinned OpenSpiel maturity labels; no runtime network access.

The official documentation rates games, not every registration, parameter set,
transform, or Python port. Only explicit identities in the reviewed manifest
inherit a documented label. An unrated helper remains ``unlisted`` rather than
being promoted because a bounded local smoke test passed.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any


_PATH = Path(__file__).with_name("data") / "upstream_games.json"
_STATUSES = {"thoroughly_tested", "lightly_tested", "known_issues"}


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    contents = _PATH.read_bytes()
    result = json.loads(contents)
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported frozen OpenSpiel status manifest")
    if not isinstance(result.get("games"), dict) or not isinstance(
        result.get("sources"), dict
    ):
        raise ValueError("Malformed frozen OpenSpiel status manifest")
    result["manifest_sha256"] = hashlib.sha256(contents).hexdigest()
    return result


def get_upstream_status(short_name: str) -> dict[str, Any]:
    """Return independent official status evidence for one registry identity.

    Release documentation is preferred. A known issue in the pinned current
    upstream documentation overrides an older maturity label. Unknown names and
    separately registered variants receive no inferred green/orange rating.
    """
    if not isinstance(short_name, str) or not short_name.strip():
        raise ValueError("short_name must be a nonempty registry name")
    manifest = _manifest()
    mapping = manifest["games"].get(short_name)
    document = (
        manifest["documentation"].get(mapping["documentation_key"], {})
        if mapping
        else {}
    )
    release = document.get("release")
    current = document.get("current")
    chosen = (
        "current"
        if current and current["status"] == "known_issues"
        else "release"
        if release
        else "current"
    )
    evidence = document.get(chosen)
    source = manifest["sources"].get(chosen) or manifest["sources"]["release"]
    status = evidence["status"] if evidence else "unlisted"
    if status != "unlisted" and status not in _STATUSES:
        raise ValueError("Unknown status in frozen OpenSpiel manifest")
    notes = (
        [
            "Not individually rated in the pinned official game-status table; no maturity status is inferred."
        ]
        if status == "unlisted"
        else [
            "Official documentation maturity is separate from this arena's configuration and replay checks."
        ]
    )
    if mapping and mapping.get("mapping_note"):
        notes.append(mapping["mapping_note"])
    if chosen == "current" and status == "known_issues":
        notes.append(
            "Current pinned upstream documentation flags known issues; this takes precedence over release maturity."
        )
    result = {
        "schema_version": 1,
        "short_name": short_name,
        "display_name": document.get("display_name", short_name),
        "status": status,
        "source_url": source["source_url"]
        + ("#L" + str(evidence["line"]) if evidence else ""),
        "source_revision": source["revision"],
        "source_date": source["commit_date"],
        "checked_at": manifest["checked_at"],
        "source_ref": source["ref"],
        "source_sha256": source["source_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "release_status": release["status"] if release else None,
        "current_status": current["status"] if current else None,
        "issue_urls": list(evidence.get("issue_urls", [])) if evidence else [],
        "notes": notes,
    }
    if mapping:
        result["registration_source_url"] = mapping["registration_source_url"]
    return deepcopy(result)
