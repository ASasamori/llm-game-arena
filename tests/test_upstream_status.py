"""Official maturity is pinned, conservative, and distinct from local coverage."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from game_arena.openspiel import upstream_status
from game_arena.openspiel.upstream_status import get_upstream_status


class UpstreamStatusTests(unittest.TestCase):
    def test_release_status_and_current_known_issue_labels_use_pinned_sources(self):
        chess = get_upstream_status("chess")
        self.assertEqual(chess["status"], "thoroughly_tested")
        self.assertEqual(chess["source_ref"], "v1.6.10")
        self.assertEqual(
            chess["source_revision"],
            "7faf3cca520cc1dc4a25bc0aaaec0887804b7fb0",  # pragma: allowlist secret -- public pinned upstream Git commit hash
        )
        for name, issue in (("quoridor", "1158"), ("rbc", "811")):
            with self.subTest(name=name):
                known = get_upstream_status(name)
                self.assertEqual(known["status"], "known_issues")
                self.assertEqual(known["source_ref"], "master")
                self.assertEqual(
                    known["source_revision"],
                    "8eb13ab561c4fac6829a5fa03635d3e1658bbb60",  # pragma: allowlist secret -- public pinned upstream Git commit hash
                )
                self.assertIn(
                    "https://github.com/google-deepmind/open_spiel/issues/" + issue,
                    known["issue_urls"],
                )

    def test_implemented_but_lightly_tested_is_not_promoted_to_green(self):
        for name in ("hex", "amazons", "2048", "checkers", "python_block_dominoes"):
            with self.subTest(name=name):
                result = get_upstream_status(name)
                self.assertEqual(result["status"], "lightly_tested")
                self.assertEqual(result["release_status"], "lightly_tested")

    def test_unlisted_helpers_ports_and_variants_inherit_no_maturity_rating(self):
        for name in (
            "add_noise",
            "matrix_rps",
            "python_kuhn_poker",
            "dark_hex_ir",
            "repeated_leduc_poker",
            "othello",
            "avalon",
            "catan",
            "unknown_registration",
        ):
            with self.subTest(name=name):
                result = get_upstream_status(name)
                self.assertEqual(result["status"], "unlisted")
                self.assertIsNone(result["release_status"])
                self.assertTrue(
                    any("Not individually rated" in note for note in result["notes"])
                )

    def test_current_known_issues_override_an_older_green_rating(self):
        manifest = deepcopy(upstream_status._manifest())
        document = manifest["documentation"][
            manifest["games"]["chess"]["documentation_key"]
        ]
        document["current"]["status"] = "known_issues"
        with patch.object(upstream_status, "_manifest", return_value=manifest):
            result = get_upstream_status("chess")
        self.assertEqual(result["release_status"], "thoroughly_tested")
        self.assertEqual(result["status"], "known_issues")
        self.assertEqual(result["source_ref"], "master")

    def test_returned_lists_are_independent_from_the_cached_manifest(self):
        result = get_upstream_status("quoridor")
        result["issue_urls"].append("modified")
        result["notes"].append("modified")
        fresh = get_upstream_status("quoridor")
        self.assertNotIn("modified", fresh["issue_urls"])
        self.assertNotIn("modified", fresh["notes"])

    def test_status_lookup_never_requires_network_or_provider_credentials(self):
        upstream_status._manifest.cache_clear()
        with patch(
            "socket.socket", side_effect=AssertionError("Unexpected network access")
        ):
            self.assertEqual(
                get_upstream_status("tic_tac_toe")["status"], "thoroughly_tested"
            )
        upstream_status._manifest.cache_clear()

    def test_frozen_manifest_has_complete_provenance_and_unique_explicit_mappings(self):
        path = Path(upstream_status.__file__).with_name("data") / "upstream_games.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["installed_release"], "1.6.10")
        self.assertEqual(len(data["games"]), 87)
        self.assertEqual(
            Counter(get_upstream_status(name)["status"] for name in data["games"]),
            {"thoroughly_tested": 38, "lightly_tested": 47, "known_issues": 2},
        )
        for source in data["sources"].values():
            self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(source["source_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(source["revision"], source["source_url"])
            self.assertIn(source["revision"], source["raw_url"])
            self.assertTrue(
                source["source_url"].startswith(
                    "https://github.com/google-deepmind/open_spiel/blob/"
                )
            )
        for name, mapping in data["games"].items():
            with self.subTest(name=name):
                self.assertIn(mapping["documentation_key"], data["documentation"])
                self.assertRegex(mapping["registration_blob_sha1"], r"^[0-9a-f]{40}$")
                self.assertTrue(
                    mapping["registration_source_path"].startswith("open_spiel/")
                )
                self.assertFalse(
                    re.search(
                        r"(?:^|/)\.\.(?:/|$)", mapping["registration_source_path"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
