"""The offline secret gate must catch candidates without publishing them."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import scan_secrets


def synthetic_token() -> str:
    # Deliberately assembled so this test source does not itself contain a token.
    return "gh" + "p_" + ("AbCdEf0123456789" * 3)[:36]


class SecretScannerTests(unittest.TestCase):
    def project(self, root: Path) -> None:
        subprocess.run(
            ["git", "init", "--quiet", str(root)], check=True, capture_output=True
        )

    def test_real_detector_catches_candidate_in_untracked_file_and_redacts_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            token = synthetic_token()
            (root / "settings.txt").write_text('access_token = "' + token + '"\n')
            output = root / "redacted.json"
            with redirect_stdout(io.StringIO()) as stdout:
                status = scan_secrets.main(
                    ["--root", str(root), "--report", str(output)]
                )
            text = stdout.getvalue() + output.read_text(encoding="utf-8")
            self.assertEqual(status, 1)
            self.assertNotIn(token, text)
            self.assertNotIn("hashed_secret", text)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["offline"])
            self.assertTrue(report["findings"])
            self.assertTrue(
                all(set(row) == {"path", "rule", "line"} for row in report["findings"])
            )
            self.assertEqual(report["findings"][0]["path"], "settings.txt")
            self.assertEqual(report["findings"][0]["line"], 1)

    def test_gitignored_environment_and_data_are_not_read_or_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            (root / ".gitignore").write_text(".env\nanalytics/\n")
            (root / "README.md").write_text("A clean synthetic-data fixture.\n")
            (root / ".env").write_text('access_token = "' + synthetic_token() + '"\n')
            (root / "analytics").mkdir()
            (root / "analytics" / "record.json").write_text(
                'access_token = "' + synthetic_token() + '"\n'
            )
            subprocess.run(
                ["git", "-C", str(root), "add", "README.md"],
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                set(scan_secrets.snapshot_files(root)), {".gitignore", "README.md"}
            )
            report = scan_secrets.scan_repository(root)
            self.assertTrue(report["successful"])
            self.assertEqual(report["files_scanned"], 2)

    def test_detector_always_receives_offline_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            completed = subprocess.CompletedProcess(
                [], 0, stdout='{"results":{}}', stderr=""
            )
            with (
                patch.object(scan_secrets, "snapshot_files", return_value=[]),
                patch.object(
                    scan_secrets.subprocess, "run", return_value=completed
                ) as run,
            ):
                scan_secrets.scan_repository(root)
            self.assertIn("--no-verify", run.call_args.args[0])

    def test_missing_or_failed_scanner_fails_closed_without_printing_raw_output(self):
        secret = synthetic_token()
        completed = subprocess.CompletedProcess([], 1, stdout=secret, stderr=secret)
        with (
            patch.object(scan_secrets, "snapshot_files", return_value=[]),
            patch.object(scan_secrets.subprocess, "run", return_value=completed),
            redirect_stderr(io.StringIO()) as stderr,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(scan_secrets.main([]), 2)
        self.assertNotIn(secret, stderr.getvalue() + stdout.getvalue())
        self.assertIn("could not complete", stderr.getvalue())

    def test_symlink_cannot_read_a_private_file_outside_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            (root / "unsafe.txt").symlink_to(root.parent / "private.env")
            with self.assertRaisesRegex(scan_secrets.ScanError, "symlink"):
                scan_secrets.scan_repository(root)

    def test_accidentally_tracked_environment_fails_without_copying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            (root / ".env").write_text("private configuration")
            with self.assertRaisesRegex(scan_secrets.ScanError, "Private environment"):
                scan_secrets.scan_repository(root)

    def test_only_reviewed_provenance_fields_are_masked_and_source_stays_unchanged(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            path = root / scan_secrets.PROVENANCE_FILE
            path.parent.mkdir(parents=True)
            public_hash = hashlib.sha256(b"public upstream source fixture").hexdigest()
            source = json.dumps({"source_sha256": public_hash}, indent=2) + "\n"
            path.write_text(source, encoding="utf-8")
            clean = scan_secrets.scan_repository(root)
            self.assertTrue(clean["successful"])
            self.assertEqual(clean["reviewed_provenance_hashes"], 1)
            self.assertEqual(path.read_text(encoding="utf-8"), source)
            for field, value in (
                ("provider_credential", synthetic_token()),
                ("api_key", public_hash),
            ):
                with self.subTest(field=field):
                    path.write_text(
                        json.dumps(
                            {"source_sha256": public_hash, field: value}, indent=2
                        )
                        + "\n"
                    )
                    report = scan_secrets.scan_repository(root)
                    self.assertFalse(report["successful"])
                    self.assertEqual(report["reviewed_provenance_hashes"], 1)
                    self.assertTrue(any(row["line"] == 3 for row in report["findings"]))

    def test_provenance_field_exception_does_not_apply_to_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            value = hashlib.sha256(b"public upstream source fixture").hexdigest()
            (root / "other.json").write_text(
                json.dumps({"source_sha256": value}, indent=2) + "\n"
            )
            report = scan_secrets.scan_repository(root)
            self.assertFalse(report["successful"])
            self.assertEqual(report["reviewed_provenance_hashes"], 0)


if __name__ == "__main__":
    unittest.main()
