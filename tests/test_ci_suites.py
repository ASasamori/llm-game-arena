"""CI must include every test, protect local credentials, and expose failures."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import test_suites


class CiSuiteTests(unittest.TestCase):
    def make_project(self, directory: str, modules: list[str]) -> Path:
        project = Path(directory)
        for module in modules:
            path = project / (module.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return project

    def test_actual_repository_assigns_every_current_and_legacy_module_once(self):
        test_suites.validate_registry()
        assigned = test_suites.modules_for_suite("all")
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), test_suites.discovered_modules())
        self.assertIn("game_arena.openspiel.test_run_harness", assigned)

    def test_new_module_cannot_silently_escape_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self.make_project(
                directory, ["tests.test_known", "tests.test_new"]
            )
            with self.assertRaisesRegex(ValueError, "Unclassified.*tests.test_new"):
                test_suites.validate_registry(project, {"unit": ["tests.test_known"]})

    def test_renamed_and_duplicate_modules_fail_with_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self.make_project(directory, ["tests.test_actual"])
            with self.assertRaisesRegex(ValueError, "Missing.*tests.test_old"):
                test_suites.validate_registry(project, {"unit": ["tests.test_old"]})
            with self.assertRaisesRegex(ValueError, "Duplicate.*tests.test_actual"):
                test_suites.validate_registry(
                    project,
                    {
                        "unit": ["tests.test_actual"],
                        "integration": ["tests.test_actual"],
                    },
                )

    def test_legacy_modules_are_inventoried_even_in_new_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            project = self.make_project(
                directory,
                ["tests.test_new", "game_arena.openspiel.test_legacy"],
            )
            with self.assertRaisesRegex(ValueError, "Unclassified.*test_legacy"):
                test_suites.validate_registry(project, {"unit": ["tests.test_new"]})

    def test_provider_secrets_and_pi_database_are_overridden_before_imports(self):
        with patch.dict(
            os.environ,
            {
                **{key: "sensitive-ambient-value" for key in test_suites.PROVIDER_KEYS},
                "DATABASE_URL": "postgresql://private-database/collection",
                "MODEL_API_MOCK_MODE": "false",
                "PYTHON_DOTENV_DISABLED": "0",
                "WORKER_TRIGGER_MODE": "cloud_run_job",
                "JEV_BUDGET_LEDGER": "/private/live-budget.sqlite",
            },
            clear=True,
        ):
            test_suites.configure_mock_environment()
            self.assertEqual(os.environ["MODEL_API_MOCK_MODE"], "true")
            self.assertEqual(os.environ["PYTHON_DOTENV_DISABLED"], "1")
            self.assertEqual(os.environ["WORKER_TRIGGER_MODE"], "none")
            self.assertEqual(os.environ["DATABASE_URL"], "sqlite+pysqlite:///:memory:")
            self.assertNotIn("JEV_BUDGET_LEDGER", os.environ)
            for variable in test_suites.PROVIDER_KEYS:
                self.assertEqual(os.environ[variable], "mock-ci-not-a-real-key")

    def test_report_distinguishes_failed_tests_from_successful_execution(self):
        class Failure(unittest.TestCase):
            def runTest(self):
                self.fail("deliberate regression fixture")

        result = unittest.TestResult()
        Failure().run(result)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "regression.json"
            test_suites.write_report(output, "regression", ["fixture"], result, 0.125)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(report["successful"])
        self.assertEqual(report["tests_run"], 1)
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn(
            "deliberate regression fixture", report["failures"][0]["traceback"]
        )
        self.assertEqual(report["errors"], [])

    def test_cli_fails_the_job_when_a_test_fails_and_still_writes_its_report(self):
        class Failure(unittest.TestCase):
            def runTest(self):
                self.fail("controlled failure must reach the CI exit status")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "path", list(sys.path)),
            patch.object(test_suites.os, "chdir"),
            patch.object(test_suites, "validate_registry"),
            patch.object(
                unittest.defaultTestLoader,
                "loadTestsFromNames",
                return_value=unittest.TestSuite([Failure()]),
            ),
            redirect_stderr(io.StringIO()),
        ):
            output = Path(directory) / "failed.json"
            status = test_suites.main(
                ["--suite", "regression", "--report", str(output)]
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 1)
        self.assertFalse(report["successful"])
        self.assertEqual(len(report["failures"]), 1)

    def test_list_validates_inventory_without_importing_or_running_tests(self):
        with (
            patch.object(unittest.defaultTestLoader, "loadTestsFromNames") as load,
            patch.object(test_suites, "configure_mock_environment") as configure,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(test_suites.main(["--suite", "unit", "--list"]), 0)
        self.assertEqual(json.loads(output.getvalue())["suite"], "unit")
        load.assert_not_called()
        configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
