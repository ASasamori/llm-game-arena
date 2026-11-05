"""Coverage reports distinguish unsupported declarations and visited evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts import viewer_conformance as audit


class ViewerConformanceTests(unittest.TestCase):
    def test_timeouts_crashes_and_bad_output_do_not_become_coverage_passes(self):
        for outcome, expected in (
            (subprocess.TimeoutExpired("worker", 0.1), "timeout"),
            (
                subprocess.CompletedProcess([], -6, "", "native constructor stopped"),
                "failed",
            ),
            (subprocess.CompletedProcess([], 0, "not-json", ""), "failed"),
        ):
            with (
                self.subTest(expected=expected),
                patch.object(
                    audit.subprocess,
                    "run",
                    side_effect=outcome if isinstance(outcome, Exception) else None,
                    return_value=outcome,
                ),
            ):
                result = audit.audit_one("tic_tac_toe", timeout=0.1)
                self.assertEqual(result["status"], expected)
                self.assertIn("reason", result)

    def test_worker_environment_drops_provider_keys_and_forces_mock_mode(self):
        credentials = {
            key: "synthetic-test-secret"
            for key in (
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "TYPESAFE_API_KEY",
                "JEV_API_KEY",
            )
        }
        with (
            patch.dict(
                audit.os.environ, credentials | {"MODEL_API_MOCK_MODE": "false"}
            ),
            patch.object(
                audit.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, '{"status":"passed"}', ""
                ),
            ) as process,
        ):
            self.assertEqual(audit.audit_one("tic_tac_toe")["status"], "passed")
        environment = process.call_args.kwargs["env"]
        self.assertEqual(environment["MODEL_API_MOCK_MODE"], "true")
        self.assertFalse(set(credentials) & set(environment))

    def test_registry_inventory_keeps_skips_separate_and_exercises_fixtures(self):
        entries = [
            {"short_name": "tic_tac_toe", "supported": True, "default_loadable": True},
            {
                "short_name": "repeated_game",
                "supported": True,
                "default_loadable": False,
            },
            {
                "short_name": "new_parameterized_game",
                "supported": True,
                "default_loadable": False,
            },
            {
                "short_name": "chat_game",
                "supported": False,
                "default_loadable": True,
                "unsupported_reasons": ["external model environment"],
            },
            {
                "short_name": "catan",
                "backend": "native",
                "supported": True,
                "default_loadable": True,
            },
        ]
        calls = []

        def checked(spec, **options):
            calls.append((spec, options["parameter_fixture"]))
            return {"game": spec, "status": "passed", "formats": {"text": 2}}

        with (
            patch.object(audit.catalog, "list_games", return_value=entries),
            patch.object(audit, "audit_one", side_effect=checked),
        ):
            result = audit.run_audit(workers=1)
        self.assertEqual(result["registered_openspiel_games"], 4)
        self.assertEqual(
            result["counts"], {"passed": 2, "unsupported": 1, "requires_parameters": 1}
        )
        self.assertEqual(calls, [("tic_tac_toe", False), ("repeated_game", True)])
        self.assertNotIn("catan", {entry["game"] for entry in result["games"]})

    def test_explicit_unsupported_and_missing_parameter_requests_are_not_failures(self):
        with patch.object(
            audit.subprocess,
            "run",
            side_effect=AssertionError("worker should not start"),
        ):
            unsupported = audit.audit_one("chat_game")
            needs_parameters = audit.audit_one("start_at")
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertEqual(needs_parameters["status"], "requires_parameters")

    def test_real_states_cover_public_hidden_simultaneous_and_mean_field_nodes(self):
        for spec, kinds, hidden in (
            ("tic_tac_toe", {"decision", "terminal"}, False),
            ("matrix_rps", {"simultaneous", "terminal"}, False),
            ("kuhn_poker(players=3)", {"chance", "decision", "terminal"}, True),
            (
                "mfg_crowd_modelling(horizon=2)",
                {"chance", "decision", "mean_field", "terminal"},
                False,
            ),
        ):
            with self.subTest(spec=spec):
                result = audit._check_configuration(spec, max_steps=32, seed=19)
                self.assertEqual(result["status"], "passed")
                self.assertTrue(kinds.issubset(result["kinds"]))
                self.assertGreater(result["decisions"], 0)
                self.assertTrue(result["observation_sources"])
                if hidden:
                    self.assertEqual(set(result["formats"]), {"unavailable"})
                else:
                    self.assertTrue(set(result["formats"]) <= {"text", "grid"})

    def test_checker_never_requires_provider_sdk_imports(self):
        script = """import builtins,json
original=builtins.__import__
blocked=('openai','anthropic','google.genai','game_arena.openspiel.model_generation')
def guarded(name,*args,**kwargs):
 if any(name==prefix or name.startswith(prefix+'.') for prefix in blocked):
  raise AssertionError('Unexpected provider import: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from scripts.viewer_conformance import _check_configuration
print(json.dumps(_check_configuration('matrix_rps',4,5)))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=audit.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "passed")

    def test_parameter_fixture_recreates_a_real_started_state(self):
        result = audit.audit_one(
            "start_at", parameter_fixture=True, max_steps=16, seed=123
        )
        self.assertEqual(result["status"], "passed")
        self.assertIn("history=0;1", result["fixture"]["specification"])
        self.assertTrue(result["populations"][0]["terminal"])

    def test_outputs_are_private_nonoverwriting_and_jsonl_is_self_describing(self):
        report = {
            "schema_version": 1,
            "counts": {"passed": 1},
            "games": [{"game": "chess", "status": "passed"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            audit._write(path, report, jsonl=True)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["type"], "summary")
            self.assertEqual(rows[1]["type"], "configuration")
            self.assertEqual(rows[1]["game"], "chess")
            with self.assertRaises(FileExistsError):
                audit._write(path, report)

    def test_invalid_limits_fail_before_spawning(self):
        cases: list[dict[str, Any]] = [
            {"max_steps": 0},
            {"max_steps": True},
            {"timeout": float("nan")},
            {"timeout": 0},
            {"seed": -1},
            {"workers": 0},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                audit.run_audit(["tic_tac_toe"], **options)


if __name__ == "__main__":
    unittest.main()
