"""Run explicit, credential-free unittest suites used by GitHub Actions.

Every test module belongs to one suite. Adding or renaming a test requires an
entry below so that CI cannot silently omit it. External-service end-to-end
tests remain in the separate disposable Postgres/Redis workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence
import unittest


PROJECT = Path(__file__).resolve().parents[1]
TEST_DIRECTORIES = ("tests", "game_arena/openspiel")
SUITES: dict[str, tuple[str, ...]] = {
    "unit": (
        "tests.test_avalon_environment",
        "tests.test_budgeted_anthropic",
        "tests.test_catalog",
        "tests.test_catan_environment",
        "tests.test_ci_suites",
        "tests.test_coup_environment",
        "tests.test_jev",
        "tests.test_presentation",
        "tests.test_settings",
        "tests.test_upstream_status",
        "tests.test_werewolf_environment",
        "game_arena.openspiel.test_model_generation_guardrails",
        "game_arena.openspiel.test_parsers_harness",
    ),
    "integration": (
        "tests.test_client_stylize_api",
        "tests.test_engine",
        "tests.test_experiments",
        "tests.test_exports",
        "tests.test_game_conformance",
        "tests.test_jev_integration",
        "tests.test_live_trial",
        "tests.test_native_integration",
        "tests.test_orchestration",
        "tests.test_report_api",
        "tests.test_reports",
        "tests.test_viewer_conformance",
    ),
    "regression": (
        "tests.test_catalog_upstream",
        "tests.test_decision_audit",
        "tests.test_episode_stops",
        "tests.test_pending_decisions",
        "tests.test_pending_exports",
        "tests.test_public_boards",
        "tests.test_replay_integrity",
        "tests.test_resumption",
        "game_arena.openspiel.test_run_harness",
    ),
    "security": ("tests.test_security", "tests.test_secret_scanner"),
}

PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "TYPESAFE_API_KEY",
    "JEV_API_KEY",
)


def discovered_modules(project: Path = PROJECT) -> set[str]:
    """Inventory both current tests and the retained legacy test modules."""
    return {
        ".".join(path.relative_to(project).with_suffix("").parts)
        for directory in TEST_DIRECTORIES
        for path in (project / directory).rglob("test_*.py")
    }


def validate_registry(
    project: Path = PROJECT,
    registry: Mapping[str, Sequence[str]] = SUITES,
) -> None:
    """Fail closed when a new, moved, removed, or duplicate test is unaccounted for."""
    assigned: dict[str, str] = {}
    problems = []
    for suite, modules in registry.items():
        if not modules:
            problems.append(f"Suite {suite!r} has no test modules")
        for module in modules:
            if module in assigned:
                problems.append(
                    f"Duplicate module {module}: {assigned[module]} and {suite}"
                )
            assigned[module] = suite
    discovered = discovered_modules(project)
    for module in sorted(discovered - assigned.keys()):
        problems.append(f"Unclassified test module: {module}; add it to SUITES")
    for module in sorted(assigned.keys() - discovered):
        problems.append(f"Missing test module: {module}; update SUITES")
    if problems:
        raise ValueError("\n".join(problems))


def modules_for_suite(
    suite: str, registry: Mapping[str, Sequence[str]] = SUITES
) -> list[str]:
    if suite == "all":
        return [module for modules in registry.values() for module in modules]
    return list(registry[suite])


def configure_mock_environment() -> None:
    """Override ambient .env credentials and the Pi database before test imports."""
    os.environ.update(
        PYTHON_DOTENV_DISABLED="1",
        MODEL_API_MOCK_MODE="true",
        MODEL_API_MOCK_CASE="default",
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        DATABASE_TARGET="temporary_local",
        WORKER_TRIGGER_MODE="none",
        ARENA_API_KEYS="mock-ci-not-a-real-key",  # pragma: allowlist secret -- explicit dummy gateway key for CI
    )
    for variable in PROVIDER_KEYS:
        os.environ[variable] = "mock-ci-not-a-real-key"
    # A local developer may have a funded live-trial ledger configured. Tests
    # must use their own temporary ledgers rather than touch that collection.
    os.environ.pop("JEV_BUDGET_LEDGER", None)


def write_report(
    path: Path,
    suite: str,
    modules: Sequence[str],
    result: unittest.TestResult,
    seconds: float,
) -> None:
    report = {
        "schema_version": 1,
        "suite": suite,
        "modules": list(modules),
        "tests_run": result.testsRun,
        "successful": result.wasSuccessful(),
        "duration_seconds": round(seconds, 3),
        "failures": [
            {"test": str(test), "traceback": trace} for test, trace in result.failures
        ],
        "errors": [
            {"test": str(test), "traceback": trace} for test, trace in result.errors
        ],
        "skipped": [
            {"test": str(test), "reason": reason} for test, reason in result.skipped
        ],
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": [str(test) for test in result.unexpectedSuccesses],
        "mock_provider_mode": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument(
        "--report", type=Path, help="Write a machine-readable JSON result"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Validate and list modules without importing tests",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_registry()
    except ValueError as exc:
        parser.error(str(exc))
    modules = modules_for_suite(args.suite)
    if args.list:
        print(json.dumps({"suite": args.suite, "modules": modules}, indent=2))
        return 0
    configure_mock_environment()
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    if args.report:
        write_report(
            args.report, args.suite, modules, result, time.perf_counter() - started
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
