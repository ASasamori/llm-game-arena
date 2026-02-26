"""
Failure taxonomy and guardrail demo for harness execution.

Runs several deterministic scenarios and prints JSON evidence:
- Successful episode
- Recoverable model API failure (fallback)
- Recoverable parser failure (fallback)
- Recoverable invalid action (fallback)
- Hard failure (invalid INPUT action)
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from game_arena.openspiel import harness, utils
from orchestrator.failures import classify_exception
from orchestrator.observability import get_metrics_snapshot, reset_metrics

os.environ.setdefault("HARNESS_RANDOM_SEED", "7")


@dataclass
class _Response:
    main_response: str


class _StaticModel:
    def __init__(self, name: str, response: str | None = None, error: Exception | None = None):
        self._name = name
        self._response = response
        self._error = error

    def generate_with_text_input(
        self,
        prompt: str,
        system_instruction: str | None = None,
        mock: bool = False,
    ) -> _Response:
        _ = (prompt, system_instruction, mock)
        if self._error is not None:
            raise self._error
        return _Response(main_response=self._response or "Final Answer: 3")

    def __str__(self) -> str:
        return self._name


def _run_case(name: str, options: utils.HarnessOptions) -> dict[str, Any]:
    try:
        result = harness.harness(options)
        return {
            "name": name,
            "outcome": "completed",
            "status": result["status"],
            "winner": result["winner"],
            "num_steps": result["num_steps"],
            "fallback_summaries": [
                s for s in result["summary"] if isinstance(s, str) and "fallback" in s.lower()
            ],
            "error": result["error"],
        }
    except Exception as exc:
        failure = classify_exception(exc, stage="demo.case")
        return {
            "name": name,
            "outcome": "exception",
            "failure": {
                "category": failure.category.value,
                "code": failure.code,
                "retryable": failure.retryable,
                "message": failure.message,
            },
        }


def main() -> None:
    reset_metrics()

    good_model = _StaticModel("GoodModel", response="Final Answer: 3\nSummary: center")
    parser_none_model = _StaticModel("ParserNoneModel", response="maybe choose middle")
    invalid_action_model = _StaticModel(
        "InvalidActionModel", response="x99\nSummary: invalid"
    )
    failing_model = _StaticModel("FailingModel", error=RuntimeError("openai timeout test"))

    cases = [
        _run_case(
            "success_rand",
            utils.HarnessOptions(type="RAND", models=[good_model], num_steps=None, input=None),
        ),
        _run_case(
            "recoverable_model_api_failure",
            utils.HarnessOptions(type="RAND", models=[failing_model], num_steps=6, input=None),
        ),
        _run_case(
            "recoverable_parser_none",
            utils.HarnessOptions(type="RAND", models=[parser_none_model], num_steps=6, input=None),
        ),
        _run_case(
            "recoverable_invalid_action",
            utils.HarnessOptions(type="RAND", models=[invalid_action_model], num_steps=6, input=None),
        ),
        _run_case(
            "hard_failure_invalid_input_action",
            utils.HarnessOptions(
                type="INPUT",
                models=[good_model],
                num_steps=4,
                input=deque([99]),
            ),
        ),
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "metrics_snapshot": get_metrics_snapshot(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
