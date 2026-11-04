"""TypeSafe Jev choices over legal actions, with no fabricated explanations.

Protocol: https://docs.typesafe.ai/api/system-one
Jev returns typed decisions rather than generated text. The short Summary/Final
Answer string is an explicitly labelled adapter envelope for the legacy arena
parser; the original JSON, probabilities, and confidence remain in metadata.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from game_arena.openspiel.budgeted_anthropic import (
    BudgetLedger,
    JEV_PRICING_SOURCE,
    _usd,
)
from game_arena.openspiel.harness import EpisodeStopped
from game_arena.openspiel import model_generation
from game_arena.openspiel.model_generation import GenerateResponse

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PRICING_SOURCE = JEV_PRICING_SOURCE
MODELS = frozenset(("jev-1.14.0", "jev-1.13.0", "jev-latest", "jev-preview"))
MAX_REQUEST_BYTES = 32_000
MAX_RESPONSE_BYTES = 1_048_576
Transport = Callable[[Request, float], tuple[dict[str, Any], str | None]]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A provider redirect must never forward the bearer credential elsewhere.
        return None


def _post(request: Request, timeout: float) -> tuple[dict[str, Any], str | None]:
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("TypeSafe response exceeded its size limit")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("TypeSafe response must be an object")
        return payload, response.headers.get("x-request-id")


def _context(
    prompt: str, system_instruction: str | None
) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        start = prompt.rfind("\n{") + 1
        context, _ = json.JSONDecoder().raw_decode(prompt[start:])
        if not isinstance(context, dict):
            raise ValueError("Arena context must be an object")
        entries = context["legal_actions"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 255:
            raise ValueError("Jev Choice requires 1-255 legal actions")
        criteria: dict[str, str] = {}
        action_ids: dict[str, int] = {}
        for entry in entries:
            action, label = entry["id"], entry["label"]
            if type(action) is not int or not isinstance(label, str):
                raise ValueError("Invalid legal action")
            name = f"action_{action}"
            if name in action_ids:
                raise ValueError("Duplicate action id")
            criteria[name], action_ids[name] = label, action
        # Whitelist the player-scoped context. Extra evaluator/other-player data
        # in a caller's object cannot accidentally become provider input.
        state = {
            key: context[key]
            for key in (
                "game",
                "num_players",
                "player",
                "utility",
                "observation_type",
                "observation",
            )
            if key in context
        }
        history = context.get("own_action_history", [])
        if not isinstance(history, list):
            raise ValueError("Invalid own-action history")
        state["own_action_history"] = [
            {key: entry[key] for key in ("action", "label") if key in entry}
            for entry in history
            if isinstance(entry, dict)
        ]
        prefix = prompt[:start]
        state["game_instructions"] = prefix.partition("Game instructions:")[2].strip()
        if system_instruction:
            state["system_instruction"] = system_instruction
        request = {
            "state": state,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": (
                        "Select the listed legal action that best advances the acting player's "
                        "own objective under the supplied game rules. Use only that player's "
                        "observation and public information. Public claims may be bluffs."
                    ),
                    "criteria": criteria,
                }
            },
        }
        return request, action_ids
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        raise EpisodeStopped(
            "provider_error",
            "Invalid or oversized legal-action context; no Jev request was made",
        ) from None


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Probability must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return result


def _answer(
    payload: dict[str, Any], action_ids: dict[str, int], requested_model: str
) -> tuple[int, float, dict[str, float]]:
    actual_model = payload["model"]
    if actual_model not in MODELS or (
        requested_model.startswith("jev-1.") and actual_model != requested_model
    ):
        raise ValueError("Jev model version differs from the requested pin")
    answers = payload["answers"]
    if not isinstance(answers, dict) or set(answers) != {"action"}:
        raise ValueError("Unexpected typed answers")
    answer = answers["action"]
    if answer["type"] != "choice" or answer["choice"] not in action_ids:
        raise ValueError("Jev returned an unlisted choice")
    raw = answer["probabilities"]
    if not isinstance(raw, dict) or set(raw) != set(action_ids):
        raise ValueError("Jev probability keys do not match the legal choices")
    probabilities = {name: _probability(value) for name, value in raw.items()}
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Jev probabilities must sum to one")
    if probabilities[answer["choice"]] + 1e-6 < max(probabilities.values()):
        raise ValueError("Jev choice is not a maximum-probability choice")
    return (
        action_ids[answer["choice"]],
        _probability(answer["confidence"]),
        probabilities,
    )


class JevModel:
    """A bounded typed-decision policy; classifier output is never called prose."""

    def __init__(
        self,
        model_name: str = "jev-1.14.0",
        api_key: str | None = None,
        ledger: BudgetLedger | None = None,
        transport: Transport | None = None,
        mock_mode: bool | None = None,
    ):
        if model_name not in MODELS:
            raise ValueError("Unsupported Jev model id")
        self._model_name = model_name
        self._mock_mode = (
            model_generation.MODEL_API_MOCK_MODE if mock_mode is None else mock_mode
        )
        self._transport = transport or _post
        self.ledger = ledger
        self._api_key = ""
        if self._mock_mode:
            return
        self._api_key = (
            api_key
            or os.environ.get("TYPESAFE_API_KEY")
            or os.environ.get("JEV_API_KEY")
            or ""
        )
        if not self._api_key or "\r" in self._api_key or "\n" in self._api_key:
            raise ValueError(
                "Set TYPESAFE_API_KEY or JEV_API_KEY to a valid TypeSafe credential"
            )
        if self.ledger is None:
            configured_path = os.environ.get("JEV_BUDGET_LEDGER")
            path = (
                Path(configured_path)
                if configured_path
                else Path(tempfile.mkdtemp(prefix="arena-jev-budget-"))
                / "budget.sqlite"
            )
            self.ledger = BudgetLedger(
                path,
                budget_usd=os.environ.get("JEV_MAX_COST_USD", "0.10"),
                max_calls=int(os.environ.get("JEV_MAX_CALLS", "100")),
                pricing_source=PRICING_SOURCE,
            )

    def __str__(self) -> str:
        return f"JevModel({self._model_name})"

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        request, action_ids = _context(prompt, system_instruction)
        if len(action_ids) == 1:
            name, action = next(iter(action_ids.items()))
            return GenerateResponse(
                main_response=f"Summary: Adapter applied the only legal action {action}; Jev was not called.\nFinal Answer: {action}",
                usage={"input_tokens": 0, "output_tokens": 0},
                metadata={
                    "provider": "typesafe",
                    "decision_type": "forced_legal_action",
                    "decision_source": "forced_legal_action",
                    "summary_source": "adapter",
                    "provider_response": None,
                    "model": None,
                    "requested_model": self._model_name,
                    "mock": bool(mock or self._mock_mode),
                    "confidence": None,
                    "probabilities": {name: 1.0},
                    "chosen_label": request["questions"]["action"]["criteria"][name],
                    "cost_usd": "0.000000000",
                    "pricing_source": PRICING_SOURCE,
                },
            )
        request["model"] = self._model_name
        serialized = json.dumps(
            request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if len(serialized.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise EpisodeStopped(
                "provider_error",
                "Jev request exceeds the conservative context bound; no request was made",
            )
        if mock or self._mock_mode:
            chosen = next(iter(action_ids))
            payload = {
                "model": self._model_name,
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": chosen,
                        "confidence": 1.0,
                        "probabilities": {
                            name: float(name == chosen) for name in action_ids
                        },
                    }
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            actual_cost, request_id, reservation = 0, None, None
            mocked = True
        else:
            ledger = self.ledger
            assert ledger is not None
            reservation = ledger.reserve(self._model_name, serialized, 64)
            http_request = Request(
                ENDPOINT,
                data=serialized.encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                payload, request_id = self._transport(http_request, 20.0)
            except Exception as exc:
                ledger.fail(reservation, type(exc).__name__)
                if isinstance(exc, HTTPError) and exc.code == 402:
                    message = "TypeSafe requires funded API credits; Jev generation stopped without retry"
                elif isinstance(exc, HTTPError) and exc.code == 401:
                    message = "TypeSafe rejected the API key; Jev generation stopped without retry"
                elif isinstance(exc, HTTPError) and 300 <= exc.code < 400:
                    message = (
                        "TypeSafe redirect was refused; the API key was not forwarded"
                    )
                else:
                    message = "TypeSafe request failed; no retry was made"
                raise EpisodeStopped("provider_error", message) from None
            try:
                usage = payload["usage"]
                input_tokens, output_tokens = (
                    usage["input_tokens"],
                    usage["output_tokens"],
                )
                if type(input_tokens) is not int or type(output_tokens) is not int:
                    raise ValueError("Invalid TypeSafe usage")
                actual_cost = ledger.complete(
                    reservation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_model=payload["model"],
                    request_id=request_id,
                    response_text=json.dumps(
                        payload, ensure_ascii=False, allow_nan=False
                    ),
                )
            except EpisodeStopped:
                raise
            except Exception:
                ledger.fail(reservation, "InvalidProviderUsage")
                raise EpisodeStopped(
                    "provider_error",
                    "TypeSafe returned invalid usage; no retry was made",
                ) from None
            mocked = False
        try:
            action, confidence, probabilities = _answer(
                payload, action_ids, self._model_name
            )
        except (ValueError, KeyError, TypeError, AttributeError, IndexError):
            raise EpisodeStopped(
                "provider_error",
                "TypeSafe returned an invalid typed decision; no retry was made",
            ) from None
        label = request["questions"]["action"]["criteria"][f"action_{action}"]
        prefix = "Adapter mock selected" if mocked else "Adapter selected"
        return GenerateResponse(
            main_response=f"Summary: {prefix} action {action} from Jev choice; no explanation was generated.\nFinal Answer: {action}",
            usage=payload["usage"],
            metadata={
                "provider": "typesafe",
                "decision_type": "choice",
                "summary_source": "adapter",
                "decision_source": "mock" if mocked else "classifier",
                "provider_response": None if mocked else payload,
                "model": None if mocked else payload["model"],
                "requested_model": self._model_name,
                "mock": mocked,
                "confidence": None if mocked else confidence,
                "probabilities": probabilities,
                "chosen_label": label,
                "request_id": request_id,
                "budget_request_id": reservation,
                "cost_usd": _usd(actual_cost),
                "pricing_source": PRICING_SOURCE,
            },
        )
