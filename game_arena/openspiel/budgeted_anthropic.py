"""Durable provider spending reservations and one-attempt Anthropic calls.

This deliberately does not use the normal provider retry decorator. Unknown
charges remain reserved after failures. A key and prompts never enter the ledger.
Provider prices were checked against their published lists on 2026-11-04.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from game_arena.openspiel.harness import EpisodeStopped
from game_arena.openspiel.parsers import parse_legal_action
from game_arena.openspiel.model_generation import (
    GenerateResponse,
    _validate_model_response,
)

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
# Integer nano-USD per token; published rates are USD per million tokens.
PRICES: dict[str, tuple[int, int]] = {
    "claude-haiku-5-5": (100, 500),
    "claude-haiku-4-5-20251001": (1000, 5000),
    "claude-sonnet-5-5": (2000, 10000),
    "claude-sonnet-5": (2000, 10000),
    "claude-sonnet-4-6": (3000, 15000),
}
ANTHROPIC_MODELS = frozenset(PRICES)
JEV_PRICING_SOURCE = (
    "https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost"
)
JEV_MODELS = frozenset({"jev-1.14.0", "jev-1.13.0", "jev-latest", "jev-preview"})
PRICES.update({model: (42, 0) for model in JEV_MODELS})
NANO_USD = 1_000_000_000


def _usd(value: int) -> str:
    return format(Decimal(value) / NANO_USD, ".9f")


class BudgetLedger:
    """Serialize paid-request admission across seats and process restarts."""

    def __init__(
        self,
        path: Path,
        budget_usd: str = "0.90",
        max_calls: int = 100,
        pricing_source: str = PRICING_SOURCE,
    ):
        try:
            amount = Decimal(budget_usd)
            if not amount.is_finite() or not Decimal("0") < amount <= Decimal("1"):
                raise ValueError(
                    "Trial budget must be greater than zero and at most $1"
                )
            cap = int(amount * NANO_USD)
        except (InvalidOperation, TypeError) as exc:
            raise ValueError("Invalid trial budget") from exc
        if cap < 1 or type(max_calls) is not int or not 1 <= max_calls <= 1000:
            raise ValueError(
                "Budget must be at least one nano-USD; max_calls must be 1-1000"
            )
        if not isinstance(pricing_source, str) or pricing_source not in {
            PRICING_SOURCE,
            JEV_PRICING_SOURCE,
        }:
            raise ValueError("A verified provider pricing source is required")
        models = ANTHROPIC_MODELS if pricing_source == PRICING_SOURCE else JEV_MODELS
        self.prices = {model: PRICES[model] for model in models}
        prices_json = json.dumps(self.prices, sort_keys=True, separators=(",", ":"))
        self.pricing_source = pricing_source
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        with self._transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), "
                "cap INTEGER NOT NULL, max_calls INTEGER NOT NULL, blocked INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, model TEXT NOT NULL, "
                "prompt_sha256 TEXT NOT NULL, input_bound INTEGER NOT NULL, output_bound INTEGER NOT NULL, "
                "reserved INTEGER NOT NULL, actual INTEGER, status TEXT NOT NULL, "
                "input_tokens INTEGER, output_tokens INTEGER, response_model TEXT, request_id TEXT, "
                "response_text TEXT, error_type TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS episode_claims (episode_id TEXT PRIMARY KEY)"
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(budget)")
            }
            for column in ("pricing_source", "prices_json"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE budget ADD COLUMN {column} TEXT")
            stored = connection.execute(
                "SELECT cap,max_calls,pricing_source,prices_json FROM budget WHERE id=1"
            ).fetchone()
            if stored is None:
                connection.execute(
                    "INSERT INTO budget(id,cap,max_calls,pricing_source,prices_json) VALUES(1,?,?,?,?)",
                    (cap, max_calls, pricing_source, prices_json),
                )
                stored = (cap, max_calls, pricing_source, prices_json)
            elif stored[2] is None or stored[3] is None:
                # Every ledger predating pricing metadata used Anthropic rates.
                # It cannot be silently repurposed with another provider/source.
                previous_source = stored[2] or PRICING_SOURCE
                previous_prices = stored[3] or json.dumps(
                    {model: PRICES[model] for model in ANTHROPIC_MODELS},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    "UPDATE budget SET pricing_source=?,prices_json=? WHERE id=1",
                    (previous_source, previous_prices),
                )
                stored = (stored[0], stored[1], previous_source, previous_prices)
            if stored != (cap, max_calls, pricing_source, prices_json):
                raise ValueError(
                    "Existing ledger has a different immutable spending limit or pricing configuration"
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_episode_once(self, episode_id: str) -> bool:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("An episode identity is required")
        with self._transaction() as connection:
            result = connection.execute(
                "INSERT OR IGNORE INTO episode_claims(episode_id) VALUES(?)",
                (episode_id,),
            )
            return result.rowcount == 1

    def reserve(self, model: str, prompt: str, max_tokens: int) -> int:
        if model not in self.prices:
            raise ValueError("Model has no verified price in this ledger")
        if type(max_tokens) is not int or not 64 <= max_tokens <= 256:
            raise ValueError("max_tokens must be an integer from 64-256")
        # Byte-level conservative bound avoids treating count_tokens estimates as
        # billing guarantees. Only a plain text user/system prompt is sent; there
        # are no tools, images, cache writes, or long-context pricing tiers.
        input_bound = len(prompt.encode("utf-8")) + 1024
        if input_bound >= 100_000:
            raise EpisodeStopped(
                "budget_exhausted",
                "Prompt exceeds the trial's short-context price bound",
            )
        input_rate, output_rate = self.prices[model]
        reservation = input_bound * input_rate + max_tokens * output_rate
        with self._transaction() as connection:
            cap, max_calls, blocked = connection.execute(
                "SELECT cap,max_calls,blocked FROM budget WHERE id=1"
            ).fetchone()
            count, charged = connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(COALESCE(actual,reserved)),0) FROM requests"
            ).fetchone()
            if blocked:
                raise EpisodeStopped(
                    "budget_exhausted",
                    "Ledger blocked after an unexpected usage response",
                )
            if count >= max_calls:
                raise EpisodeStopped(
                    "call_limit",
                    "Maximum trial request count reached before another request",
                )
            if charged + reservation > cap:
                raise EpisodeStopped(
                    "budget_exhausted",
                    "Next request would exceed the conservative trial budget",
                )
            cursor = connection.execute(
                "INSERT INTO requests(model,prompt_sha256,input_bound,output_bound,reserved,status) "
                "VALUES(?,?,?,?,?,'reserved')",
                (
                    model,
                    hashlib.sha256(prompt.encode()).hexdigest(),
                    input_bound,
                    max_tokens,
                    reservation,
                ),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def fail(self, request: int, error_type: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE requests SET status='uncertain',error_type=? WHERE id=? AND status='reserved'",
                (error_type, request),
            )

    def complete(
        self,
        request: int,
        *,
        input_tokens: int,
        output_tokens: int,
        response_model: str,
        request_id: str | None,
        response_text: str,
    ) -> int:
        invalid = any(
            type(value) is not int or value < 0
            for value in (input_tokens, output_tokens)
        )
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT model,input_bound,output_bound,reserved,status FROM requests WHERE id=?",
                (request,),
            ).fetchone()
            if row is None or row[4] != "reserved":
                raise ValueError("Request reservation is not outstanding")
            model, input_bound, output_bound, reserved, _ = row
            input_rate, output_rate = self.prices[model]
            unexpected = (
                invalid
                or input_tokens > input_bound
                or (output_rate > 0 and output_tokens > output_bound)
            )
            if unexpected:
                connection.execute("UPDATE budget SET blocked=1 WHERE id=1")
                connection.execute(
                    "UPDATE requests SET status='usage_overrun' WHERE id=?", (request,)
                )
                actual = reserved
            else:
                actual = input_tokens * input_rate + output_tokens * output_rate
                connection.execute(
                    "UPDATE requests SET actual=?,status='completed',input_tokens=?,output_tokens=?,"
                    "response_model=?,request_id=?,response_text=? WHERE id=?",
                    (
                        actual,
                        input_tokens,
                        output_tokens,
                        response_model,
                        request_id,
                        response_text,
                        request,
                    ),
                )
        if unexpected:
            raise EpisodeStopped(
                "budget_exhausted",
                "Provider usage exceeded its reservation; further calls are blocked",
            )
        return actual

    def summary(self) -> dict[str, Any]:
        with self._transaction() as connection:
            cap, maximum, blocked, pricing_source = connection.execute(
                "SELECT cap,max_calls,blocked,pricing_source FROM budget WHERE id=1"
            ).fetchone()
            count, confirmed, outstanding = connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(actual),0),"
                "COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) FROM requests"
            ).fetchone()
            statuses = dict(
                connection.execute(
                    "SELECT status,COUNT(*) FROM requests GROUP BY status"
                )
            )
        return {
            "budget_usd": _usd(cap),
            "confirmed_cost_usd": _usd(confirmed),
            "reserved_uncertain_usd": _usd(outstanding),
            "accounted_cost_usd": _usd(confirmed + outstanding),
            "requests": count,
            "max_calls": maximum,
            "statuses": statuses,
            "blocked": bool(blocked),
            "pricing_source": pricing_source,
        }


class BudgetedClaudeModel:
    """Stateless text-only model with no retries and pre-admitted maximum cost."""

    _mock_mode = False

    def __init__(
        self,
        model_name: str,
        ledger: BudgetLedger,
        api_key: str | None = None,
        max_tokens: int = 192,
        client: Any | None = None,
    ):
        if model_name not in ANTHROPIC_MODELS:
            raise ValueError("Model has no verified Anthropic trial price")
        if type(max_tokens) is not int or not 64 <= max_tokens <= 256:
            raise ValueError("max_tokens must be an integer from 64-256")
        self._model_name, self.ledger, self.max_tokens = model_name, ledger, max_tokens
        if client is None:
            from anthropic import Anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for this real-model trial"
                )
            client = Anthropic(api_key=key, timeout=30, max_retries=0)
        self._client: Any = client

    def __str__(self) -> str:
        return f"ClaudeModel({self._model_name})"

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        if mock:
            raise ValueError(
                "Budgeted trials require an explicitly real or fake injected client"
            )
        try:
            start = prompt.rfind("\n{") + 1
            context, _ = json.JSONDecoder().raw_decode(prompt[start:])
            legal = {entry["id"]: entry["label"] for entry in context["legal_actions"]}
            if not legal or any(type(action) is not int for action in legal):
                raise ValueError("Invalid legal action context")
        except (ValueError, KeyError, TypeError):
            raise EpisodeStopped(
                "provider_error", "Invalid structured arena prompt; no request was made"
            ) from None
        combined = prompt + (system_instruction or "")
        reservation = self.ledger.reserve(self._model_name, combined, self.max_tokens)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
        }
        if self._model_name in {
            "claude-haiku-5-5",
            "claude-sonnet-5",
            "claude-sonnet-5-5",
        }:
            kwargs["output_config"] = {"effort": "low"}
        if self._model_name == "claude-sonnet-5-5":
            kwargs["thinking"] = {"type": "between_tools"}
        if system_instruction:
            kwargs["system"] = system_instruction
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            self.ledger.fail(reservation, type(exc).__name__)
            status = getattr(exc, "status_code", None)
            detail = f" ({status})" if type(status) is int else ""
            raise EpisodeStopped(
                "provider_error",
                f"Anthropic request failed: {type(exc).__name__}{detail}; no retry was made",
            ) from None
        try:
            raw = "".join(
                block.text for block in response.content if block.type == "text"
            )
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            request_id = getattr(response, "_request_id", None)
            actual = self.ledger.complete(
                reservation,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                response_model=response.model,
                request_id=request_id,
                response_text=raw,
            )
            text = _validate_model_response(raw)
            parse_legal_action(text, legal)
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise ValueError("Model exhausted the output limit")
        except EpisodeStopped:
            raise
        except Exception:
            self.ledger.fail(reservation, "InvalidProviderResponse")
            raise EpisodeStopped(
                "provider_error",
                "Provider response was incomplete or invalid; no retry was made",
            ) from None
        return GenerateResponse(
            main_response=text,
            usage=usage,
            metadata={
                "provider": "anthropic",
                "mock": False,
                "model": response.model,
                "request_id": request_id,
                "cost_usd": _usd(actual),
                "budget_request_id": reservation,
                "reserved_cost_usd": _usd(
                    (len(combined.encode("utf-8")) + 1024) * PRICES[self._model_name][0]
                    + self.max_tokens * PRICES[self._model_name][1]
                ),
            },
        )
