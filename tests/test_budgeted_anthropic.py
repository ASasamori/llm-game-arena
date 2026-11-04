"""Budget guarantees use fake clients only; no provider credentials or traffic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from game_arena.openspiel.budgeted_anthropic import (
    ANTHROPIC_MODELS,
    BudgetLedger,
    BudgetedClaudeModel,
    DEFAULT_GAME_SYSTEM,
    DEFAULT_JSON_SYSTEM,
    DEFAULT_TOOL_SYSTEM,
    JEV_MODELS,
    JEV_PRICING_SOURCE,
    PRICING_SOURCE,
    PRICES,
)
from game_arena.openspiel.harness import EpisodeStopped
from game_arena.openspiel.parsers import parse_legal_action

PROMPT = (
    "Arena instruction\n"
    + json.dumps(
        {"legal_actions": [{"id": 0, "label": "approve"}, {"id": 1, "label": "reject"}]}
    )
    + "\nRespond with Summary and Final Answer."
)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response or SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text", text="Summary: Approve this team.\nFinal Answer: 0"
                )
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            model="provider-resolved-model",
            _request_id="req-test",
            stop_reason="end_turn",
        )
        self.error = error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def tool_client(
    action: Any = 1, summary: Any = "Reject this unsuitable team."
) -> FakeClient:
    client = FakeClient()
    client.response.content = [
        SimpleNamespace(
            type="tool_use",
            id="toolu-test",
            name="choose_action",
            input={"action": action, "summary": summary},
        )
    ]
    client.response.stop_reason = "tool_use"
    return client


class BudgetedAnthropicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.sqlite"

    def ledger(self, **kwargs):
        return BudgetLedger(self.path, **kwargs)

    def model(self, ledger=None, client=None, model_name="claude-haiku-5-5", **kwargs):
        return BudgetedClaudeModel(
            model_name, ledger or self.ledger(), client=client or FakeClient(), **kwargs
        )

    def test_preflight_denial_makes_no_http_request(self):
        ledger = self.ledger(budget_usd="0.000001")
        client = FakeClient()
        with self.assertRaises(EpisodeStopped) as caught:
            self.model(ledger, client).generate_with_text_input(PROMPT)
        self.assertEqual(caught.exception.code, "budget_exhausted")
        self.assertEqual(client.calls, [])
        self.assertEqual(ledger.summary()["requests"], 0)

    def test_confirmed_usage_reconciles_and_keeps_provider_evidence(self):
        ledger = self.ledger()
        response = self.model(ledger).generate_with_text_input(PROMPT)
        self.assertEqual(response.usage, {"input_tokens": 100, "output_tokens": 20})
        self.assertEqual(response.metadata["model"], "provider-resolved-model")
        self.assertEqual(response.metadata["request_id"], "req-test")
        self.assertEqual(response.metadata["cost_usd"], "0.000020000")
        self.assertGreater(float(response.metadata["reserved_cost_usd"]), 0.000020)
        self.assertEqual(ledger.summary()["confirmed_cost_usd"], "0.000020000")
        self.assertEqual(ledger.summary()["reserved_uncertain_usd"], "0.000000000")
        with sqlite3.connect(self.path) as connection:
            raw, status = connection.execute(
                "SELECT response_text,status FROM requests"
            ).fetchone()
        self.assertIn("Final Answer: 0", raw)
        self.assertEqual(status, "completed")

    def test_default_system_is_concise_and_recorded_with_its_input_bound(self):
        ledger = self.ledger()
        client = FakeClient()
        response = self.model(ledger, client).generate_with_text_input(PROMPT)
        self.assertEqual(client.calls[0]["system"], DEFAULT_GAME_SYSTEM)
        self.assertIn("exactly two lines", DEFAULT_GAME_SYSTEM)
        self.assertIn("at most 12 words", DEFAULT_GAME_SYSTEM)
        self.assertEqual(response.metadata["system_instruction"], DEFAULT_GAME_SYSTEM)
        with sqlite3.connect(self.path) as connection:
            bound, prompt_hash = connection.execute(
                "SELECT input_bound,prompt_sha256 FROM requests"
            ).fetchone()
        self.assertEqual(bound, len((PROMPT + DEFAULT_GAME_SYSTEM).encode()) + 1024)
        import hashlib

        self.assertEqual(
            prompt_hash,
            hashlib.sha256((PROMPT + DEFAULT_GAME_SYSTEM).encode()).hexdigest(),
        )

    def test_explicit_system_is_preserved_and_counted_instead_of_default(self):
        ledger = self.ledger()
        client = FakeClient()
        custom = "Custom game instruction with unicode: café."
        response = self.model(ledger, client).generate_with_text_input(
            PROMPT, system_instruction=custom
        )
        self.assertEqual(client.calls[0]["system"], custom)
        self.assertEqual(response.metadata["system_instruction"], custom)
        with sqlite3.connect(self.path) as connection:
            bound = connection.execute("SELECT input_bound FROM requests").fetchone()[0]
        self.assertEqual(bound, len((PROMPT + custom).encode()) + 1024)

    def test_supported_models_force_one_strict_action_tool(self):
        for model_name in ANTHROPIC_MODELS - {"claude-sonnet-5-5"}:
            client = tool_client()
            response = self.model(
                client=client, model_name=model_name, action_protocol="tool"
            ).generate_with_text_input(PROMPT)
            arguments = client.calls[0]
            self.assertEqual(
                arguments["tool_choice"], {"type": "tool", "name": "choose_action"}
            )
            self.assertEqual(arguments["system"], DEFAULT_TOOL_SYSTEM)
            tool = arguments["tools"][0]
            self.assertTrue(tool["strict"])
            self.assertEqual(tool["name"], "choose_action")
            schema = tool["input_schema"]
            self.assertEqual(list(schema["properties"]), ["action", "summary"])
            self.assertEqual(schema["properties"]["action"]["enum"], [0, 1])
            self.assertEqual(schema["required"], ["action", "summary"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(
                parse_legal_action(
                    response.main_response, {0: "approve", 1: "reject"}
                ).action,
                1,
            )
            self.assertEqual(response.metadata["action_protocol"], "tool")
            self.assertEqual(response.metadata["response_protocol"], "tool_use")
            self.assertEqual(
                response.metadata["system_instruction"], DEFAULT_TOOL_SYSTEM
            )
            self.assertEqual(response.metadata["action_schema"], schema)
            self.assertEqual(
                response.metadata["provider_response"]["content"][0]["input"],
                {"action": 1, "summary": "Reject this unsuitable team."},
            )

    def test_sonnet55_uses_json_schema_without_unsupported_forced_tools(self):
        client = FakeClient()
        raw = json.dumps({"action": 0, "summary": "Approve the proposed team."})
        client.response.content[0].text = raw
        response = self.model(
            client=client, model_name="claude-sonnet-5-5", action_protocol="tool"
        ).generate_with_text_input(PROMPT)
        arguments = client.calls[0]
        self.assertNotIn("tools", arguments)
        self.assertNotIn("tool_choice", arguments)
        self.assertEqual(arguments["thinking"], {"type": "between_tools"})
        self.assertEqual(arguments["output_config"]["effort"], "low")
        self.assertEqual(arguments["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(arguments["system"], DEFAULT_JSON_SYSTEM)
        self.assertEqual(response.metadata["response_protocol"], "json_schema")
        self.assertEqual(
            response.metadata["action_schema"],
            arguments["output_config"]["format"]["schema"],
        )
        self.assertEqual(
            response.metadata["provider_response"]["content"],
            [{"type": "text", "text": raw}],
        )
        self.assertEqual(
            parse_legal_action(
                response.main_response, {0: "approve", 1: "reject"}
            ).action,
            0,
        )

    def test_structured_reservation_includes_tools_schema_and_actual_system(self):
        client = tool_client()
        response = self.model(
            client=client, action_protocol="tool"
        ).generate_with_text_input(PROMPT)
        body = json.dumps(
            client.calls[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        bound = len(body.encode()) + 1024
        with sqlite3.connect(self.path) as connection:
            saved_bound, reserved = connection.execute(
                "SELECT input_bound,reserved FROM requests"
            ).fetchone()
        self.assertEqual(saved_bound, bound)
        expected = (
            bound * PRICES["claude-haiku-5-5"][0] + 192 * PRICES["claude-haiku-5-5"][1]
        )
        self.assertEqual(reserved, expected)
        self.assertEqual(
            float(response.metadata["reserved_cost_usd"]), expected / 1_000_000_000
        )
        self.assertEqual(
            self.ledger().summary()["reserved_uncertain_usd"], "0.000000000"
        )

    def test_invalid_or_truncated_tool_results_are_billed_once_without_retry(self):
        clients = [tool_client(action=999), tool_client(summary=None), tool_client()]
        clients[2].response.stop_reason = "max_tokens"
        missing = tool_client()
        missing.response.content = []
        clients.append(missing)
        multiple = tool_client()
        multiple.response.content *= 2
        clients.append(multiple)
        extra = tool_client()
        extra.response.content[0].input["extra"] = "unlisted"
        clients.append(extra)
        for client in clients:
            with self.assertRaises(EpisodeStopped) as caught:
                self.model(
                    client=client, action_protocol="tool"
                ).generate_with_text_input(PROMPT)
            self.assertEqual(caught.exception.code, "provider_error")
            self.assertEqual(len(client.calls), 1)
        summary = self.ledger().summary()
        self.assertEqual(summary["requests"], len(clients))
        self.assertEqual(summary["statuses"], {"completed": len(clients)})
        self.assertEqual(summary["reserved_uncertain_usd"], "0.000000000")

    def test_unsupported_action_protocol_is_rejected_without_requests(self):
        client = tool_client()
        with self.assertRaises(ValueError):
            self.model(client=client, action_protocol="other")
        self.assertEqual(client.calls, [])

    def test_transport_failure_retains_reservation_and_never_retries(self):
        ledger = self.ledger()
        client = FakeClient(error=TimeoutError("this secret must not be echoed"))
        with self.assertRaises(EpisodeStopped) as caught:
            self.model(ledger, client).generate_with_text_input(PROMPT)
        self.assertEqual(caught.exception.code, "provider_error")
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(len(client.calls), 1)
        result = BudgetLedger(self.path).summary()
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["statuses"], {"uncertain": 1})
        self.assertGreater(float(result["reserved_uncertain_usd"]), 0)

    def test_all_seats_share_persisted_limit(self):
        ledger = self.ledger(budget_usd="0.0003")
        client = FakeClient(error=TimeoutError())
        model = self.model(ledger, client)
        with self.assertRaises(EpisodeStopped):
            model.generate_with_text_input(PROMPT)
        other_ledger = BudgetLedger(self.path, budget_usd="0.0003")
        other_client = FakeClient()
        with self.assertRaises(EpisodeStopped) as caught:
            self.model(other_ledger, other_client).generate_with_text_input(PROMPT)
        self.assertEqual(caught.exception.code, "budget_exhausted")
        self.assertEqual(other_client.calls, [])

    def test_concurrent_reservations_cannot_exceed_cap(self):
        ledger = self.ledger(budget_usd="0.0003")

        def reserve(_):
            try:
                ledger.reserve("claude-haiku-5-5", PROMPT, 192)
                return True
            except EpisodeStopped:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            admitted = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(admitted), 1)
        self.assertLessEqual(float(ledger.summary()["accounted_cost_usd"]), 0.0003)

    def test_call_count_is_a_preflight_limit(self):
        ledger = self.ledger(max_calls=1)
        client = FakeClient()
        model = self.model(ledger, client)
        model.generate_with_text_input(PROMPT)
        with self.assertRaises(EpisodeStopped) as caught:
            model.generate_with_text_input(PROMPT)
        self.assertEqual(caught.exception.code, "call_limit")
        self.assertEqual(len(client.calls), 1)

    def test_episode_claim_is_once_even_after_reopening(self):
        self.assertTrue(self.ledger().claim_episode_once("episode-a"))
        self.assertFalse(self.ledger().claim_episode_once("episode-a"))
        self.assertTrue(self.ledger().claim_episode_once("episode-b"))

    def test_usage_overrun_blocks_all_further_calls(self):
        ledger = self.ledger()
        client = FakeClient()
        client.response.usage.output_tokens = 9999
        with self.assertRaises(EpisodeStopped):
            self.model(ledger, client).generate_with_text_input(PROMPT)
        self.assertTrue(ledger.summary()["blocked"])
        other = FakeClient()
        with self.assertRaises(EpisodeStopped):
            self.model(ledger, other).generate_with_text_input(PROMPT)
        self.assertEqual(other.calls, [])
        self.assertGreater(float(ledger.summary()["reserved_uncertain_usd"]), 0)

    def test_thinking_controls_and_output_cap_for_each_model(self):
        for model_name in ANTHROPIC_MODELS:
            client = FakeClient()
            self.model(client=client, model_name=model_name).generate_with_text_input(
                PROMPT
            )
            arguments = client.calls[0]
            self.assertEqual(arguments["max_tokens"], 192)
            self.assertNotIn("tools", arguments)
            self.assertNotIn("temperature", arguments)
            expected = (
                "between_tools" if model_name == "claude-sonnet-5-5" else "disabled"
            )
            self.assertEqual(arguments["thinking"], {"type": expected})
            if model_name in {
                "claude-haiku-5-5",
                "claude-sonnet-5",
                "claude-sonnet-5-5",
            }:
                self.assertEqual(arguments["output_config"], {"effort": "low"})

    def test_invalid_response_is_accounted_then_controlled_stop(self):
        for text in ("Summary: Wrong.\nFinal Answer: 999", "No final answer."):
            client = FakeClient()
            client.response.content[0].text = text
            with self.assertRaises(EpisodeStopped) as caught:
                self.model(client=client).generate_with_text_input(PROMPT)
            self.assertEqual(caught.exception.code, "provider_error")
            self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.ledger().summary()["statuses"], {"completed": 2})

    def test_truncated_response_and_bad_prompt_never_retry(self):
        client = FakeClient()
        client.response.stop_reason = "max_tokens"
        with self.assertRaises(EpisodeStopped):
            self.model(client=client).generate_with_text_input(PROMPT)
        with self.assertRaises(EpisodeStopped):
            self.model(client=client).generate_with_text_input("unstructured prompt")
        self.assertEqual(len(client.calls), 1)

    def test_limits_and_immutable_ledger_configuration(self):
        for budget in ("0", "-1", "1.01", "NaN", "Infinity", "bad"):
            with self.assertRaises(ValueError):
                self.ledger(budget_usd=budget)
        for count in (True, 0, 1001):
            with self.assertRaises(ValueError):
                self.ledger(max_calls=count)
        ledger = self.ledger()
        with self.assertRaises(ValueError):
            BudgetLedger(self.path, budget_usd="0.8")
        for tokens in (True, 63, 257):
            with self.assertRaises(ValueError):
                self.model(ledger, max_tokens=tokens)
        with self.assertRaises(ValueError):
            self.model(ledger, model_name="claude-opus-5")

    def test_credentials_are_not_saved_and_sdk_retry_is_disabled(self):
        key = "test-private-credential-not-real"
        fake = FakeClient()
        with patch("anthropic.Anthropic", return_value=fake) as factory:
            model = BudgetedClaudeModel("claude-haiku-5-5", self.ledger(), api_key=key)
            model.generate_with_text_input(PROMPT)
        factory.assert_called_once_with(api_key=key, timeout=30, max_retries=0)
        self.assertNotIn(key.encode(), self.path.read_bytes())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_jev_pricing_charges_input_and_allows_large_free_outputs(self):
        ledger = self.ledger(pricing_source=JEV_PRICING_SOURCE)
        for model in JEV_MODELS:
            request = ledger.reserve(model, PROMPT, 64)
            cost = ledger.complete(
                request,
                input_tokens=100,
                output_tokens=4000,
                response_model=model,
                request_id="jev-request",
                response_text="probabilities",
            )
            self.assertEqual(cost, 4200)
        summary = ledger.summary()
        self.assertEqual(summary["confirmed_cost_usd"], "0.000016800")
        self.assertEqual(summary["reserved_uncertain_usd"], "0.000000000")
        self.assertEqual(summary["pricing_source"], JEV_PRICING_SOURCE)
        self.assertFalse(summary["blocked"])

    def test_provider_price_source_and_schedule_cannot_be_changed_on_reopen(self):
        ledger = self.ledger(pricing_source=JEV_PRICING_SOURCE)
        self.assertTrue(ledger.claim_episode_once("jev-trial"))
        with self.assertRaises(ValueError):
            self.ledger()
        with patch.dict(PRICES, {"jev-1.14.0": (43, 0)}), self.assertRaises(ValueError):
            self.ledger(pricing_source=JEV_PRICING_SOURCE)
        reopened = self.ledger(pricing_source=JEV_PRICING_SOURCE)
        self.assertFalse(reopened.claim_episode_once("jev-trial"))
        self.assertEqual(reopened.summary()["pricing_source"], JEV_PRICING_SOURCE)
        with self.assertRaises(ValueError):
            reopened.reserve("claude-haiku-5-5", PROMPT, 192)
        with self.assertRaises(ValueError):
            self.ledger(pricing_source="https://unverified.example/prices")

    def test_jev_models_cannot_be_sent_to_anthropic_client(self):
        ledger = self.ledger(pricing_source=JEV_PRICING_SOURCE)
        client = FakeClient()
        for model in JEV_MODELS:
            with self.assertRaises(ValueError):
                BudgetedClaudeModel(model, ledger, client=client)
        self.assertEqual(client.calls, [])

    def test_existing_pre_pricing_ledgers_migrate_as_anthropic_without_losing_reservations(
        self,
    ):
        ledger = self.ledger()
        ledger.reserve("claude-haiku-5-5", PROMPT, 192)
        original = ledger.summary()
        with sqlite3.connect(self.path) as connection:
            connection.execute("ALTER TABLE budget DROP COLUMN pricing_source")
            connection.execute("ALTER TABLE budget DROP COLUMN prices_json")
        with self.assertRaises(ValueError):
            self.ledger(pricing_source=JEV_PRICING_SOURCE)
        migrated = self.ledger()
        self.assertEqual(migrated.summary(), original)
        self.assertEqual(migrated.summary()["pricing_source"], PRICING_SOURCE)


if __name__ == "__main__":
    unittest.main()
