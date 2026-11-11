"""TypeSafe protocol, privacy, budget and replay checks use fake HTTP only."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from http.client import HTTPMessage
from pathlib import Path
import tempfile
import unittest
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from game_arena.openspiel.budgeted_anthropic import BudgetLedger
from game_arena.openspiel.harness import EpisodeStopped, harness
from game_arena.openspiel.model_generation import LLMModel
from game_arena.openspiel.jev import (
    ENDPOINT,
    JevModel,
    MODELS,
    PRICING_SOURCE,
    _NoRedirect,
)
from game_arena.openspiel.parsers import parse_legal_action
from game_arena.openspiel.replay import replay_episode
from game_arena.openspiel.utils import HarnessOptions


def prompt(actions=None, **extras):
    context = {
        "game": "avalon",
        "num_players": 6,
        "player": 2,
        "utility": "GENERAL_SUM",
        "observation_type": "observation_string",
        "observation": json.dumps({"role": "servant", "public_history": []}),
        "own_action_history": [
            {
                "action": 0,
                "label": "pass",
                "other_player_secret": "do-not-forward",  # pragma: allowlist secret -- test sentinel
            }
        ],
        "legal_actions": actions
        if actions is not None
        else [{"id": 0, "label": "reject"}, {"id": 1, "label": "approve"}],
        "full_state": {"all_private_roles": "do-not-forward"},
    }
    context.update(extras)
    return (
        "Play the game.\nGame instructions: Public claims may be false.\n"
        + json.dumps(context)
        + '\nRespond with "Summary:" and "Final Answer:".'
    )


def response(probabilities=None, choice="action_1", **extras):
    payload = {
        "model": "jev-1.14.0",
        "answers": {
            "action": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.7,
                "probabilities": probabilities or {"action_0": 0.1, "action_1": 0.9},
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 1000},
    }
    payload.update(extras)
    return payload


class FakeTransport:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else response()
        self.error = error
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return deepcopy(self.payload), "req-jev-test"


class ChoicePolicy:
    """Return exact legal distributions to exercise the true adapter+harness."""

    def __init__(self):
        self.calls = 0

    def __call__(self, request, timeout):
        self.calls += 1
        data = json.loads(request.data)
        options = data["questions"]["action"]["criteria"]
        observation = json.loads(data["state"]["observation"])
        chosen = "action_1" if observation["phase"] == "vote" else next(iter(options))
        probabilities = {key: float(key == chosen) for key in options}
        return response(probabilities=probabilities, choice=chosen), "fixture-request"


class JevTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jev-budget.sqlite"

    def ledger(self, **kwargs):
        return BudgetLedger(
            self.path,
            budget_usd=kwargs.pop("budget_usd", "0.10"),
            pricing_source=PRICING_SOURCE,
            **kwargs,
        )

    def model(self, transport=None, ledger=None, **kwargs):
        return JevModel(
            api_key="jev-unit-secret",  # pragma: allowlist secret -- synthetic test credential or sentinel only
            ledger=ledger or self.ledger(),
            transport=transport or FakeTransport(),
            mock_mode=False,
            **kwargs,
        )

    def test_request_is_exact_official_typed_choice_and_player_scoped(self):
        transport = FakeTransport()
        result = self.model(transport).generate_with_text_input(
            prompt(), "Respect game rules."
        )
        request, timeout = transport.calls[0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer jev-unit-secret")
        self.assertEqual(timeout, 20)
        body = json.loads(request.data)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(
            body["questions"]["action"]["criteria"],
            {"action_0": "reject", "action_1": "approve"},
        )
        self.assertEqual(body["state"]["player"], 2)
        self.assertEqual(
            body["state"]["observation"],
            json.dumps({"role": "servant", "public_history": []}),
        )
        self.assertEqual(
            body["state"]["game_instructions"], "Public claims may be false."
        )
        self.assertEqual(body["state"]["system_instruction"], "Respect game rules.")
        self.assertNotIn("do-not-forward", json.dumps(body))
        self.assertNotIn("Final Answer:", json.dumps(body))
        self.assertEqual(
            parse_legal_action(
                result.main_response, {0: "reject", 1: "approve"}
            ).action,
            1,
        )

    def test_preserves_real_json_probabilities_and_honest_adapter_envelope(self):
        payload = response()
        result = self.model(FakeTransport(payload)).generate_with_text_input(prompt())
        self.assertFalse(result.metadata["mock"])
        self.assertEqual(result.metadata["decision_source"], "classifier")
        self.assertEqual(result.metadata["summary_source"], "adapter")
        self.assertEqual(result.metadata["provider_response"], payload)
        self.assertEqual(
            result.metadata["probabilities"], {"action_0": 0.1, "action_1": 0.9}
        )
        self.assertEqual(result.metadata["confidence"], 0.7)
        self.assertIn("Adapter selected", result.main_response)
        self.assertIn("no explanation was generated", result.main_response)
        self.assertEqual(result.metadata["cost_usd"], "0.000004200")
        self.assertEqual(self.ledger().summary()["confirmed_cost_usd"], "0.000004200")
        self.assertEqual(result.usage["output_tokens"], 1000)
        self.assertNotIn(b"jev-unit-secret", self.path.read_bytes())

    def test_mock_mode_requires_no_key_ledger_or_network(self):
        transport = FakeTransport()
        with patch.dict("os.environ", {}, clear=True):
            model = JevModel(mock_mode=True, transport=transport)
            result = model.generate_with_text_input(prompt())
        self.assertTrue(result.metadata["mock"])
        self.assertEqual(result.metadata["decision_source"], "mock")
        self.assertIsNone(result.metadata["provider_response"])
        self.assertIsNone(result.metadata["confidence"])
        self.assertEqual(transport.calls, [])
        self.assertIsNone(model.ledger)
        self.assertEqual(result.usage["input_tokens"], 0)

    def test_global_mock_switch_is_respected(self):
        with (
            patch("game_arena.openspiel.model_generation.MODEL_API_MOCK_MODE", True),
            patch.dict("os.environ", {}, clear=True),
        ):
            self.assertTrue(JevModel()._mock_mode)

    def test_real_missing_key_and_unsupported_models_fail_without_http(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            JevModel(mock_mode=False)
        for name in ("claude-haiku-5-5", "jev-made-up", "https://other.example/jev"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                JevModel(name, mock_mode=True)
        with self.assertRaises(ValueError):
            JevModel(
                api_key="bad\nheader",  # pragma: allowlist secret -- invalid test credential
                mock_mode=False,
            )

    def test_auth_environment_aliases_and_default_private_ledger(self):
        for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
            with patch.dict(
                "os.environ",
                {name: "environment-key", "JEV_BUDGET_LEDGER": str(self.path)},
                clear=True,
            ):
                model = JevModel(mock_mode=False, transport=FakeTransport())
                assert model.ledger is not None
                self.assertEqual(model.ledger.summary()["budget_usd"], "0.100000000")
                self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_forced_single_action_is_local_and_never_claims_jev_confidence(self):
        transport = FakeTransport()
        ledger = self.ledger()
        result = self.model(transport, ledger).generate_with_text_input(
            prompt([{"id": 500, "label": "success card"}])
        )
        self.assertEqual(
            parse_legal_action(result.main_response, {500: "success card"}).action, 500
        )
        self.assertEqual(result.metadata["decision_source"], "forced_legal_action")
        self.assertIsNone(result.metadata["provider_response"])
        self.assertIsNone(result.metadata["confidence"])
        self.assertFalse(result.metadata["mock"])
        self.assertEqual(transport.calls, [])
        self.assertEqual(ledger.summary()["requests"], 0)

    def test_denied_budget_and_call_limit_happen_before_http(self):
        ledger = self.ledger(budget_usd="0.000001")
        transport = FakeTransport()
        with self.assertRaises(EpisodeStopped) as error:
            self.model(transport, ledger).generate_with_text_input(prompt())
        self.assertEqual(error.exception.code, "budget_exhausted")
        self.assertEqual(transport.calls, [])

    def test_ambiguous_failure_retains_reservation_and_never_retries(self):
        transport = FakeTransport(
            error=TimeoutError("jev-unit-secret credential must not leak")
        )
        ledger = self.ledger()
        with self.assertRaises(EpisodeStopped) as error:
            self.model(transport, ledger).generate_with_text_input(prompt())
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(len(transport.calls), 1)
        self.assertGreater(float(ledger.summary()["reserved_uncertain_usd"]), 0)

    def test_http_credit_key_and_redirect_errors_are_safe(self):
        for status, expected in (
            (401, "API key"),
            (402, "credits"),
            (307, "redirect"),
            (503, "failed"),
        ):
            with self.subTest(status=status):
                transport = FakeTransport(
                    error=HTTPError(
                        ENDPOINT,
                        status,
                        "jev-unit-secret",
                        HTTPMessage(),
                        io.BytesIO(b"secret"),
                    )
                )
                with self.assertRaises(EpisodeStopped) as error:
                    self.model(transport).generate_with_text_input(prompt())
                self.assertIn(expected, str(error.exception))
                self.assertNotIn("secret", str(error.exception))
                self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(
            _NoRedirect().redirect_request(
                Request(ENDPOINT),
                io.BytesIO(),
                307,
                "redirect",
                HTTPMessage(),
                "https://unrelated.example",
            )
        )

    def test_choice_boundaries_and_large_payload_reject_before_http(self):
        transport = FakeTransport()
        model = self.model(transport)
        malformed = [
            prompt([]),
            prompt([{"id": 0, "label": "a"}] * 2),
            prompt([{"id": True, "label": "a"}]),
            prompt([{"id": n, "label": str(n)} for n in range(256)]),
            prompt(observation="x" * 33000),
            "unstructured prompt",
        ]
        for text in malformed:
            with self.subTest(length=len(text)), self.assertRaises(EpisodeStopped):
                model.generate_with_text_input(text)
        self.assertEqual(transport.calls, [])
        assert model.ledger is not None
        self.assertEqual(model.ledger.summary()["requests"], 0)

    def test_exactly_255_choices_are_supported_with_full_distribution(self):
        actions = [{"id": n, "label": f"team-{n}"} for n in range(255)]
        probabilities = {f"action_{n}": float(n == 254) for n in range(255)}
        transport = FakeTransport(response(probabilities, "action_254"))
        result = self.model(transport).generate_with_text_input(prompt(actions))
        self.assertEqual(
            parse_legal_action(
                result.main_response, {n: str(n) for n in range(255)}
            ).action,
            254,
        )
        self.assertEqual(len(result.metadata["probabilities"]), 255)

    def test_invalid_decisions_stop_after_reconciling_confirmed_usage(self):
        mutations = [
            lambda p: p["answers"]["action"].update(choice="action_99"),
            lambda p: p["answers"]["action"].update(probabilities={"action_0": 1.0}),
            lambda p: p["answers"]["action"].update(
                probabilities={"action_0": 0.2, "action_1": 0.2}
            ),
            lambda p: p["answers"]["action"].update(
                probabilities={"action_0": float("nan"), "action_1": 0.9}
            ),
            lambda p: p["answers"]["action"].update(
                probabilities={"action_0": True, "action_1": 0.0}
            ),
            lambda p: p["answers"]["action"].update(confidence=1.1),
            lambda p: p["answers"]["action"].update(choice="action_0"),
            lambda p: p.update(model="jev-1.13.0"),
            lambda p: p["answers"]["action"].update(type="noul"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                value = response()
                mutate(value)
                transport = FakeTransport(value)
                with self.assertRaises(EpisodeStopped):
                    self.model(transport).generate_with_text_input(prompt())
                self.assertEqual(len(transport.calls), 1)
        self.assertLess(float(self.ledger().summary()["accounted_cost_usd"]), 0.10)

    def test_invalid_usage_is_reserved_and_unexpected_input_blocks_further_calls(self):
        transport = FakeTransport(
            response(usage={"input_tokens": "wrong", "output_tokens": 0})
        )
        ledger = self.ledger()
        with self.assertRaises(EpisodeStopped):
            self.model(transport, ledger).generate_with_text_input(prompt())
        self.assertGreater(float(ledger.summary()["reserved_uncertain_usd"]), 0)
        transport = FakeTransport(
            response(usage={"input_tokens": 999999, "output_tokens": 0})
        )
        with self.assertRaises(EpisodeStopped):
            self.model(transport, ledger).generate_with_text_input(prompt())
        self.assertTrue(ledger.summary()["blocked"])

    def test_version_aliases_record_resolved_pin(self):
        self.assertEqual(len(MODELS), 4)
        result = self.model(model_name="jev-latest").generate_with_text_input(prompt())
        self.assertEqual(result.metadata["model"], "jev-1.14.0")
        self.assertEqual(result.metadata["requested_model"], "jev-latest")

    def test_true_adapter_and_fake_http_replay_three_quest_avalon_prefix(self):
        ledger = self.ledger()
        policy = ChoicePolicy()
        models: list[LLMModel] = [self.model(policy, ledger) for _ in range(6)]
        record: dict[str, Any] = dict(
            harness(
                HarnessOptions(
                    type="PVP",
                    models=models,
                    game="avalon(players=6)",
                    input=None,
                    num_steps=1000,
                    seed=123,
                    stop_after_quests=3,
                    failure_policy="raise",
                )
            )
        )
        self.assertEqual(record["status"], "TRUNCATED")
        self.assertEqual(record["stop_reason"], "quest_limit")
        self.assertEqual(len(record["game_progress"]["quest_results"]), 3)
        self.assertTrue(replay_episode(record)["verified"])
        self.assertLess(policy.calls, record["decision_count"])
        self.assertEqual(ledger.summary()["requests"], policy.calls)
        self.assertLess(float(ledger.summary()["confirmed_cost_usd"]), 0.01)
