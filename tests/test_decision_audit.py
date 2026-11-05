"""Selection mechanisms are explicit, verifiable and separate from model text."""

from copy import deepcopy
from typing import Any
import unittest

from game_arena.openspiel.decision_audit import describe_selection
from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.replay import ReplayError, replay_episode


class DecisionAuditTests(unittest.TestCase):
    def test_policy_classification_uses_each_seat_and_retains_original_evidence(self):
        decision = {
            "player": 1,
            "action": 2,
            "legal_actions": [1, 2],
            "summary": "Reported explanation",
            "response": "Original output",
            "response_metadata": {},
        }
        original = deepcopy(decision)
        manifest = {
            "run_type": "PVP",
            "model_configs": [{"mock": True}, {"mock": False}],
        }
        self.assertEqual(describe_selection(decision, manifest)["kind"], "model")
        manifest["model_configs"][1]["mock"] = True
        self.assertEqual(describe_selection(decision, manifest)["kind"], "mock")
        self.assertEqual(decision, original)
        for mode, kind in (("RAND", "random"), ("INPUT", "scripted")):
            manifest["run_type"] = mode
            self.assertEqual(describe_selection(decision, manifest)["kind"], kind)
        decision["fallback_reason"] = {"code": "parse_failed"}
        self.assertEqual(describe_selection(decision, manifest)["kind"], "fallback")

    def test_one_legal_action_does_not_imply_no_model_call(self):
        decision = {
            "player": 0,
            "action": 0,
            "legal_actions": [0],
            "response_metadata": {},
        }
        manifest = {"run_type": "PVP", "model_configs": [{"mock": False}]}
        self.assertEqual(describe_selection(decision, manifest)["kind"], "model")
        decision["response_metadata"] = {
            "decision_source": "forced_legal_action",
            "summary_source": "adapter",
        }
        self.assertEqual(describe_selection(decision, manifest)["kind"], "forced")
        decision["legal_actions"] = []
        self.assertEqual(describe_selection(decision, manifest)["kind"], "inactive")

    def test_classifier_summary_is_never_attributed_to_generated_reasoning(self):
        decision = {
            "player": 0,
            "action": 1,
            "legal_actions": [0, 1],
            "response_metadata": {
                "decision_source": "classifier",
                "summary_source": "adapter",
            },
        }
        manifest = {"run_type": "PVP", "model_configs": [{"mock": False}]}
        selection = describe_selection(decision, manifest)
        self.assertEqual(selection["kind"], "classifier")
        self.assertEqual(selection["summary_source"], "adapter")
        manifest["model_configs"][0]["mock"] = True
        self.assertEqual(describe_selection(decision, manifest)["kind"], "mock")

    def test_mock_and_random_traces_verify_and_reject_changed_mechanism(self):
        for game, players, mode in (
            ("tic_tac_toe", 2, "PVP"),
            ("matching_pennies_3p", 3, "PVP"),
            ("tic_tac_toe", 2, "RAND"),
        ):
            with self.subTest(game=game, mode=mode):
                options: Any = {
                    "type": mode,
                    "models": [
                        MockModel(str(p))
                        for p in range(players if mode == "PVP" else 1)
                    ],
                    "game": game,
                    "num_steps": 10,
                    "input": None,
                    "seed": 21,
                    "failure_policy": "raise",
                }
                record: dict[str, Any] = dict(harness(options))
                self.assertTrue(replay_episode(record)["verified"])
                decisions = [
                    decision
                    for event in record["steps"]
                    for decision in event["decisions"]
                ]
                self.assertEqual(
                    {decision["selection"]["kind"] for decision in decisions},
                    {"mock", "random"} if mode == "RAND" else {"mock"},
                )
                corrupted = deepcopy(record)
                corrupted["steps"][0]["decisions"][0]["selection"]["kind"] = "invented"
                with self.assertRaisesRegex(
                    ReplayError, "selection mechanism mismatch"
                ):
                    replay_episode(corrupted)
                for event in record["steps"]:
                    event.pop("review_state_before", None)
                    event.pop("review_state_after", None)
                    for decision in event["decisions"]:
                        decision.pop("selection", None)
                record["manifest"].pop("recorded_features", None)
                self.assertTrue(replay_episode(record)["verified"])

    def test_review_state_is_replayed_without_changing_player_evidence(self):
        options: Any = {
            "type": "PVP",
            "models": [MockModel("a"), MockModel("b")],
            "game": "tic_tac_toe",
            "num_steps": 3,
            "input": None,
            "seed": 31,
        }
        record: dict[str, Any] = dict(harness(options))
        self.assertEqual(record["steps"][0]["decisions"][0]["observation"], "")
        self.assertEqual(record["steps"][0]["state_before"], "...\n...\n...")
        for field in ("review_state_before", "review_state_after"):
            corrupted = deepcopy(record)
            corrupted["steps"][0][field]["node"]["kind"] = "invented"
            with self.assertRaisesRegex(ReplayError, "review state mismatch"):
                replay_episode(corrupted)
        for field in ("review_state_before", "selection"):
            corrupted = deepcopy(record)
            target = (
                corrupted["steps"][0]
                if field != "selection"
                else corrupted["steps"][0]["decisions"][0]
            )
            target.pop(field)
            with self.assertRaisesRegex(ReplayError, "missing required"):
                replay_episode(corrupted)


if __name__ == "__main__":
    unittest.main()
