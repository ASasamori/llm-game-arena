"""Official maturity labels constrain new admission without changing game rules."""

from typing import Any
import unittest
from unittest.mock import patch

from game_arena.openspiel import catalog
from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.replay import replay_episode


class UpstreamAdmissionTests(unittest.TestCase):
    def test_green_and_orange_games_retain_their_distinct_status(self):
        entries = {entry["short_name"]: entry for entry in catalog.list_games()}
        for name, status in (
            ("connect_four", "thoroughly_tested"),
            ("chess", "thoroughly_tested"),
            ("hex", "lightly_tested"),
            ("2048", "lightly_tested"),
        ):
            self.assertEqual(entries[name]["upstream_status"]["status"], status)
            self.assertTrue(entries[name]["supported"])
        for name in ("quoridor", "rbc"):
            self.assertEqual(entries[name]["upstream_status"]["status"], "known_issues")
            self.assertFalse(entries[name]["supported"])
            self.assertIn(
                "known issues", " ".join(entries[name]["unsupported_reasons"])
            )

    def test_known_issue_games_and_nested_wrappers_are_rejected_before_agents(self):
        from orchestrator.jobs import validate_run_spec

        class GuardModel(MockModel):
            def generate_with_text_input(self, *args, **kwargs):
                raise AssertionError(
                    "An excluded game must not request an agent action"
                )

        for spec in (
            "quoridor",
            "rbc",
            "misere(game=quoridor())",
            "cached_tree(game=rbc())",
        ):
            with self.subTest(spec=spec):
                info = catalog.inspect_game(spec)
                self.assertFalse(info["supported"])
                with self.assertRaisesRegex(
                    catalog.UnsupportedGameError, "known issues"
                ):
                    catalog.load_game(spec)
                options: Any = {
                    "type": "PVP",
                    "models": [GuardModel("a"), GuardModel("b")],
                    "game": spec,
                    "num_steps": 1,
                    "seed": 0,
                    "input": None,
                }
                with self.assertRaisesRegex(
                    catalog.UnsupportedGameError, "known issues"
                ):
                    harness(options)
                with self.assertRaisesRegex(ValueError, "known issues"):
                    validate_run_spec(
                        run_type="PVP",
                        num_episodes=1,
                        num_steps=1,
                        model_names=["MOCK:a", "MOCK:b"],
                        input_actions=None,
                        env=spec,
                    )

    def test_historical_known_issue_traces_can_still_verify_recorded_evidence(self):
        # Build a bounded historical fixture using the unchanged upstream rules.
        # No new API run or provider is admitted through the current policy.
        original_load = catalog.load_game
        game = original_load("quoridor", allow_known_issues_for_replay=True)
        info = catalog._metadata(game.get_type())
        info.update(supported=True, unsupported_reasons=[])
        with (
            patch.object(catalog, "inspect_game", return_value=info),
            patch.object(
                catalog,
                "load_game",
                side_effect=lambda spec, seed=0: original_load(
                    spec, seed, allow_known_issues_for_replay=True
                ),
            ),
        ):
            options: Any = {
                "type": "PVP",
                "models": [MockModel("a"), MockModel("b")],
                "game": "quoridor",
                "num_steps": 1,
                "seed": 0,
                "input": None,
            }
            record = dict(harness(options))
        self.assertTrue(replay_episode(record)["verified"])
        self.assertFalse(catalog.inspect_game("quoridor")["supported"])


if __name__ == "__main__":
    unittest.main()
