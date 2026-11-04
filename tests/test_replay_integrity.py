"""Replay must reject internally inconsistent evaluator evidence and assumptions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import unittest

from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import MockModel
from game_arena.openspiel.replay import ReplayError, replay_episode
from game_arena.openspiel.utils import HarnessOptions


def _episode(
    game: str = "tic_tac_toe", players: int = 2, **extra: Any
) -> dict[str, Any]:
    options: Any = HarnessOptions(
        type="PVP",
        models=[MockModel(name=f"mock-{player}") for player in range(players)],
        num_steps=100,
        input=None,
        game=game,
        seed=71,
    )
    options.update(extra)
    return dict(harness(options))


def _set_path(record: dict[str, Any], path: tuple[Any, ...], value: Any) -> None:
    target: Any = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


class ReplayIntegrityTests(unittest.TestCase):
    def test_model_assignments_winner_and_derived_evidence_are_verified(self) -> None:
        original = _episode()
        self.assertTrue(replay_episode(original)["verified"])
        mutations = [
            (("winner",), "wrong-model"),
            (("winner_players",), [999]),
            (("player_agents",), ["wrong-a", "wrong-b"]),
            (("agent",), ["wrong-a", "wrong-b"]),
            (("seed",), 999),
            (("episode_id",), "00000000-0000-0000-0000-000000000000"),
            (("arena_type",), "RAND"),
            (("game",), "chess"),
            (("game_spec",), "chess()"),
            (("decision_count",), 999),
            (("fallback_count",), 1),
            (("degraded",), True),
            (("action",), []),
            (("summary",), []),
            (("steps", 0, "decisions", 0, "agent"), "wrong-model"),
            (("steps", 0, "decisions", 0, "legal_action_strings"), ["wrong-label"]),
            (("steps", 0, "decisions", 0, "summary"), "wrong-summary"),
            (("steps", 0, "action_strings"), ["wrong-label"]),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                record = deepcopy(original)
                _set_path(record, path, value)
                with self.assertRaises(ReplayError):
                    replay_episode(record)

    def test_model_prompt_cannot_be_removed_to_bypass_response_verification(
        self,
    ) -> None:
        record = _episode()
        record["steps"][0]["decisions"][0]["prompt"] = None
        with self.assertRaises(ReplayError):
            replay_episode(record)

    def test_recorded_population_must_match_the_configured_assumption(self) -> None:
        original = _episode(
            "mfg_crowd_modelling(horizon=2)", 1, mean_field_distribution="uniform"
        )
        self.assertTrue(replay_episode(original)["verified"])
        record = deepcopy(original)
        record["manifest"]["mean_field_distribution"] = {"wrong_support": 123.0}
        with self.assertRaises(ReplayError):
            replay_episode(record)
        record = deepcopy(original)
        event = next(
            event for event in record["steps"] if event["kind"] == "mean_field"
        )
        event["distribution"] = [1.0] + [0.0] * (len(event["distribution"]) - 1)
        with self.assertRaises(ReplayError):
            replay_episode(record)

    def test_chance_labels_and_player_decisions_cannot_be_fabricated(self) -> None:
        original = _episode("kuhn_poker")
        record = deepcopy(original)
        record["steps"][0]["action_strings"] = ["wrong-label"]
        with self.assertRaises(ReplayError):
            replay_episode(record)
        record = deepcopy(original)
        record["steps"][0]["decisions"] = [{"player": 0, "agent": "fake"}]
        with self.assertRaises(ReplayError):
            replay_episode(record)


if __name__ == "__main__":
    unittest.main()
