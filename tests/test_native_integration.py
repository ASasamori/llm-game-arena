"""Standalone environments pass before their separate mock-arena integration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

from game_arena.environments import registry
from game_arena.environments.cli import replay, simulate


PROJECT = Path(__file__).resolve().parents[1]


class AStandaloneEnvironmentIntegrationTests(unittest.TestCase):
    def test_every_player_bound_and_effective_configuration_roundtrips(self) -> None:
        for name, low, high in (
            ("one_night_werewolf", 3, 10),
            ("avalon", 5, 10),
            ("coup", 2, 6),
        ):
            for players in range(low, high + 1):
                variants = (
                    [{"starting_player": p, "max_turns": 1} for p in range(players)]
                    if name == "coup"
                    else [{"discussion_rounds": r} for r in range(4)]
                )
                for variant in variants:
                    with self.subTest(name=name, players=players, variant=variant):
                        game = registry.create_game(
                            name, {"players": players} | variant
                        )
                        spec = registry.canonical_spec(game)
                        loaded = registry.load_game(spec)
                        self.assertEqual(loaded.parameters, game.parameters)
                        self.assertEqual(loaded.name, name)
                        self.assertEqual(loaded.num_players(), players)
                        metadata = registry.describe(loaded)
                        self.assertEqual(metadata["backend"], "native")
                        self.assertEqual(metadata["num_players"], players)
                        self.assertTrue(metadata["supported"])
                        self.assertTrue(metadata["rules"])

    def test_aliases_boolean_parameters_and_role_decks_are_canonical(self) -> None:
        game = registry.load_game("werewolf(players=3,lone_wolf=false)")
        self.assertEqual(game.name, "one_night_werewolf")
        self.assertFalse(game.parameters["lone_wolf"])
        self.assertEqual(
            registry.load_game(registry.canonical_spec(game)).parameters,
            game.parameters,
        )
        self.assertEqual(
            registry.load_game("one_night_ultimate_werewolf").name, game.name
        )
        game = registry.load_game(
            'avalon(players=5,roles="merlin|percival|servant|assassin|morgana")'
        )
        self.assertEqual(
            registry.load_game(registry.canonical_spec(game)).parameters,
            game.parameters,
        )

    def test_malformed_specs_never_execute_code_or_ignore_invalid_parameters(
        self,
    ) -> None:
        invalid = [
            "unknown_game",
            "coup(3)",
            "coup(players=3,players=4)",
            "coup(**{'players':3})",
            "coup(players=1+2)",
            "coup(players=__import__('os').getpid())",
            "coup(players=3);exit()",
            "coup(players=3",
            "coup(players=3))",
            "coup(players=1)",
            "coup(players=true)",
            "coup(starting_player=-1)",
            "coup(max_turns=0)",
            "coup(extra=1)",
            "avalon(players=4)",
            "avalon(discussion_rounds=4)",
            "one_night_werewolf(players=11)",
            "one_night_werewolf(lone_wolf=1)",
        ]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                registry.load_game(spec)

    def test_seeded_native_traces_reproduce_all_player_counts_without_models(
        self,
    ) -> None:
        for name, low, high in (
            ("one_night_werewolf", 3, 10),
            ("avalon", 5, 10),
            ("coup", 2, 6),
        ):
            for players in range(low, high + 1):
                with self.subTest(name=name, players=players):
                    spec = f"{name}(players={players})"
                    first = simulate(spec, seed=53)
                    second = simulate(spec, seed=53)
                    self.assertEqual(first, second)
                    self.assertTrue(first["terminal"])
                    self.assertEqual(first["status"], "COMPLETED")
                    self.assertEqual(len(first["returns"]), players)
                    self.assertTrue(replay(first)["verified"])

    def test_cli_simulate_and_replay_in_fresh_processes_for_every_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for game in registry.NAMES:
                with self.subTest(game=game):
                    path = Path(directory) / (game + ".json")
                    generated = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "game_arena.environments",
                            "simulate",
                            "--game",
                            game,
                            "--seed",
                            "44",
                            "--output",
                            str(path),
                        ],
                        cwd=PROJECT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result = json.loads(generated.stdout)
                    self.assertTrue(result["verified"])
                    self.assertTrue(result["terminal"])
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    reproduced = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "game_arena.environments",
                            "replay",
                            str(path),
                        ],
                        cwd=PROJECT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertTrue(json.loads(reproduced.stdout)["verified"])

    def test_standalone_simulation_never_imports_openspiel_or_model_modules(
        self,
    ) -> None:
        script = """import builtins,json,sys
original_import=builtins.__import__
blocked=('pyspiel','open_spiel','openai','anthropic','google.genai','game_arena.openspiel','orchestrator')
def guarded(name,*args,**kwargs):
 if any(name==prefix or name.startswith(prefix+'.') for prefix in blocked):
  raise AssertionError('Standalone environment imported '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded
from game_arena.environments.cli import replay,simulate
for name in ('one_night_werewolf','avalon','coup'):
 result=simulate(name,seed=7)
 assert result['terminal'] and replay(result)['verified']
assert not any(any(name==p or name.startswith(p+'.') for p in blocked) for name in sys.modules)
print(json.dumps({'independent':True}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(json.loads(result.stdout)["independent"])

    def test_replay_rejects_changed_actions_observations_and_missing_tail(self) -> None:
        original = simulate("avalon", seed=5)
        player_index = next(
            i for i, event in enumerate(original["steps"]) if event["player"] >= 0
        )
        for corruption in ("action", "observation", "tail", "returns"):
            changed = deepcopy(original)
            if corruption == "action":
                changed["steps"][player_index]["action"] = 100000
            elif corruption == "observation":
                changed["steps"][player_index]["observation"] = (
                    "changed private evidence"
                )
            elif corruption == "tail":
                changed["steps"].pop()
            else:
                changed["returns"][0] += 1
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                replay(changed)

    def test_declared_transition_cap_retains_truncated_status_on_replay(self) -> None:
        record = simulate("avalon", max_steps=1)
        self.assertFalse(record["terminal"])
        self.assertEqual(record["status"], "TRUNCATED")
        self.assertTrue(replay(record)["verified"])


class BMockArenaIntegrationTests(unittest.TestCase):
    def execute(self, spec: str, players: int, **changes: Any) -> dict[str, Any]:
        from game_arena.openspiel.harness import harness
        from game_arena.openspiel.model_generation import MockModel

        options: Any = {
            "type": "PVP",
            "models": [MockModel(str(p)) for p in range(players)],
            "num_steps": 10000,
            "input": None,
            "game": spec,
            "seed": 104,
            "failure_policy": "raise",
        }
        options.update(changes)
        return dict(harness(options))

    def test_all_native_games_play_and_replay_through_existing_harness(self) -> None:
        from game_arena.openspiel import catalog
        from game_arena.openspiel.replay import replay_episode

        for spec, players in (("one_night_werewolf", 3), ("avalon", 5), ("coup", 3)):
            with self.subTest(game=spec):
                metadata = catalog.inspect_game(spec)
                self.assertTrue(metadata["supported"])
                self.assertEqual(metadata["backend"], "native")
                result = self.execute(spec, players)
                self.assertEqual(result["env"], "NATIVE")
                self.assertEqual(result["status"], "SUCCESS")
                self.assertTrue(result["terminal"])
                self.assertEqual(result["fallback_count"], 0)
                self.assertFalse(result["degraded"])
                self.assertEqual(result["manifest"]["environment_backend"], "native")
                self.assertEqual(
                    result["winner_players"],
                    [p for p, value in enumerate(result["returns"]) if value > 0],
                )
                self.assertTrue(replay_episode(result)["verified"])

    def test_model_prompts_have_exact_private_view_and_builtin_rules(self) -> None:
        from game_arena.openspiel import catalog
        from game_arena.openspiel.harness import _initial_state

        for spec, players in (("one_night_werewolf", 3), ("avalon", 5), ("coup", 3)):
            with self.subTest(game=spec):
                result = self.execute(spec, players)
                state = _initial_state(
                    catalog.load_game(result["game_spec"], seed=result["seed"])
                )
                for event in result["steps"]:
                    self.assertNotIn("state_before", event)
                    self.assertNotIn("state_after", event)
                    for decision in event["decisions"]:
                        prompt = decision["prompt"]
                        self.assertIn("Game instructions: ", prompt)
                        self.assertIn(result["manifest"]["rules"], prompt)
                        self.assertEqual(
                            decision["observation_type"], "observation_string"
                        )
                        self.assertEqual(
                            decision["observation"],
                            state.observation_string(decision["player"]),
                        )
                        payload = json.JSONDecoder().raw_decode(
                            prompt[prompt.index("{") :]
                        )[0]
                        self.assertEqual(
                            payload["observation"], decision["observation"]
                        )
                        self.assertNotIn("state", payload)
                        self.assertNotIn("chance_outcomes", payload)
                        self.assertNotIn("opponent_private_state", payload)
                    state.apply_action(event["actions"][0])

    def test_native_configuration_rejection_precedes_any_model_call(self) -> None:
        from unittest.mock import patch
        from game_arena.openspiel.model_generation import MockModel

        with patch.object(
            MockModel,
            "generate_with_text_input",
            side_effect=AssertionError("Model called"),
        ):
            for spec, players in (
                ("avalon(players=4)", 4),
                ("coup(max_turns=0)", 3),
                ("werewolf(lone_wolf=1)", 3),
            ):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    self.execute(spec, players)

    def test_native_alias_and_parameterized_game_reproduce(self) -> None:
        from game_arena.openspiel.replay import replay_episode

        result = self.execute(
            "werewolf(players=4,discussion_rounds=0,lone_wolf=false)", 4
        )
        self.assertEqual(result["game"], "one_night_werewolf")
        self.assertIn("lone_wolf=false", result["game_spec"])
        self.assertTrue(replay_episode(result)["verified"])

    def test_original_openspiel_tic_tac_toe_behavior_still_replays(self) -> None:
        from game_arena.openspiel.replay import replay_episode

        result = self.execute("tic_tac_toe", 2)
        self.assertEqual(result["env"], "OPENSPIEL")
        self.assertEqual(result["game"], "tic_tac_toe")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["steps"][0]["state_before"], "...\n...\n...")
        self.assertTrue(replay_episode(result)["verified"])


if __name__ == "__main__":
    unittest.main()
