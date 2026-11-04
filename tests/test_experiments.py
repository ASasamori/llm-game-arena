"""Offline planning and persistence tests; never create provider clients."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from game_arena.openspiel.cli import main
from game_arena.openspiel.datasets import DatasetStore, export_dataset
from game_arena.openspiel.experiments import (
    ExperimentPlan,
    PlanConfig,
    derive_seed,
    execute_mock_plan,
)


class ExperimentTests(unittest.TestCase):
    def plan(self, **kwargs) -> ExperimentPlan:
        return ExperimentPlan(PlanConfig(games=("tic_tac_toe",), **kwargs))

    def record(self, job) -> dict:
        return {
            "episode_id": job.episode_id,
            "game": job.game,
            "status": "SUCCESS",
            "seed": job.seed,
            "returns": [1, -1],
            "terminal": True,
            "num_steps": 1,
            "steps": [
                {
                    "index": 0,
                    "kind": "decision",
                    "players": [0],
                    "actions": [0],
                    "decisions": [{"player": 0, "action": 0}],
                    "returns": [1, -1],
                }
            ],
        }

    def test_stable_seeds_and_seat_rotation(self) -> None:
        first = self.plan(model_names=("a", "b"), episodes_per_matchup=4, seed=31)
        second = self.plan(model_names=("a", "b"), episodes_per_matchup=4, seed=31)
        jobs = list(first.jobs())
        self.assertEqual(jobs, list(second.jobs()))
        self.assertEqual(jobs[0].models, ("a", "b"))
        self.assertEqual(jobs[1].models, ("b", "a"))
        self.assertEqual(len({job.seed for job in jobs}), 4)
        self.assertNotEqual(derive_seed(31, "chance"), derive_seed(31, "fallback"))

    def test_n_player_and_solo_plans(self) -> None:
        solo = ExperimentPlan(PlanConfig(games=("catch",)))
        multiplayer = ExperimentPlan(
            PlanConfig(
                games=("matching_pennies_3p",),
                model_names=("a", "b", "c", "d"),
                episodes_per_matchup=3,
            )
        )
        self.assertEqual(len(next(solo.jobs()).models), 1)
        self.assertEqual(multiplayer.episode_count, 12)
        self.assertEqual(len(next(multiplayer.jobs()).models), 3)

    def test_plan_is_lazy_for_large_episode_counts(self) -> None:
        plan = self.plan(episodes_per_matchup=10**9)
        self.assertEqual(plan.episode_count, 10**9)
        self.assertEqual(next(plan.jobs()).ordinal, 0)
        self.assertEqual(
            next(plan.jobs(shard_index=999_999_999, shard_count=10**9)).ordinal,
            999_999_999,
        )

    def test_shards_partition_identity_without_changing_seeds(self) -> None:
        plan = self.plan(episodes_per_matchup=17)
        entire = list(plan.jobs())
        parts = [
            list(plan.jobs(shard_index=index, shard_count=4)) for index in range(4)
        ]
        combined = sorted(
            (job for part in parts for job in part), key=lambda job: job.ordinal
        )
        self.assertEqual(entire, combined)
        self.assertEqual(len({job.episode_id for job in combined}), 17)

    def test_invalid_configuration_is_rejected(self) -> None:
        for config in (
            {"episodes_per_matchup": 0},
            {"max_steps": 0},
            {"seed": -1},
            {"model_names": ("a", "a")},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.plan(**config)
        with self.assertRaises(ValueError):
            ExperimentPlan(
                PlanConfig(games=("matching_pennies_3p",), model_names=("a", "b"))
            )
        with self.assertRaises(ValueError):
            list(self.plan().jobs(shard_count=0))

    def test_atomic_artifact_resume_checks_job_and_content(self) -> None:
        plan = self.plan()
        job = next(plan.jobs())
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(plan.manifest)
            self.assertFalse(
                store.is_complete(job.episode_id, expected_job=job.as_dict())
            )
            store.write_episode(self.record(job), job=job.as_dict())
            self.assertTrue(
                store.is_complete(job.episode_id, expected_job=job.as_dict())
            )
            loaded = store.read_episode(job.episode_id)
            self.assertEqual(loaded["steps"][0]["actions"], [0])
            wrong = job.as_dict()
            wrong["seed"] += 1
            with self.assertRaisesRegex(ValueError, "job mismatch"):
                store.is_complete(job.episode_id, expected_job=wrong)
            steps = Path(directory) / loaded["steps_file"]
            steps.write_text(steps.read_text() + "{}\n")
            with self.assertRaises(ValueError):
                store.is_complete(job.episode_id, expected_job=job.as_dict())

    def test_corrupt_summary_is_not_silently_resumed(self) -> None:
        plan = self.plan()
        job = next(plan.jobs())
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(plan.manifest)
            path = store.write_episode(self.record(job), job=job.as_dict())
            data = json.loads(path.read_text())
            data["episode"]["seed"] += 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                store.read_episode(job.episode_id)

    def test_interrupted_write_is_not_committed(self) -> None:
        plan = self.plan()
        job = next(plan.jobs())
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(plan.manifest)
            path = store._summary_path(job.episode_id)
            path.parent.mkdir(parents=True)
            path.with_suffix(".steps.jsonl").write_text("{}\n")
            self.assertFalse(
                store.is_complete(job.episode_id, expected_job=job.as_dict())
            )
            store.write_episode(self.record(job), job=job.as_dict())
            self.assertTrue(
                store.is_complete(job.episode_id, expected_job=job.as_dict())
            )

    def test_manifest_prevents_configuration_and_build_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(self.plan(seed=1).manifest)
            store.initialize(self.plan(seed=1).manifest)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                store.initialize(self.plan(seed=2).manifest)
            manifest = self.plan(seed=1).manifest
            manifest["runtime"] = {"source_sha256": "changed"}
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                store.initialize(manifest)

    def test_mock_resume_does_not_reexecute_completed_episodes(self) -> None:
        plan = self.plan(episodes_per_matchup=3)
        with tempfile.TemporaryDirectory() as directory:

            def execute(job, output):
                DatasetStore(output).write_episode(self.record(job), job=job.as_dict())
                return "SUCCESS"

            with patch(
                "game_arena.openspiel.experiments._execute_mock_job",
                side_effect=execute,
            ) as mock:
                first = execute_mock_plan(plan, directory)
                second = execute_mock_plan(plan, directory)
            self.assertEqual(first["completed"], 3)
            self.assertEqual(second["resumed"], 3)
            self.assertEqual(mock.call_count, 3)

    def test_worker_failure_leaves_committed_work_resumable(self) -> None:
        plan = self.plan(episodes_per_matchup=3)
        calls = 0
        with tempfile.TemporaryDirectory() as directory:

            def execute(job, output):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected worker interruption")
                DatasetStore(output).write_episode(self.record(job), job=job.as_dict())
                return "SUCCESS"

            with patch(
                "game_arena.openspiel.experiments._execute_mock_job",
                side_effect=execute,
            ):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    execute_mock_plan(plan, directory)
                result = execute_mock_plan(plan, directory)
            self.assertEqual(result["resumed"], 1)
            self.assertEqual(result["completed"], 2)

    def test_mock_execution_failure_is_recorded_and_retried(self) -> None:
        plan = self.plan()
        job = next(plan.jobs())
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "game_arena.openspiel.harness.harness",
                side_effect=RuntimeError("injected engine failure"),
            ):
                failed = execute_mock_plan(plan, directory)
            self.assertEqual(failed["failed"], 1)
            evidence = DatasetStore(directory).read_episode(job.episode_id)
            self.assertEqual(evidence["status"], "FAILED")
            self.assertEqual(evidence["returns"], [])
            self.assertIn("injected engine failure", evidence["error"])
            retried = execute_mock_plan(plan, directory)
            self.assertEqual(retried["completed"], 1)
            self.assertEqual(retried["failed"], 0)
            self.assertTrue(
                DatasetStore(directory).read_episode(job.episode_id)["terminal"]
            )

    def test_real_mock_build_generates_multiplayer_and_solo_records(self) -> None:
        from game_arena.openspiel.replay import replay_episode

        plan = ExperimentPlan(
            PlanConfig(
                games=(
                    "tic_tac_toe",
                    "kuhn_poker",
                    "matrix_rps",
                    "matching_pennies_3p",
                    "catch",
                ),
                max_steps=100,
                seed=8,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            counts = execute_mock_plan(plan, directory)
            self.assertEqual(counts["completed"], 5)
            self.assertEqual(counts["failed"], 0)
            self.assertEqual(counts["truncated"], 0)
            for job in plan.jobs():
                record = DatasetStore(directory).read_episode(job.episode_id)
                self.assertEqual(len(record["returns"]), len(job.models))
                self.assertTrue(record["terminal"])
                self.assertEqual(record["seed"], job.seed)
                replayed = replay_episode(record)
                self.assertEqual(replayed["returns"], record["returns"])

    def test_normalized_export_has_stable_join_keys(self) -> None:
        job = next(self.plan().jobs())
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as output,
        ):
            store = DatasetStore(directory)
            store.initialize(self.plan().manifest)
            store.write_episode(self.record(job), job=job.as_dict())
            counts = export_dataset(directory, output, compressed=False)
            self.assertEqual(counts, {"episodes": 1, "steps": 1, "decisions": 1})
            step = json.loads((Path(output) / "steps.jsonl").read_text())
            decision = json.loads((Path(output) / "decisions.jsonl").read_text())
            self.assertEqual(step["episode_id"], decision["episode_id"])
            self.assertEqual(step["index"], decision["step_index"])
            self.assertNotIn("decisions", step)

    def test_export_cannot_replace_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(directory)
            store.initialize(self.plan().manifest)
            before = store.manifest()
            with self.assertRaisesRegex(ValueError, "destination must differ"):
                export_dataset(directory, directory)
            self.assertEqual(store.manifest(), before)

    def test_resource_tamper_after_plan_is_rejected_despite_cached_inspection(
        self,
    ) -> None:
        import pyspiel

        with tempfile.TemporaryDirectory() as directory:
            resource = Path(directory) / "fixture.nfg"
            original = pyspiel.game_to_nfg_string(pyspiel.load_game("matrix_rps"))
            resource.write_text(original)
            plan = ExperimentPlan(PlanConfig(games=(f"nfg_game(filename={resource})",)))
            # The modified definition is still a valid game. Inspection has
            # already cached the original file's metadata in this process.
            resource.write_text(original + "\n")
            output = Path(directory) / "dataset"
            with patch("game_arena.openspiel.harness.harness") as engine:
                with self.assertRaisesRegex(ValueError, "resources differ"):
                    execute_mock_plan(plan, output)
            engine.assert_not_called()
            self.assertEqual(list(DatasetStore(output).iter_episode_ids()), [])
            resource.write_text(original)
            self.assertEqual(execute_mock_plan(plan, output)["completed"], 1)

    def test_dry_run_does_not_execute(self) -> None:
        with (
            redirect_stdout(io.StringIO()) as stdout,
            patch("game_arena.openspiel.cli.execute_mock_plan") as execute,
        ):
            code = main(["plan", "--games", "tic_tac_toe", "--episodes", "20"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["execution"], "dry_run")
        execute.assert_not_called()

    def test_artifact_paths_reject_traversal(self) -> None:
        with self.assertRaises(ValueError):
            DatasetStore("/tmp").read_episode("../secret")


if __name__ == "__main__":
    unittest.main()
