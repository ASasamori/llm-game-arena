"""Lazy, reproducible experiment scheduling with bounded mock workers.

The offline executor deliberately constructs only local mock models. Real model
execution remains behind the existing queue/runner entry point.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import multiprocessing
from pathlib import Path
from typing import Any, Iterator
import uuid

from game_arena.openspiel.datasets import DatasetStore, fingerprint

PLAN_SCHEMA_VERSION = 1


def derive_seed(seed: int, *identity: object) -> int:
    """Independent streams stay stable across process counts and retry order."""
    digest = fingerprint([seed, *identity])
    return int(digest[:16], 16) & ((1 << 63) - 1)


@dataclass(frozen=True)
class PlanConfig:
    games: tuple[str, ...]
    episodes_per_matchup: int = 1
    model_names: tuple[str, ...] = ("mock",)
    seed: int = 0
    max_steps: int = 1000
    rotate_seats: bool = True
    mean_field_distribution: str | dict[str, float] | None = None
    mean_field_population: int = 0

    def __post_init__(self) -> None:
        if not self.games or not all(
            isinstance(game, str) and game.strip() for game in self.games
        ):
            raise ValueError("games must contain nonempty OpenSpiel specifications")
        if len(set(self.games)) != len(self.games):
            raise ValueError("Duplicate game specifications would duplicate episodes")
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("model_names must contain unique labels")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.model_names
        ):
            raise ValueError("Model labels must be nonempty strings")
        if type(self.episodes_per_matchup) is not int or self.episodes_per_matchup < 1:
            raise ValueError("episodes_per_matchup must be a positive integer")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            type(self.mean_field_population) is not int
            or self.mean_field_population < 0
        ):
            raise ValueError("mean_field_population must be a nonnegative integer")
        if (
            isinstance(self.mean_field_distribution, str)
            and self.mean_field_distribution != "uniform"
        ):
            raise ValueError("The named mean field distribution must be 'uniform'")


@dataclass(frozen=True)
class EpisodeJob:
    experiment_id: str
    ordinal: int
    episode_id: str
    game: str
    models: tuple[str, ...]
    matchup: tuple[str, ...]
    episode_index: int
    seed: int
    max_steps: int
    mean_field_distribution: str | dict[str, float] | None = None
    mean_field_population: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentPlan:
    """Inspect each game once, then generate arbitrarily many jobs lazily."""

    def __init__(self, config: PlanConfig):
        from game_arena.openspiel.catalog import inspect_game, resolve_game_spec
        from game_arena.openspiel.provenance import runtime_manifest

        self.config = config
        self.specs = tuple(resolve_game_spec(game) for game in config.games)
        self.games: tuple[dict[str, Any], ...] = tuple(
            inspect_game(game) for game in self.specs
        )
        if len(
            {
                game.get("canonical_game", spec)
                for spec, game in zip(self.specs, self.games, strict=True)
            }
        ) != len(self.games):
            raise ValueError("Game specifications resolve to duplicate configurations")
        for game in self.games:
            if not game.get("supported", True):
                raise ValueError(
                    "; ".join(game.get("unsupported_reasons", ["Unsupported game"]))
                )
            players = int(game["num_players"])
            if players < 1:
                raise ValueError("Game must have at least one player")
            if len(config.model_names) != 1 and len(config.model_names) < players:
                raise ValueError(
                    f"{game.get('short_name')} needs {players} model labels or a single selfplay label"
                )
            dynamics = str(game.get("dynamics", "")).lower()
            if "mean" in dynamics and config.mean_field_distribution is None:
                raise ValueError(
                    "Mean field games require an explicit mean_field_distribution"
                )
        self.manifest = {
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "config": asdict(config),
            "games": self.games,
            "runtime": runtime_manifest(),
        }
        self.experiment_id = fingerprint(self.manifest)
        self.manifest["experiment_id"] = self.experiment_id

    @property
    def episode_count(self) -> int:
        from math import comb

        return sum(
            (
                1
                if len(self.config.model_names) == 1
                else comb(len(self.config.model_names), int(game["num_players"]))
            )
            * self.config.episodes_per_matchup
            for game in self.games
        )

    def jobs(
        self, *, shard_index: int = 0, shard_count: int = 1
    ) -> Iterator[EpisodeJob]:
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        ordinal = 0
        for spec, game in zip(self.specs, self.games, strict=True):
            players = int(game["num_players"])
            matchups = (
                ((self.config.model_names[0],) * players,)
                if len(self.config.model_names) == 1
                else combinations(self.config.model_names, players)
            )
            for matchup in matchups:
                # Jump directly to this shard's indexes, even for enormous plans.
                first_index = (shard_index - ordinal) % shard_count
                for episode_index in range(
                    first_index, self.config.episodes_per_matchup, shard_count
                ):
                    seat_offset = (
                        episode_index % players if self.config.rotate_seats else 0
                    )
                    seats = matchup[seat_offset:] + matchup[:seat_offset]
                    identity = [self.experiment_id, spec, matchup, episode_index]
                    episode_id = str(
                        uuid.UUID(
                            hex=hashlib.sha256(
                                fingerprint(identity).encode()
                            ).hexdigest()[:32]
                        )
                    )
                    yield EpisodeJob(
                        experiment_id=self.experiment_id,
                        ordinal=ordinal + episode_index,
                        episode_id=episode_id,
                        game=spec,
                        models=tuple(seats),
                        matchup=tuple(matchup),
                        episode_index=episode_index,
                        seed=derive_seed(
                            self.config.seed, spec, matchup, episode_index
                        ),
                        max_steps=self.config.max_steps,
                        mean_field_distribution=self.config.mean_field_distribution,
                        mean_field_population=self.config.mean_field_population,
                    )
                ordinal += self.config.episodes_per_matchup


def _execute_mock_job(job: EpisodeJob, directory: str) -> str:
    from datetime import datetime, timezone

    from game_arena.openspiel import catalog
    from game_arena.openspiel.harness import harness
    from game_arena.openspiel.model_generation import MockModel
    from game_arena.openspiel.provenance import runtime_manifest
    from game_arena.openspiel.utils import HarnessOptions

    store = DatasetStore(directory)
    planned_manifest = store.manifest()
    expected_runtime = planned_manifest["runtime"]
    if expected_runtime != runtime_manifest():
        raise ValueError("Worker build differs from the experiment manifest")
    matching = [
        info
        for spec, info in zip(
            planned_manifest["config"]["games"], planned_manifest["games"], strict=True
        )
        if catalog.resolve_game_spec(spec) == job.game
    ]
    if len(matching) != 1:
        raise ValueError("Worker game does not match the experiment manifest")
    expected_sources = matching[0].get("source_files", [])
    # Never use cached inspection here: definitions can change after planning.
    if catalog.game_source_files(job.game) != expected_sources:
        raise ValueError(
            "Referenced game resources differ from the experiment manifest"
        )
    options = HarnessOptions(
        type="PVP",
        models=[MockModel(name=name) for name in job.models],
        num_steps=job.max_steps,
        input=None,
        game=job.game,
        seed=job.seed,
        episode_id=job.episode_id,
        metadata={
            "experiment_id": job.experiment_id,
            "ordinal": job.ordinal,
            "episode_index": job.episode_index,
            "matchup": list(job.matchup),
        },
        mean_field_distribution=job.mean_field_distribution,
        mean_field_population=job.mean_field_population,
    )
    started = datetime.now(timezone.utc)
    result: dict[str, Any]
    try:
        result = dict(harness(options))
    except Exception as error:
        # Keep execution failures as evidence, without inventing an outcome.
        # Resume retries FAILED records; a process crash publishes no summary.
        result = {
            "episode_id": job.episode_id,
            "game": job.game,
            "env": "OPENSPIEL",
            "arena_type": "PVP",
            "status": "FAILED",
            "error": str(error),
            "failure_type": type(error).__name__,
            "seed": job.seed,
            "returns": [],
            "terminal": False,
            "num_steps": 0,
            "steps": [],
            "action": [],
            "summary": [],
            "player_agents": list(job.models),
            "start_time": started,
            "end_time": datetime.now(timezone.utc),
            "manifest": {
                **expected_runtime,
                "game": job.game,
                "seed": job.seed,
                "game_sources": expected_sources,
            },
        }
    actual_manifest = result.get("manifest", {})
    if any(
        actual_manifest.get(key) != value for key, value in expected_runtime.items()
    ):
        raise ValueError("Episode build differs from the experiment manifest")
    if actual_manifest.get("game_sources", []) != expected_sources:
        raise ValueError("Episode game resources differ from the experiment manifest")
    store.write_episode(result, job=job.as_dict())
    return str(result["status"])


def execute_mock_plan(
    plan: ExperimentPlan,
    directory: str | Path,
    *,
    workers: int = 1,
    shard_index: int = 0,
    shard_count: int = 1,
    max_pending: int | None = None,
) -> dict[str, int]:
    """Execute only mock policies, resuming checksum-validated episode artifacts.

    Exceptions stop the executor and leave completed episodes available to resume.
    At most max_pending futures and one episode per worker are retained in RAM.
    Shard parameters intentionally do not alter the manifest or episode identity.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    pending_limit = workers * 2 if max_pending is None else max_pending
    if pending_limit < workers:
        raise ValueError("max_pending must be at least workers")
    # Validate sharding before creating any artifact.
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    store = DatasetStore(directory)
    store.initialize(plan.manifest)
    counts = {"completed": 0, "resumed": 0, "truncated": 0, "failed": 0}

    def pending_jobs() -> Iterator[EpisodeJob]:
        for job in plan.jobs(shard_index=shard_index, shard_count=shard_count):
            if store.is_complete(job.episode_id, expected_job=job.as_dict()):
                counts["resumed"] += 1
            else:
                yield job

    def record_status(status: str) -> None:
        counts["completed"] += 1
        if status == "TRUNCATED":
            counts["truncated"] += 1
        elif status != "SUCCESS":
            counts["failed"] += 1

    if workers == 1:
        for job in pending_jobs():
            record_status(_execute_mock_job(job, str(directory)))
        return counts

    jobs = iter(pending_jobs())
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending: set[Future[str]] = set()
        exhausted = False
        while pending or not exhausted:
            while len(pending) < pending_limit and not exhausted:
                try:
                    job = next(jobs)
                except StopIteration:
                    exhausted = True
                else:
                    pending.add(executor.submit(_execute_mock_job, job, str(directory)))
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record_status(future.result())
    return counts
