"""Generic, reproducible arena episode execution.

OpenSpiel and standalone native environments own game rules. This module owns player isolation, independent RNG
streams, legal-action selection and an auditable transition log. No provider is
constructed here; callers inject stateless model implementations.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
import math
import os
import random
import time
from typing import Any
import uuid

import pyspiel

from game_arena.openspiel import catalog, parsers, utils
from game_arena.openspiel.provenance import derive_seed, runtime_manifest
from game_arena.openspiel.views import build_prompt, evaluator_state, state_hash
from orchestrator.failures import FailureCategory, classify_exception
from orchestrator.observability import increment_counter

LOGGER = logging.getLogger(__name__)
SUPPORTED_TYPES = ["INPUT", "RAND", "PVP"]
ENV = "OPENSPIEL"


def _initial_state(game: Any, population: int = 0) -> Any:
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
        states = game.new_initial_states()
        if not 0 <= population < len(states):
            raise ValueError(f"mean_field_population must be in [0, {len(states)})")
        return states[population]
    if population != 0:
        raise ValueError("mean_field_population applies only to mean-field games")
    return game.new_initial_state()


def _distribution(
    state: Any, config: str | dict[str, float]
) -> tuple[list[str], list[float]]:
    support = list(state.distribution_support())
    if config == "uniform":
        values = [1.0 / len(support)] * len(support) if support else []
    elif isinstance(config, dict):
        missing = [label for label in support if label not in config]
        if missing:
            raise ValueError(
                f"Mean-field distribution is missing support entries: {missing[:3]}"
            )
        values = [float(config[label]) for label in support]
    else:
        raise ValueError(
            "mean_field_distribution must be 'uniform' or a support-to-probability mapping"
        )
    if any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("Mean-field probabilities must be finite and non-negative")
    if values and not math.isclose(sum(values), 1.0, abs_tol=1e-8):
        raise ValueError("Mean-field probabilities must sum to one")
    return support, values


def _seed(options: utils.HarnessOptions) -> int:
    value = (
        options["seed"]
        if "seed" in options
        else int(os.environ.get("HARNESS_RANDOM_SEED", "0"))
    )
    if type(value) is not int or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _legacy_fields(steps: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    actions = [
        " | ".join(event["action_strings"])
        if event["kind"] != "mean_field"
        else "<mean_field_update>"
        for event in steps
    ]
    summaries = [
        " | ".join(d["summary"] for d in event["decisions"])
        or (
            "Chance outcome"
            if event["kind"] == "chance"
            else "Exogenous mean-field distribution update"
        )
        for event in steps
    ]
    return actions, summaries


def harness(options: utils.HarnessOptions) -> utils.EpisodeResult:
    """Execute one bounded episode; all random choices belong to this episode.

    PVP supplies one model per seat (including one-player environments). RAND
    and INPUT supply one model in seat zero, with all other seats using the
    baseline. A step means a state transition, including chance and population
    updates; a simultaneous transition may contain several player decisions.
    """
    mode = options["type"]
    if mode not in SUPPORTED_TYPES:
        raise ValueError(f"Type: {mode} is not supported")
    models = options["models"]
    if not models:
        raise ValueError("At least one model is required")
    requested_steps = options["num_steps"]
    if requested_steps is not None and (
        type(requested_steps) is not int or requested_steps <= 0
    ):
        raise ValueError("num_steps must be a positive integer or None")
    failure_policy = options.get("failure_policy", "random")
    if failure_policy not in {"random", "raise"}:
        raise ValueError("failure_policy must be random or raise")
    seed = _seed(options)
    spec = catalog.resolve_game_spec(options["game"])
    # A malformed native game constructor can abort its interpreter. Admission
    # happens in catalog's disposable subprocess, cached per game configuration.
    info = catalog.inspect_game(spec)
    if not info["supported"]:
        raise catalog.UnsupportedGameError("; ".join(info["unsupported_reasons"]))
    game = catalog.load_game(spec, seed=seed)
    backend = getattr(game, "environment_backend", "openspiel")
    effective_rules = options.get("rules", "") or getattr(game, "rules", "")
    player_count = game.num_players()
    if mode == "PVP" and len(models) != player_count:
        raise ValueError(
            f"PVP requires one model per player ({player_count}), got {len(models)}"
        )
    if mode in {"RAND", "INPUT"} and len(models) != 1:
        raise ValueError(f"{mode} requires exactly one model")
    if mode == "INPUT" and (not options["input"] or player_count < 2):
        raise ValueError(
            "INPUT requires an opponent and a non-empty input action sequence"
        )
    # Never consume the caller's script. Retries must start with the same input.
    scripted_actions = deque(options["input"] or ())
    population = options.get("mean_field_population", 0)
    if type(population) is not int or population < 0:
        raise ValueError("mean_field_population must be a non-negative integer")
    mean_field = options.get("mean_field_distribution")
    if (
        game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD
        and mean_field is None
    ):
        raise ValueError(
            "Mean-field games require an explicit mean_field_distribution; 'uniform' is an exogenous baseline, not an equilibrium"
        )
    state = _initial_state(game, population)
    # OpenSpiel max_game_length often excludes chance/mean-field transitions.
    max_steps = requested_steps or max(1000, int(game.max_history_length()) + 1)
    started_at = datetime.now(timezone.utc)
    episode_id = options.get("episode_id") or str(uuid.uuid4())
    player_agents = (
        [str(model) for model in models]
        if mode == "PVP"
        else [str(models[0])] + [mode] * (player_count - 1)
    )
    rngs = [
        random.Random(derive_seed(seed, "player", player))
        for player in range(player_count)
    ]
    chance_rng = random.Random(derive_seed(seed, "chance"))
    fallback_rngs = [
        random.Random(derive_seed(seed, "fallback", player))
        for player in range(player_count)
    ]
    own_history: list[list[dict[str, Any]]] = [[] for _ in range(player_count)]
    steps: list[dict[str, Any]] = []
    fallback_count = 0
    manifest = {
        **runtime_manifest(),
        "episode_id": episode_id,
        "game": catalog.canonical_game_spec(spec, seed=seed),
        "game_sources": catalog.game_source_files(spec),
        "run_type": mode,
        "seed": seed,
        "player_agents": player_agents,
        "model_configs": [
            {
                "identity": str(model),
                "mock": getattr(
                    model, "_mock_mode", type(model).__name__ == "MockModel"
                ),
                "model_name": getattr(model, "_model_name", None),
            }
            for model in models
        ],
        "max_steps": max_steps,
        "failure_policy": failure_policy,
        "rules": effective_rules,
        "environment_backend": backend,
        "input_actions": list(scripted_actions) if mode == "INPUT" else None,
        "mean_field_distribution": mean_field,
        "mean_field_population": population,
        "metadata": options.get("metadata", {}),
        "initial_state_hash": state_hash(state),
    }
    increment_counter("harness.episode.started", tags={"run_type": mode})

    def choose(player: int) -> dict[str, Any]:
        nonlocal fallback_count
        legal_actions = list(state.legal_actions(player))
        if not legal_actions:
            # OpenSpiel's simultaneous API uses INVALID_ACTION for inactive seats.
            return {
                "player": player,
                "agent": player_agents[player],
                "action": pyspiel.INVALID_ACTION,
                "action_string": "<inactive>",
                "legal_actions": [],
                "summary": "Inactive player",
                "fallback_reason": None,
            }
        prompt, view_type, observation, legal = build_prompt(
            state, player, legal_actions, own_history[player], effective_rules
        )
        record: dict[str, Any] = {
            "player": player,
            "agent": player_agents[player],
            "observation_type": view_type,
            "observation": observation,
            "legal_actions": legal_actions,
            "legal_action_strings": [legal[a] for a in legal_actions],
            "prompt": None,
            "response": None,
            "usage": {},
            "response_metadata": {},
            "fallback_reason": None,
        }
        if mode != "PVP" and player != 0:
            if mode == "RAND":
                action = rngs[player].choice(legal_actions)
                summary = "Random action"
            else:
                if not scripted_actions:
                    raise ValueError(
                        "INPUT action sequence exhausted before episode completion"
                    )
                action = scripted_actions.popleft()
                if type(action) is not int or action not in legal:
                    raise ValueError(
                        f"The input action {action} is not in legal actions"
                    )
                summary = "Input action"
        else:
            model = models[player] if mode == "PVP" else models[0]
            record["prompt"] = prompt
            call_started = time.monotonic()
            stage = "model_call"
            try:
                response = model.generate_with_text_input(prompt)
                record["response"] = response.main_response
                record["usage"] = getattr(response, "usage", {})
                record["response_metadata"] = getattr(response, "metadata", {})
                stage = "parse"
                if not response.main_response or len(response.main_response) > 8000:
                    raise ValueError("Model response must contain 1 to 8000 characters")
                parsed = parsers.parse_legal_action(
                    response.main_response,
                    legal,
                    game=game.get_type().short_name,
                    player=player,
                )
                action, summary = parsed.action, parsed.summary
            except Exception as exc:
                if failure_policy == "raise":
                    raise
                fallback_count += 1
                category = (
                    FailureCategory.MODEL_API
                    if stage == "model_call"
                    else FailureCategory.PARSER
                )
                failure = classify_exception(
                    exc,
                    stage=f"harness.{stage}",
                    default_category=category,
                    default_code=f"{stage}_failed",
                )
                record["fallback_reason"] = {
                    "category": category.value,
                    "code": failure.code,
                    "message": str(exc),
                }
                increment_counter("harness.fallback.total", tags={"reason": stage})
                increment_counter(
                    "harness.failure.total",
                    tags={"category": category.value, "code": failure.code},
                )
                action = fallback_rngs[player].choice(legal_actions)
                summary = f"{failure.compact()} Random legal action fallback applied."
                LOGGER.warning(
                    "legal_action_fallback", extra={"player": player, "stage": stage}
                )
            finally:
                record["latency_ms"] = (time.monotonic() - call_started) * 1000
        record.update(action=action, action_string=legal[action], summary=summary)
        return record

    for index in range(max_steps):
        if state.is_terminal():
            break
        before = state_hash(state)
        event: dict[str, Any] = {
            "index": index,
            "state_hash_before": before,
            "decisions": [],
        }
        if (snapshot := evaluator_state(state)) is not None:
            event["state_before"] = snapshot
        if state.is_chance_node():
            outcomes = list(state.chance_outcomes())
            if (
                not outcomes
                or any(not math.isfinite(p) or p < 0 for _, p in outcomes)
                or not math.isclose(sum(p for _, p in outcomes), 1.0, abs_tol=1e-7)
            ):
                raise ValueError("Game supplied an invalid chance distribution")
            action = chance_rng.choices(
                [a for a, _ in outcomes], weights=[p for _, p in outcomes], k=1
            )[0]
            player = int(pyspiel.PlayerId.CHANCE)
            event.update(
                kind="chance",
                players=[player],
                actions=[action],
                action_strings=[state.action_to_string(player, action)],
                chance_outcomes=outcomes,
            )
            state.apply_action(action)
        elif state.is_mean_field_node():
            if mean_field is None:
                raise ValueError("A mean-field distribution is required")
            support, probabilities = _distribution(state, mean_field)
            event.update(
                kind="mean_field",
                players=[],
                actions=[],
                action_strings=[],
                distribution_support=support,
                distribution=probabilities,
            )
            state.update_distribution(probabilities)
        elif state.is_simultaneous_node():
            decisions = [choose(player) for player in range(player_count)]
            event.update(
                kind="simultaneous",
                players=list(range(player_count)),
                actions=[d["action"] for d in decisions],
                action_strings=[d["action_string"] for d in decisions],
                decisions=decisions,
            )
            state.apply_actions(event["actions"])
        else:
            player = state.current_player()
            if not 0 <= player < player_count:
                raise ValueError(f"Unsupported player node {player}")
            decision = choose(player)
            if decision["action"] == pyspiel.INVALID_ACTION:
                raise ValueError("Non-terminal decision node has no legal actions")
            event.update(
                kind="decision",
                players=[player],
                actions=[decision["action"]],
                action_strings=[decision["action_string"]],
                decisions=[decision],
            )
            state.apply_action(decision["action"])
        for decision in event["decisions"]:
            own_history[decision["player"]].append(
                {"action": decision["action"], "label": decision["action_string"]}
            )
        event.update(
            rewards=list(state.rewards()),
            returns=list(state.returns()),
            state_hash=state_hash(state),
            terminal=state.is_terminal(),
        )
        if (snapshot := evaluator_state(state)) is not None:
            event["state_after"] = snapshot
        steps.append(event)

    terminal = state.is_terminal()
    returns = list(state.returns())
    competitive = game.get_type().utility in {
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.Utility.CONSTANT_SUM,
    }
    winners = (
        [p for p, value in enumerate(returns) if value > 0]
        if terminal and backend == "native"
        else [p for p, value in enumerate(returns) if value == max(returns)]
        if terminal and competitive
        else []
    )
    winner = (
        player_agents[winners[0]]
        if len(winners) == 1 and player_count > 1
        else "UNDETERMINED/TIE"
    )
    action_list, summary_list = _legacy_fields(steps)
    increment_counter("harness.episode.completed", tags={"run_type": mode})
    return utils.EpisodeResult(
        episode_id=episode_id,
        env="NATIVE" if backend == "native" else ENV,
        agent=[str(model) for model in models],
        arena_type=mode,
        game=game.get_type().short_name,
        game_spec=manifest["game"],
        seed=seed,
        status="SUCCESS" if terminal else "TRUNCATED",
        winner=winner,
        winner_players=winners,
        terminal=terminal,
        returns=returns,
        num_steps=len(steps),
        decision_count=sum(len(event["decisions"]) for event in steps),
        fallback_count=fallback_count,
        degraded=fallback_count > 0,
        error="No errors"
        if terminal
        else f"Reached configured num_steps={max_steps} before terminal state.",
        summary=summary_list,
        action=action_list,
        start_time=started_at,
        end_time=datetime.now(timezone.utc),
        player_agents=player_agents,
        steps=steps,
        manifest=manifest,
        final_state_hash=state_hash(state),
    )
