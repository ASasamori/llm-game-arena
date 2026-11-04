"""Verify an episode from recorded transitions without querying any model."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import pyspiel

from game_arena.openspiel import catalog, parsers
from game_arena.openspiel.harness import _distribution, _initial_state, _legacy_fields
from game_arena.openspiel.provenance import derive_seed, runtime_manifest
from game_arena.openspiel.views import build_prompt, state_hash


class ReplayError(ValueError):
    """The trace is incompatible, incomplete, or diverges from OpenSpiel."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _same_values(expected: Any, actual: Any) -> bool:
    return len(expected) == len(actual) and all(
        math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        for a, b in zip(expected, actual, strict=True)
    )


def replay_episode(record: dict[str, Any]) -> dict[str, Any]:
    """Recreate and check every transition, player view, prompt and utility.

    Live model generation is deliberately absent. The same recorded decisions
    reproduce the game even when a provider cannot reproduce its own outputs.
    """
    manifest = record.get("manifest", {})
    runtime = runtime_manifest()
    _require(
        manifest.get("schema_version") == runtime["schema_version"],
        "Unsupported trace schema",
    )
    _require(
        manifest.get("open_spiel_version") == runtime["open_spiel_version"],
        "Replay requires the recorded OpenSpiel version",
    )
    _require(
        manifest.get("prompt_version") == runtime["prompt_version"],
        "Unsupported prompt version",
    )
    _require(
        manifest.get("parser_version") == runtime["parser_version"],
        "Unsupported parser version",
    )
    spec = record.get("game_spec") or manifest.get("game")
    _require(isinstance(spec, str), "Trace is missing its canonical game specification")
    _require(spec == manifest.get("game"), "Episode game configuration mismatch")
    _require(record.get("seed") == manifest.get("seed"), "Episode seed mismatch")
    _require(
        record.get("episode_id") == manifest.get("episode_id"),
        "Episode identity mismatch",
    )
    _require(
        record.get("arena_type") == manifest.get("run_type"),
        "Episode run type mismatch",
    )
    agents = manifest["player_agents"]
    _require(
        record.get("player_agents") == agents, "Episode player assignments mismatch"
    )
    _require(
        record.get("agent")
        == [model["identity"] for model in manifest["model_configs"]],
        "Episode model identities mismatch",
    )
    _require(
        catalog.game_source_files(spec) == manifest.get("game_sources", []),
        "Referenced game resources have changed; restore the recorded files",
    )
    game = catalog.load_game(spec, seed=manifest["seed"])
    state = _initial_state(game, manifest.get("mean_field_population", 0))
    _require(
        state_hash(state) == manifest.get("initial_state_hash"),
        "Initial state hash mismatch",
    )
    player_count = game.num_players()
    _require(
        record.get("game") == game.get_type().short_name, "Episode game name mismatch"
    )
    _require(len(agents) == player_count, "Episode player count mismatch")
    chance_rng = random.Random(derive_seed(manifest["seed"], "chance"))
    player_rngs = [
        random.Random(derive_seed(manifest["seed"], "player", p))
        for p in range(player_count)
    ]
    fallback_rngs = [
        random.Random(derive_seed(manifest["seed"], "fallback", p))
        for p in range(player_count)
    ]
    scripted = deque(manifest.get("input_actions") or [])
    own_history: list[list[dict[str, Any]]] = [[] for _ in range(player_count)]
    events = record.get("steps")
    if not isinstance(events, list):
        raise ReplayError("Trace is missing transitions")
    for index, event in enumerate(events):
        prefix = f"Transition {index}"
        _require(not state.is_terminal(), f"{prefix} occurs after terminal state")
        _require(event.get("index") == index, f"{prefix} index mismatch")
        _require(
            event.get("state_hash_before") == state_hash(state),
            f"{prefix} input hash mismatch",
        )
        players, actions = event["players"], event["actions"]
        kind = event["kind"]
        if kind == "chance":
            _require(state.is_chance_node(), f"{prefix} is not a chance node")
            _require(
                players == [int(pyspiel.PlayerId.CHANCE)] and len(actions) == 1,
                f"{prefix} invalid chance action shape",
            )
            outcomes = list(state.chance_outcomes())
            _require(
                actions[0] in {a for a, probability in outcomes if probability > 0},
                f"{prefix} illegal chance action",
            )
            _require(
                [list(pair) for pair in event["chance_outcomes"]]
                == [list(pair) for pair in outcomes],
                f"{prefix} chance distribution mismatch",
            )
            _require(
                actions[0]
                == chance_rng.choices(
                    [a for a, _ in outcomes], weights=[p for _, p in outcomes], k=1
                )[0],
                f"{prefix} seeded chance mismatch",
            )
            _require(
                event["action_strings"]
                == [state.action_to_string(players[0], actions[0])],
                f"{prefix} chance label mismatch",
            )
            _require(
                event["decisions"] == [],
                f"{prefix} chance cannot contain player decisions",
            )
            state.apply_action(actions[0])
        elif kind == "mean_field":
            _require(state.is_mean_field_node(), f"{prefix} is not a mean-field node")
            _require(
                players == [] and actions == [],
                f"{prefix} invalid mean-field action shape",
            )
            _require(
                event["distribution_support"] == list(state.distribution_support()),
                f"{prefix} population support mismatch",
            )
            probabilities = event["distribution"]
            try:
                _, configured = _distribution(
                    state, manifest["mean_field_distribution"]
                )
            except (ValueError, TypeError) as exc:
                raise ReplayError(
                    f"{prefix} invalid configured mean-field policy"
                ) from exc
            _require(
                _same_values(probabilities, configured),
                f"{prefix} configured population policy mismatch",
            )
            _require(
                event["action_strings"] == [] and event["decisions"] == [],
                f"{prefix} invalid mean-field decision data",
            )
            _require(
                len(probabilities) == len(event["distribution_support"]),
                f"{prefix} distribution shape mismatch",
            )
            _require(
                all(math.isfinite(p) and p >= 0 for p in probabilities),
                f"{prefix} invalid population probabilities",
            )
            _require(
                not probabilities
                or math.isclose(sum(probabilities), 1.0, abs_tol=1e-8),
                f"{prefix} population mass mismatch",
            )
            state.update_distribution(probabilities)
        else:
            simultaneous = state.is_simultaneous_node()
            _require(
                kind == ("simultaneous" if simultaneous else "decision"),
                f"{prefix} node kind mismatch",
            )
            _require(
                not state.is_chance_node() and not state.is_mean_field_node(),
                f"{prefix} expected a player node",
            )
            expected_players = (
                list(range(player_count)) if simultaneous else [state.current_player()]
            )
            _require(
                players == expected_players and len(actions) == len(players),
                f"{prefix} player/action shape mismatch",
            )
            decisions = event["decisions"]
            _require(
                len(decisions) == len(players), f"{prefix} decision count mismatch"
            )
            for player, action, decision in zip(
                players, actions, decisions, strict=True
            ):
                legal = list(state.legal_actions(player))
                _require(
                    decision["agent"] == agents[player],
                    f"{prefix} assigned agent mismatch",
                )
                _require(
                    decision["player"] == player and decision["action"] == action,
                    f"{prefix} decision/action mismatch",
                )
                _require(
                    decision["legal_actions"] == legal,
                    f"{prefix} legal actions mismatch",
                )
                if not legal and simultaneous:
                    _require(
                        action == pyspiel.INVALID_ACTION,
                        f"{prefix} inactive player action mismatch",
                    )
                    continue
                _require(
                    type(action) is int and action in legal,
                    f"{prefix} illegal player action",
                )
                prompt, view_type, observation, labels = build_prompt(
                    state, player, legal, own_history[player], manifest.get("rules", "")
                )
                _require(
                    decision["action_string"] == labels[action],
                    f"{prefix} action label mismatch",
                )
                _require(
                    decision["legal_action_strings"] == list(labels.values()),
                    f"{prefix} legal action labels mismatch",
                )
                if decision.get("fallback_reason") is not None:
                    _require(
                        manifest["failure_policy"] == "random"
                        and action == fallback_rngs[player].choice(legal),
                        f"{prefix} fallback policy mismatch",
                    )
                elif manifest["run_type"] == "RAND" and player != 0:
                    _require(
                        action == player_rngs[player].choice(legal),
                        f"{prefix} random baseline mismatch",
                    )
                elif manifest["run_type"] == "INPUT" and player != 0:
                    _require(
                        bool(scripted) and action == scripted.popleft(),
                        f"{prefix} input script mismatch",
                    )
                _require(
                    decision["observation_type"] == view_type
                    and decision["observation"] == observation,
                    f"{prefix} player observation mismatch",
                )
                if decision.get("prompt") is not None:
                    _require(
                        decision["prompt"] == prompt, f"{prefix} player prompt mismatch"
                    )
                    if decision.get("fallback_reason") is None:
                        try:
                            selected = parsers.parse_legal_action(
                                decision["response"],
                                labels,
                                game=game.get_type().short_name,
                                player=player,
                            )
                        except (ValueError, TypeError) as exc:
                            raise ReplayError(
                                f"{prefix} response cannot be parsed"
                            ) from exc
                        _require(
                            selected.action == action,
                            f"{prefix} response/action mismatch",
                        )
                        _require(
                            selected.summary == decision["summary"],
                            f"{prefix} response summary mismatch",
                        )
            _require(
                event["action_strings"]
                == [decision["action_string"] for decision in decisions],
                f"{prefix} joint action labels mismatch",
            )
            if simultaneous:
                state.apply_actions(actions)
            else:
                state.apply_action(actions[0])
            for decision in decisions:
                own_history[decision["player"]].append(
                    {"action": decision["action"], "label": decision["action_string"]}
                )
        _require(
            event["state_hash"] == state_hash(state), f"{prefix} output hash mismatch"
        )
        _require(
            _same_values(event["rewards"], state.rewards()), f"{prefix} reward mismatch"
        )
        _require(
            _same_values(event["returns"], state.returns()),
            f"{prefix} utility mismatch",
        )
        _require(
            event["terminal"] == state.is_terminal(), f"{prefix} terminal flag mismatch"
        )
    _require(len(events) == record["num_steps"], "Episode transition count mismatch")
    _require(
        record["terminal"] == state.is_terminal(), "Episode terminal flag mismatch"
    )
    _require(
        _same_values(record["returns"], state.returns()), "Episode utility mismatch"
    )
    _require(
        record["final_state_hash"] == state_hash(state), "Final state hash mismatch"
    )
    _require(
        record["status"] == ("SUCCESS" if state.is_terminal() else "TRUNCATED"),
        "Episode status mismatch",
    )
    _require(
        state.is_terminal() or len(events) == manifest["max_steps"],
        "Truncated trace ends before its configured limit",
    )
    competitive = game.get_type().utility in {
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.Utility.CONSTANT_SUM,
    }
    values = list(state.returns())
    winners = (
        [p for p, value in enumerate(values) if value == max(values)]
        if state.is_terminal() and competitive
        else []
    )
    winner = (
        agents[winners[0]]
        if len(winners) == 1 and player_count > 1
        else "UNDETERMINED/TIE"
    )
    _require(
        record["winner_players"] == winners and record["winner"] == winner,
        "Episode winner mismatch",
    )
    fallback_count = sum(
        decision.get("fallback_reason") is not None
        for event in events
        for decision in event["decisions"]
    )
    _require(
        record["fallback_count"] == fallback_count
        and record["degraded"] == (fallback_count > 0),
        "Episode fallback accounting mismatch",
    )
    _require(
        record["decision_count"] == sum(len(event["decisions"]) for event in events),
        "Episode decision count mismatch",
    )
    actions, summaries = _legacy_fields(events)
    _require(
        record["action"] == actions and record["summary"] == summaries,
        "Episode action/summary projection mismatch",
    )
    return {
        "episode_id": record["episode_id"],
        "num_steps": len(events),
        "returns": list(state.returns()),
        "terminal": state.is_terminal(),
        "state_hash": state_hash(state),
        "source_matches": manifest.get("source_sha256") == runtime["source_sha256"],
        "verified": True,
    }
