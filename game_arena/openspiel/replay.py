"""Verify an episode from recorded transitions without querying any model."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import pyspiel

from game_arena.openspiel import catalog, parsers
from game_arena.openspiel.decision_audit import describe_selection
from game_arena.openspiel.presentation import describe_state
from game_arena.openspiel.harness import (
    CONTROLLED_STOP_REASONS,
    _distribution,
    _initial_state,
    _legacy_fields,
    _public_game_progress,
    _quest_limit_reached,
    _validate_quest_limit,
)
from game_arena.openspiel.provenance import derive_seed, runtime_manifest
from game_arena.openspiel.views import (
    build_prompt,
    evaluator_state,
    public_state,
    state_hash,
)


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


def _verify_pending_transition(
    record: dict[str, Any],
    state: Any,
    manifest: dict[str, Any],
    own_history: list[list[dict[str, Any]]],
    player_rngs: list[random.Random],
    fallback_rngs: list[random.Random],
    scripted: deque,
) -> None:
    """Check unapplied request evidence without advancing the engine state.

    The remote exception, token usage and provider metadata are recorded claims.
    Replay can validate the request context and any completed local choices; it
    deliberately cannot certify that an external provider produced an error.
    """
    pending = record.get("pending_transition")
    features = manifest.get("recorded_features", [])
    if "pending_attempts_v1" in features:
        _require(isinstance(pending, dict), "Missing required pending transition")
        _require(
            all(
                key in record
                for key in (
                    "pending_decision_count",
                    "pending_fallback_count",
                    "failed_attempt_count",
                )
            ),
            "Missing required pending attempt counts",
        )
    if pending is None:
        _require(
            all(
                record.get(key, 0) == 0
                for key in (
                    "pending_decision_count",
                    "pending_fallback_count",
                    "failed_attempt_count",
                )
            ),
            "Episode claims pending attempts without pending evidence",
        )
        return
    _require(isinstance(pending, dict), "Invalid pending transition")
    _require(
        not state.is_terminal()
        and not state.is_chance_node()
        and not state.is_mean_field_node(),
        "Pending agent request is not at an acting-player node",
    )
    simultaneous = state.is_simultaneous_node()
    players = (
        list(range(state.get_game().num_players()))
        if simultaneous
        else [state.current_player()]
    )
    _require(
        pending.get("index") == len(record["steps"])
        and pending.get("applied") is False
        and pending.get("kind") == ("simultaneous" if simultaneous else "decision")
        and pending.get("players") == players,
        "Pending transition position or application flag mismatch",
    )
    _require(
        pending.get("state_hash_before") == state_hash(state),
        "Pending transition state hash mismatch",
    )
    for key, snapshot in (
        ("state_before", evaluator_state(state)),
        ("public_state_before", public_state(state)),
        ("review_state_before", describe_state(state)),
    ):
        if key in pending:
            _require(
                snapshot is not None and pending[key] == snapshot,
                "Pending transition board mismatch",
            )
        elif key == "review_state_before" and "openspiel_review_v1" in features:
            raise ReplayError("Pending transition is missing its required review state")
    decisions = pending.get("decisions")
    _require(
        isinstance(decisions, list) and len(decisions) < len(players),
        "Pending decision batch is not an incomplete actor prefix",
    )
    _require(
        pending.get("decision_count") == len(decisions),
        "Pending transition decision count mismatch",
    )
    failure = pending.get("failed_attempt")
    _require(isinstance(failure, dict), "Pending transition has no failed attempt")
    for offset, decision in enumerate([*decisions, failure]):
        prefix = f"Pending attempt {offset}"
        actor = players[offset]
        _require(isinstance(decision, dict), f"{prefix} is invalid")
        _require(
            decision.get("player") == actor
            and decision.get("agent") == manifest["player_agents"][actor],
            f"{prefix} assigned actor mismatch",
        )
        legal = list(state.legal_actions(actor))
        _require(
            decision.get("legal_actions") == legal, f"{prefix} legal actions mismatch"
        )
        failed = offset == len(decisions)
        if not legal and not failed:
            _require(
                simultaneous
                and decision.get("action") == pyspiel.INVALID_ACTION
                and decision.get("action_string") == "<inactive>",
                f"{prefix} inactive action mismatch",
            )
        else:
            detail = decision.get("failure", {}) if failed else {}
            stage = detail.get("stage") if isinstance(detail, dict) else None
            if not failed or stage not in {"legal_actions", "observation"}:
                prompt, view_type, observation, labels = build_prompt(
                    state, actor, legal, own_history[actor], manifest.get("rules", "")
                )
                expects_model = manifest["run_type"] == "PVP" or actor == 0
                _require(
                    decision.get("observation_type") == view_type
                    and decision.get("observation") == observation,
                    f"{prefix} observation mismatch",
                )
                _require(
                    decision.get("legal_action_strings") == list(labels.values()),
                    f"{prefix} legal labels mismatch",
                )
                _require(
                    decision.get("prompt") == (prompt if expects_model else None),
                    f"{prefix} prompt mismatch",
                )
                if failed:
                    if stage == "parse":
                        response = decision.get("response")
                        invalid = (
                            not isinstance(response, str)
                            or not response
                            or len(response) > 8000
                        )
                        if not invalid:
                            try:
                                parsers.parse_legal_action(
                                    response, labels, game=record["game"], player=actor
                                )
                            except (ValueError, TypeError):
                                invalid = True
                        _require(
                            invalid,
                            f"{prefix} claims a parse failure for a valid response",
                        )
                    elif (
                        stage == "policy"
                        and not expects_model
                        and manifest["run_type"] == "INPUT"
                    ):
                        if "requested_action" in decision:
                            _require(
                                bool(scripted)
                                and decision["requested_action"] == scripted.popleft(),
                                f"{prefix} input script mismatch",
                            )
                            _require(
                                type(decision["requested_action"]) is not int
                                or decision["requested_action"] not in legal,
                                f"{prefix} claims a failure for a legal scripted action",
                            )
                        else:
                            _require(
                                not scripted,
                                f"{prefix} claims an exhausted input script",
                            )
                else:
                    action = decision.get("action")
                    _require(
                        type(action) is int
                        and action in legal
                        and decision.get("action_string") == labels[action],
                        f"{prefix} action mismatch",
                    )
                    if decision.get("fallback_reason") is not None:
                        _require(
                            manifest["failure_policy"] == "random"
                            and action == fallback_rngs[actor].choice(legal),
                            f"{prefix} fallback mismatch",
                        )
                    elif manifest["run_type"] == "RAND" and actor != 0:
                        _require(
                            action == player_rngs[actor].choice(legal),
                            f"{prefix} random policy mismatch",
                        )
                    elif manifest["run_type"] == "INPUT" and actor != 0:
                        _require(
                            bool(scripted) and action == scripted.popleft(),
                            f"{prefix} scripted policy mismatch",
                        )
                    else:
                        try:
                            selected = parsers.parse_legal_action(
                                decision["response"],
                                labels,
                                game=record["game"],
                                player=actor,
                            )
                        except (ValueError, TypeError) as exc:
                            raise ReplayError(
                                f"{prefix} response cannot be parsed"
                            ) from exc
                        _require(
                            selected.action == action
                            and selected.summary == decision.get("summary"),
                            f"{prefix} response/action mismatch",
                        )
        if failed:
            detail = decision.get("failure")
            _require(
                decision.get("action") is None
                and decision.get("action_string") is None
                and isinstance(detail, dict)
                and isinstance(detail.get("type"), str)
                and isinstance(detail.get("message"), str)
                and detail.get("stage")
                in {"legal_actions", "observation", "policy", "model_call", "parse"}
                and type(detail.get("controlled")) is bool,
                f"{prefix} failed request metadata is invalid",
            )
            if record["status"] == "TRUNCATED":
                _require(
                    detail["controlled"]
                    and detail.get("code") == record.get("stop_reason"),
                    f"{prefix} controlled stop mismatch",
                )
            else:
                _require(
                    record["status"] == "FAILED"
                    and record.get("stop_reason") == "execution_error",
                    f"{prefix} failed episode status mismatch",
                )
                _require(
                    (
                        detail["controlled"]
                        and detail.get("code") in CONTROLLED_STOP_REASONS
                        and detail.get("code")
                        not in manifest.get("permitted_stop_reasons", [])
                    )
                    or (
                        not detail["controlled"]
                        and detail.get("code") == "execution_error"
                    ),
                    f"{prefix} failure policy mismatch",
                )
            _require(
                record.get("stop_message") == (detail["message"] or detail["type"]),
                f"{prefix} failure message mismatch",
            )
        else:
            if "decision_selection_v1" in features:
                _require(
                    "selection" in decision, f"{prefix} missing selection mechanism"
                )
            if "selection" in decision:
                _require(
                    decision["selection"] == describe_selection(decision, manifest),
                    f"{prefix} selection mechanism mismatch",
                )
    _require(
        record.get("pending_decision_count", 0) == len(decisions),
        "Episode pending decision accounting mismatch",
    )
    _require(
        record.get("pending_fallback_count", 0)
        == sum(decision.get("fallback_reason") is not None for decision in decisions),
        "Episode pending fallback accounting mismatch",
    )
    _require(
        record.get("failed_attempt_count", 0) == 1,
        "Episode failed attempt accounting mismatch",
    )


def replay_episode(record: dict[str, Any]) -> dict[str, Any]:
    """Recreate and check every transition, player view, prompt and utility.

    Live model generation is deliberately absent. The same recorded decisions
    reproduce the game even when a provider cannot reproduce its own outputs.
    """
    manifest = record.get("manifest", {})
    features = manifest.get("recorded_features", [])
    _require(
        isinstance(features, list)
        and all(
            isinstance(feature, str)
            and feature
            in {"decision_selection_v1", "openspiel_review_v1", "pending_attempts_v1"}
            for feature in features
        ),
        "Unsupported recorded evidence features",
    )
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
    game = catalog.load_game(
        spec, seed=manifest["seed"], allow_known_issues_for_replay=True
    )
    quest_limit = manifest.get("stop_after_quests")
    try:
        _validate_quest_limit(game, quest_limit)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    permitted_stops = manifest.get("permitted_stop_reasons", [])
    _require(
        isinstance(permitted_stops, list)
        and all(
            isinstance(reason, str) and reason in CONTROLLED_STOP_REASONS
            for reason in permitted_stops
        ),
        "Invalid permitted stop reasons",
    )
    _require(
        type(manifest.get("max_steps")) is int and manifest["max_steps"] > 0,
        "Invalid configured transition limit",
    )
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
        if "openspiel_review_v1" in features:
            _require(
                "review_state_before" in event and "review_state_after" in event,
                f"{prefix} missing required review state",
            )
        _require(not state.is_terminal(), f"{prefix} occurs after terminal state")
        _require(
            not _quest_limit_reached(state, quest_limit),
            f"{prefix} occurs after its configured quest limit",
        )
        _require(event.get("index") == index, f"{prefix} index mismatch")
        _require(
            event.get("state_hash_before") == state_hash(state),
            f"{prefix} input hash mismatch",
        )
        if "state_before" in event:
            snapshot = evaluator_state(state)
            _require(
                snapshot is not None and event["state_before"] == snapshot,
                f"{prefix} input state snapshot mismatch",
            )
        if "public_state_before" in event:
            board = public_state(state)
            _require(
                board is not None and event["public_state_before"] == board,
                f"{prefix} input public board mismatch",
            )
        if "review_state_before" in event:
            _require(
                event["review_state_before"] == describe_state(state),
                f"{prefix} input review state mismatch",
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
                if "decision_selection_v1" in features:
                    _require(
                        "selection" in decision,
                        f"{prefix} missing required selection mechanism",
                    )
                if "selection" in decision:
                    _require(
                        decision["selection"] == describe_selection(decision, manifest),
                        f"{prefix} selection mechanism mismatch",
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
                expects_model = manifest["run_type"] == "PVP" or player == 0
                _require(
                    isinstance(decision.get("prompt"), str)
                    if expects_model
                    else decision.get("prompt") is None,
                    f"{prefix} model prompt presence mismatch",
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
        if "state_after" in event:
            snapshot = evaluator_state(state)
            _require(
                snapshot is not None and event["state_after"] == snapshot,
                f"{prefix} output state snapshot mismatch",
            )
        if "public_state_after" in event:
            board = public_state(state)
            _require(
                board is not None and event["public_state_after"] == board,
                f"{prefix} output public board mismatch",
            )
        if "review_state_after" in event:
            _require(
                event["review_state_after"] == describe_state(state),
                f"{prefix} output review state mismatch",
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
    _verify_pending_transition(
        record, state, manifest, own_history, player_rngs, fallback_rngs, scripted
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
        record["status"]
        == (
            "FAILED"
            if isinstance(record.get("pending_transition"), dict)
            and record.get("stop_reason") == "execution_error"
            else "SUCCESS"
            if state.is_terminal()
            else "TRUNCATED"
        ),
        "Episode status mismatch",
    )
    stop_reason = record.get("stop_reason")
    _require(
        stop_reason is None or isinstance(stop_reason, str),
        "Invalid episode stop reason",
    )
    _require(
        len(events) <= manifest["max_steps"],
        "Trace exceeds its configured transition limit",
    )
    if record["status"] == "FAILED":
        _require(
            not state.is_terminal()
            and isinstance(record.get("pending_transition"), dict)
            and stop_reason == "execution_error"
            and len(events) < manifest["max_steps"]
            and not _quest_limit_reached(state, quest_limit),
            "Failed partial trace has no consistent unapplied request",
        )
    elif state.is_terminal():
        _require(
            stop_reason is None, "A terminal episode cannot claim a controlled stop"
        )
    elif stop_reason == "quest_limit":
        _require(
            _quest_limit_reached(state, quest_limit),
            "Quest stop condition was not reached",
        )
    elif stop_reason is not None:
        _require(
            stop_reason in CONTROLLED_STOP_REASONS and stop_reason in permitted_stops,
            "Episode used an unpermitted controlled stop",
        )
        _require(
            len(events) < manifest["max_steps"]
            and not _quest_limit_reached(state, quest_limit),
            "Controlled stop conflicts with a reached episode limit",
        )
    else:
        _require(
            len(events) == manifest["max_steps"],
            "Truncated trace ends before its configured limit",
        )
    if stop_reason is not None:
        _require(
            isinstance(record.get("stop_message"), str)
            and bool(record["stop_message"]),
            "Controlled stop message is missing",
        )
    else:
        _require(record.get("stop_message") is None, "Stop message has no stop reason")
    if "game_progress" in record or quest_limit is not None:
        _require(
            record.get("game_progress") == _public_game_progress(game, state),
            "Recorded public game progress mismatch",
        )
    competitive = game.get_type().utility in {
        pyspiel.GameType.Utility.ZERO_SUM,
        pyspiel.GameType.Utility.CONSTANT_SUM,
    }
    values = list(state.returns())
    winners = (
        [p for p, value in enumerate(values) if value > 0]
        if state.is_terminal()
        and getattr(game, "environment_backend", "openspiel") == "native"
        else [p for p, value in enumerate(values) if value == max(values)]
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
        "pending_transition_verified": record.get("pending_transition") is not None,
        "failure_claim_verification": "recorded_only"
        if record.get("pending_transition") is not None
        else None,
    }
