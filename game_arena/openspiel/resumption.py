"""Continue a validated Avalon prefix without requesting any prior decision again.

This module owns no credentials, spending ledger, database, or queue. Callers
retain the same durable provider ledger and use an independent once-only claim
for the resume attempt. Only new decisions reach the supplied live models.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

from game_arena.openspiel.harness import harness
from game_arena.openspiel.model_generation import GenerateResponse, LLMModel
from game_arena.openspiel.replay import replay_episode
from game_arena.openspiel.utils import EpisodeResult, HarnessOptions


class ResumeError(ValueError):
    """A saved prefix cannot safely precede another provider request."""


@dataclass
class _ReplayOrder:
    players: deque[int]


class PrefixReplayModel:
    """Reuse one seat's exact recorded responses before delegating new calls."""

    def __init__(
        self,
        player: int,
        model: LLMModel,
        cached: list[dict[str, Any]],
        order: _ReplayOrder,
    ):
        self.player, self.model = player, model
        self.cached = deque(deepcopy(cached))
        self.order = order
        self._model_name = getattr(model, "_model_name", None)
        self._mock_mode = getattr(model, "_mock_mode", None)

    def __str__(self) -> str:
        return str(self.model)

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        if self.cached:
            if not self.order.players or self.order.players[0] != self.player:
                raise ResumeError("Recorded player decision order changed")
            record = self.cached[0]
            if prompt != record["prompt"] or system_instruction is not None or mock:
                raise ResumeError(
                    "Recorded prompt or request mode changed before resume"
                )
            self.cached.popleft()
            self.order.players.popleft()
            return GenerateResponse(
                record["response"],
                usage=deepcopy(record.get("usage", {})),
                metadata=deepcopy(record.get("response_metadata", {})),
            )
        if self.order.players:
            raise ResumeError("A new request would precede unreplayed saved decisions")
        return self.model.generate_with_text_input(prompt, system_instruction, mock)


def _without_latency(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_latency(item)
            for key, item in value.items()
            if key != "latency_ms"
        }
    if isinstance(value, (list, tuple)):
        return [_without_latency(item) for item in value]
    return value


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    def encode(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Unexpected trace value: {type(value).__name__}")

    return json.loads(json.dumps(record, default=encode, allow_nan=False))


def resume_episode(previous: dict[str, Any], models: list[LLMModel]) -> EpisodeResult:
    """Replay the exact saved prefix, then ask only for subsequent decisions.

    Only clean native Avalon PVP traces stopped by a provider error are admitted.
    The transition/quest limits, episode identity, seed and player assignments
    remain frozen. Original prefix records (including actual latencies) survive
    unchanged, and the complete returned episode is independently replayed.
    """
    if (
        previous.get("status") != "TRUNCATED"
        or previous.get("terminal") is not False
        or previous.get("stop_reason") != "provider_error"
        or previous.get("arena_type") != "PVP"
        or previous.get("game") != "avalon"
        or previous.get("env") != "NATIVE"
        or previous.get("degraded")
        or previous.get("fallback_count", 0)
    ):
        raise ResumeError(
            "Only clean provider-stopped native Avalon PVP prefixes can resume"
        )
    # This verifies every saved state, action, observation, prompt and outcome
    # before constructing a wrapper that could delegate a live provider call.
    replay_episode(previous)
    manifest = previous["manifest"]
    if manifest.get("failure_policy") != "raise":
        raise ResumeError("Resumption requires the original strict failure policy")
    identities = [str(model) for model in models]
    if identities != previous["player_agents"] or len(models) != len(
        manifest["model_configs"]
    ):
        raise ResumeError("Resume model identities or seat count changed")
    for model, stored in zip(models, manifest["model_configs"], strict=True):
        if getattr(model, "_model_name", None) != stored.get("model_name") or getattr(
            model, "_mock_mode", None
        ) != stored.get("mock"):
            raise ResumeError("Resume model version or mock mode changed")
    prefix = previous["steps"]
    if any(event["kind"] not in {"chance", "decision"} for event in prefix):
        raise ResumeError(
            "Avalon resumption supports only sequential explicit-chance traces"
        )
    cached: list[list[dict[str, Any]]] = [[] for _ in models]
    order = _ReplayOrder(deque())
    for event in prefix:
        for decision in event["decisions"]:
            if not isinstance(decision.get("prompt"), str) or not isinstance(
                decision.get("response"), str
            ):
                raise ResumeError("Saved model request or response is unavailable")
            cached[decision["player"]].append(decision)
            order.players.append(decision["player"])
    wrappers: list[LLMModel] = [
        PrefixReplayModel(p, model, cached[p], order) for p, model in enumerate(models)
    ]
    options = HarnessOptions(
        type="PVP",
        models=wrappers,
        game=previous["game_spec"],
        num_steps=manifest["max_steps"],
        input=None,
        seed=manifest["seed"],
        episode_id=previous["episode_id"],
        failure_policy=manifest["failure_policy"],
        rules=manifest.get("rules", ""),
        metadata=deepcopy(manifest.get("metadata", {})),
        permitted_stop_reasons=deepcopy(manifest.get("permitted_stop_reasons", [])),
    )
    if manifest.get("stop_after_quests") is not None:
        options["stop_after_quests"] = manifest["stop_after_quests"]
    result = harness(options)
    steps = result.get("steps", [])
    if order.players or len(steps) < len(prefix):
        raise ResumeError("Resume ended before the saved prefix was reconstructed")
    for old, regenerated in zip(prefix, steps, strict=False):
        comparable = deepcopy(regenerated)
        # Public boards are additive audit evidence. Earlier saved prefixes
        # lack them; retain those records verbatim while validating every field
        # that was originally recorded before continuing with new decisions.
        for key in (
            "public_state_before",
            "public_state_after",
            "review_state_before",
            "review_state_after",
        ):
            if key not in old:
                comparable.pop(key, None)
        for saved_decision, new_decision in zip(
            old["decisions"], comparable["decisions"], strict=True
        ):
            if "selection" not in saved_decision:
                new_decision.pop("selection", None)
        if _without_latency(old) != _without_latency(comparable):
            raise ResumeError("Saved transition changed during prefix reconstruction")
    steps[: len(prefix)] = deepcopy(prefix)
    if any(
        "selection" not in decision
        for event in steps
        for decision in event["decisions"]
    ):
        resumed_manifest = result.setdefault("manifest", {})
        features = resumed_manifest.get("recorded_features", [])
        resumed_manifest["recorded_features"] = [
            feature for feature in features if feature != "decision_selection_v1"
        ]
    original_start = previous["start_time"]
    result["start_time"] = (
        datetime.fromisoformat(original_start)
        if isinstance(original_start, str)
        else original_start
    )
    history = deepcopy(manifest.get("resume_history", []))
    history.append(
        {
            "source_revision": manifest.get("source_revision"),
            "source_sha256": manifest.get("source_sha256"),
            "prefix_transitions": len(prefix),
            "prefix_decisions": previous.get("decision_count", 0),
            "replayed_provider_requests": 0,
            **(
                {"pending_transition": deepcopy(previous["pending_transition"])}
                if previous.get("pending_transition") is not None
                else {}
            ),
        }
    )
    result.setdefault("manifest", {})["resume_history"] = history
    replay_episode(_json_record(dict(result)))
    return result
