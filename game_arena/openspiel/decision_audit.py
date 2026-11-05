"""Explain recorded action selection without inventing an agent's reasoning."""

from __future__ import annotations

from typing import Any


def describe_selection(
    decision: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Classify the mechanism from recorded evidence and frozen configuration.

    This is an arena annotation, not a model explanation or proof that a remote
    provider was contacted. Original responses and metadata remain unchanged.
    """
    player = decision["player"]
    legal = decision.get("legal_actions", [])
    mode = manifest["run_type"]
    metadata = decision.get("response_metadata") or {}
    configs = manifest.get("model_configs", [])
    requested = (
        "model"
        if mode == "PVP" or player == 0
        else "random"
        if mode == "RAND"
        else "scripted"
    )
    config_index = player if mode == "PVP" else 0
    config = (
        configs[config_index]
        if requested == "model" and config_index < len(configs)
        else {}
    )
    mocked = config.get("mock")
    if not legal:
        kind, source, explanation = (
            "inactive",
            "none",
            "This seat had no legal actions and did not make a decision.",
        )
    elif decision.get("fallback_reason") is not None:
        kind, source, explanation = (
            "fallback",
            "fallback",
            "The agent failed; the arena applied its seeded random legal-action fallback.",
        )
    elif requested == "random":
        kind, source, explanation = (
            "random",
            "baseline",
            "The arena sampled a legal action using this seat's recorded seed.",
        )
    elif requested == "scripted":
        kind, source, explanation = (
            "scripted",
            "baseline",
            "The arena consumed the next action from the recorded input script.",
        )
    elif metadata.get("decision_source") == "forced_legal_action" and len(legal) == 1:
        kind, source, explanation = (
            "forced",
            "adapter",
            "The adapter applied the only legal action without a classifier call.",
        )
    elif mocked is True:
        kind, source, explanation = (
            "mock",
            "adapter" if metadata.get("summary_source") == "adapter" else "mock",
            "An offline mock policy supplied this action; it does not measure model skill.",
        )
    elif metadata.get("decision_source") == "classifier":
        kind, source, explanation = (
            "classifier",
            "adapter",
            "The adapter selected a legal action from the recorded classifier probabilities.",
        )
    else:
        kind, source, explanation = (
            "model",
            "model",
            "The arena parsed this legal action from the agent's recorded response.",
        )
    return {
        "schema_version": 1,
        "kind": kind,
        "requested_kind": requested,
        "summary_source": source,
        "choice_count": len(legal),
        "action": decision["action"],
        "explanation": explanation,
    }
