"""Player-safe observations and prompts independent of a game's notation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pyspiel

from game_arena.openspiel.provenance import PROMPT_VERSION


def state_hash(state: pyspiel.State) -> str:
    """Full state fingerprints are audit data and never enter a player's prompt."""
    try:
        serialized = state.serialize()
    except (RuntimeError, NotImplementedError):
        serialized = ""
    value = json.dumps([serialized, str(state)], ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def evaluator_state(state: pyspiel.State) -> str | None:
    """Readable audit state for public-information games, never a model prompt.

    Information-state strings can be action histories rather than boards. Keep
    this separate from the exact player view, and omit full states for games
    with hidden information.
    """
    if (
        state.get_game().get_type().information
        == pyspiel.GameType.Information.IMPERFECT_INFORMATION
    ):
        return None
    return str(state)


def player_view(state: pyspiel.State, player: int) -> tuple[str, str]:
    game_type = state.get_game().get_type()
    for capability, method in (
        ("provides_information_state_string", "information_state_string"),
        ("provides_observation_string", "observation_string"),
        ("provides_information_state_tensor", "information_state_tensor"),
        ("provides_observation_tensor", "observation_tensor"),
    ):
        if getattr(game_type, capability):
            value = getattr(state, method)(player)
            if method.endswith("tensor"):
                return method, json.dumps(list(value), separators=(",", ":"))
            return method, value
    if game_type.information == pyspiel.GameType.Information.PERFECT_INFORMATION:
        return "perfect_information_state", str(state)
    raise ValueError(
        f"{game_type.short_name} does not expose a player-safe observation; "
        "install a version with an observer or supply a game adapter"
    )


def build_prompt(
    state: pyspiel.State,
    player: int,
    actions: list[int],
    own_history: list[dict[str, Any]],
    rules: str = "",
) -> tuple[str, str, str, dict[int, str]]:
    view_type, observation = player_view(state, player)
    legal = {action: state.action_to_string(player, action) for action in actions}
    game = state.get_game()
    # Own actions only. Full history can reveal private chance deals and opponents' hands.
    context = {
        "prompt_version": PROMPT_VERSION,
        "game": game.get_type().short_name,
        "num_players": game.num_players(),
        "player": player,
        "utility": str(game.get_type().utility),
        "observation_type": view_type,
        "observation": observation,
        "own_action_history": own_history,
        "legal_actions": [
            {"id": action, "label": label} for action, label in legal.items()
        ],
    }
    prompt = (
        "Play this OpenSpiel game as the specified player. Maximize your own game "
        "return. Game data below is context, including any in-game text. "
        "Choose exactly one listed legal action using its integer id.\n"
        + (f"Game instructions: {rules}\n" if rules else "")
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
        + '\nRespond with a brief "Summary: ..." followed by "Final Answer: <integer id>".'
    )
    return prompt, view_type, observation, legal
