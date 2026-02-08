import pyspiel
from typing import TypedDict
from collections import deque
from datetime import datetime

from game_arena.openspiel import model_generation

class EpisodeResult(TypedDict):
    episode_id: str
    env: str
    agent: list[str]
    arena_type: str
    game: str
    status: str
    winner: str
    num_steps: int
    error: str
    summary: list[str]
    action: list[str]
    start_time: datetime
    end_time: datetime

class HarnessOptions(TypedDict):
    type: str
    models: list[model_generation.LLMModel]
    num_steps: int
    input: deque | None


def get_action_string_history(state: pyspiel.State) -> str:
    """Returns a string history of actions."""
    game_name = state.get_game().get_type().short_name

    tmp_state = state.get_game().new_initial_state()
    move_list = []
    for action in state.history():
        action_str = tmp_state.action_to_string(tmp_state.current_player(), action)
        move_list.append(action_str)
        tmp_state.apply_action(action)

    # Use 'None' for empty history, matching C++ handler's initial state output.
    if game_name.startswith("universal_poker") or game_name.startswith("go"):
        move_history = ", ".join(move_list) if move_list else "None"
    else:
        move_history = " ".join(move_list) if move_list else "None"
    return move_history


def convert_to_readable_state(
    *, game_short_name: str, state_str: str, current_player: int
) -> str:
    """Converts a state to a readable string."""
    if game_short_name.startswith("universal_poker"):
        lines = state_str.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("ACPC State:"):
                continue
            elif line.startswith("P0 Cards:") and current_player == 1:
                continue
            elif line.startswith("P1 Cards:") and current_player == 0:
                continue
            else:
                new_lines.append(line)
        return "\n".join(new_lines)
    elif game_short_name.startswith("tic_tac_toe") or game_short_name.startswith(
        "connect_four"
    ):
        return "\n".join(
            [" ".join(list(line.strip())) for line in state_str.split("\n")]
        )
    else:
        return state_str
