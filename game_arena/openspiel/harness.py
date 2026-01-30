'''
Harness/library for running OpenSpiel games
Not sure if absl/app will work properly with queuing systems, however
'''
from absl import app

import pyspiel
import termcolor
from typing import TypedDict
import random
import uuid

from game_arena.openspiel import utils
from game_arena.openspiel import prompts
from game_arena.openspiel import model_generation
from game_arena.openspiel import parsers
from game_arena.openspiel import notations

colored = termcolor.colored

class EpisodeResult(TypedDict):
    episode_id: str
    env: str
    agent: str
    # Not sure what seed should be used for
    game: str
    status: str
    winner: str
    num_steps: int
    error: str


# TODO: Add logic to get metrics of both players, just one for now
# Specific featuers I want right now: AI models, turns, skills of the player (skills should be based upon prompt, so build later)
# AI_Model should be of class AIModel, will fix later
def main(NUM_MOVES: int, AI_MODEL: str) -> EpisodeResult:
    
    # Generate dynamic UUIDs for this episode
    episode_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    # _NUM_MOVES = 3
    pyspiel_game = pyspiel.load_game("connect_four")
    pyspiel_state = pyspiel_game.new_initial_state()

    model_player_one = model_generation.GeminiModel()
    # TODO: Create non-rand model
    # model_player_two = "RAND"  # Random player

    parser = parsers.ChainedMoveParser(
        [parsers.RuleBasedMoveParser(), parsers.SoftMoveParser("connect_four")]
    )

    for move_number in range(NUM_MOVES):
        print(f"Pre-move debug string: {pyspiel_state.to_string()}")
        if pyspiel_state.is_terminal():
            print(colored("Game is terminal, ending move loop.", "red"))
            break

        print(colored(f"Commencing move {move_number}...", "green"))

        prompt_subs = {
            "game_short_name": "Connect Four",
            "readable_state_str": utils.convert_to_readable_state(
                game_short_name="connect_four",
                state_str=pyspiel_state.to_string(),
                current_player=pyspiel_state.current_player(),
            ),
            "move_history": utils.get_action_string_history(pyspiel_state) or "None",
            "player_name": notations.game_player_map[pyspiel_state.current_player()],
            "move_notation": notations.GAME_SPECIFIC_NOTATIONS["connect_four"][
                "move_notation"
            ],
            "notation": notations.GAME_SPECIFIC_NOTATIONS["connect_four"][
                "state_notation"
            ],
        }

        prompt = prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS.format(**prompt_subs)

        print(colored(f"Prompt: {prompt}", "blue"))
        if pyspiel_state.current_player() == 0:
            response = model_player_one.generate_with_text_input(prompt)
            print(
                colored(
                    f"Model player {pyspiel_state.current_player()} main response:\n"
                    f" {response.main_response}",
                    "yellow",
                )
            )
            # TODO: Look into how this  actually works!
            parser_input = parsers.TextParserInput(
                text=response.main_response,
                state_str=pyspiel_state.to_string(),
                legal_moves=parsers.get_legal_action_strings(pyspiel_state),
                player_number=pyspiel_state.current_player(),
            )
            parser_output = parser.parse(parser_input)
            if parser_output is None:
                print(colored("Parser output is None, ending game", "red"))
            else:
                print(colored(f"Parser output: {parser_output}", "magenta"))
            pyspiel_state.apply_action(pyspiel_state.string_to_action(parser_output))
        else:
            # Random player
            action = random.choice(
                pyspiel_state.legal_actions(pyspiel_state.current_player())
            )
            action_string = pyspiel_state.action_to_string(
                pyspiel_state.current_player(), action
            )
            print(
                "Player ",
                pyspiel_state.current_player(),
                ", randomly sampled action: ",
                action_string,
            )
            pyspiel_state.apply_action(action)
    
    returns = pyspiel_state.returns()
    error = "No errors"
    
    # TODO: Add logic to get winner, just one for now
    winner = "AI" if returns[0] > returns[1] else "Random" if returns[1] > returns[0] else "TIE"
    return EpisodeResult(
        episode_id=episode_id,
        env="openspiel",
        game="connect_four",
        agent="GEMINI",
        status="SUCCESS",
        winner=winner,
        num_steps=NUM_MOVES,
        error=error,
    )


# if __name__ == "__main__":
#     app.run(main)
