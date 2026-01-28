from absl import app

import pyspiel
import termcolor
import random

from game_arena.openspiel import utils
from game_arena.openspiel import prompts
from game_arena.openspiel import model_generation
from game_arena.openspiel import parsers
from game_arena.openspiel import notations

colored = termcolor.colored

_NUM_MOVES = 3


def main(_) -> None:
    pyspiel_game = pyspiel.load_game("connect_four")
    pyspiel_state = pyspiel_game.new_initial_state()

    model_player_one = model_generation.GeminiModel()
    # TODO: Create non-rand model
    # model_player_two = "RAND"  # Random player

    parser = parsers.ChainedMoveParser(
        [parsers.RuleBasedMoveParser(), parsers.SoftMoveParser("connect_four")]
    )

    for move_number in range(_NUM_MOVES):
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


if __name__ == "__main__":
    app.run(main)
