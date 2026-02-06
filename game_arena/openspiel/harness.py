"""
Harness/library for running OpenSpiel games
"""

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

# TODO: Add logic to get metrics of both players, just one for now
# Specific featuers I want right now: AI models, turns, skills of the player (skills should be based upon prompt, so build later)
# AI_Model should be of class AIModel, will fix later
def harness(options: utils.HarnessOptions) -> utils.EpisodeResult | None:
    # Generate dynamic UUIDs for this episode
    episode_id = str(uuid.uuid4())
    start_time = utils.datetime.now()
    NUM_MOVES = options["num_moves"]

    pyspiel_game = pyspiel.load_game("connect_four")
    pyspiel_state = pyspiel_game.new_initial_state()

    model_player_one = model_generation.GeminiModel()
    type = options["type"]
    model_player_two = model_generation.GPTModel() if type == "PVP" else None
    input_q = options["input"] if options["input"] else None
    
    # model_player_two = opt # if opt in options # else throw error

    parser = parsers.ChainedMoveParser(
        [parsers.RuleBasedMoveParser(), parsers.SoftMoveParser(game_short_name="connect_four")]
    )
    
    action_list = [] # String list
    summary_list = [] # String list
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
            # Change player name based on inputs
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
        if pyspiel_state.current_player() != 0 and type != "PVP":
            if type == "RAND":
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
                action_list.append(action_string)
                summary_list.append("Random action")
            else:
                if not input_q:
                    return None # throw error
                action = input_q.popleft()
                if action not in pyspiel_state.legal_actions():
                    return None # throw error
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
                action_list.append(action_string)
                summary_list.append("Input action")
        else:
            if pyspiel_state.current_player() == 0:
                model = model_player_one
            else:
                model = model_player_two
            response = model.generate_with_text_input(prompt) # type: ignore
            print(
                colored(
                    f"Model player {pyspiel_state.current_player()} main response:\n"
                    f" {response.main_response}",
                    "yellow",
                )
            )
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
            action_list.append(parser_output)
            summary_list.append(parser.summary)
            pyspiel_state.apply_action(pyspiel_state.string_to_action(parser_output))

    returns = pyspiel_state.returns()
    error = "No errors"
    # Redo winner logic later; especially with multiple model
    winner = (
        "AI"
        if returns[0] > returns[1]
        else "Random"
        if returns[1] > returns[0]
        else "UNDETERMINED/TIE"
    )
    end_time = utils.datetime.now()
    return utils.EpisodeResult(
        episode_id=episode_id,
        env="openspiel",
        game="connect_four",
        # Change to multiple soon, and index based on players
        # Ex: [GEM, GPT]; action_list: [x1,o2,...]
        # Ex: GEM is x, GPT is o 
        # There should be a dictionary in the DB to show how many players per game in future
        agent=options["models"], 
        status="SUCCESS",
        winner=winner,
        num_steps=NUM_MOVES,
        error=error,
        summary=summary_list,
        action=action_list,
        start_time=start_time,
        end_time=end_time,
    )


# if __name__ == "__main__":
#     app.run(main)
