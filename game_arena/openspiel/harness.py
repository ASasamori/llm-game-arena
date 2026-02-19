"""
Harness/library for running OpenSpiel games
"""

import pyspiel
import termcolor
import random
import uuid

from game_arena.openspiel import utils
from game_arena.openspiel import prompts
from game_arena.openspiel import model_generation
from game_arena.openspiel import parsers
from game_arena.openspiel import notations

colored = termcolor.colored

SUPPORTED_TYPES = ["INPUT", "RAND", "PVP"]
ENV = "OPENSPIEL"
GAME = "connect_four"


# TODO: Specific featuers I want right now: AI models, turns, skills of the player (skills should be based upon prompt, so build later)
# AI_Model should be of class AIModel, will fix later
def harness(options: utils.HarnessOptions) -> utils.EpisodeResult:
    episode_id = str(uuid.uuid4())
    start_time = utils.datetime.now()
    NUM_STEPS = options["num_steps"]

    pyspiel_game = pyspiel.load_game(GAME)
    pyspiel_state = pyspiel_game.new_initial_state()

    model_player_one = options["models"][0]
    type = options["type"]

    if type not in SUPPORTED_TYPES:
        raise ValueError(f"Type: {type} is not supported")

    if type == "INPUT" and not options["input"]:
        raise ValueError(f"Type is {type} and the input queue is {options['input']}")

    if type == "PVP" and not len(options["models"]) > 1:
        raise ValueError(f"Need more models other than just {options['models']}")

    model_player_two = options["models"][1] if options["type"] == "PVP" else None

    parser = parsers.ChainedMoveParser(
        [
            parsers.RuleBasedMoveParser(),
            parsers.SoftMoveParser(game_short_name="connect_four"),
        ]
    )

    action_list = []
    summary_list = []
    status = "SUCCESS"
    error = "No errors"
    for move_number in range(NUM_STEPS):
        print(f"Pre-move debug string: {pyspiel_state.to_string()}")
        if pyspiel_state.is_terminal():
            print(colored("Game is terminal, ending move loop.", "red"))
            break

        print(colored(f"Commencing move {move_number}...", "green"))
        player_name = (
            str(model_player_one)
            if pyspiel_state.current_player() == 0
            else str(model_player_two)
            if type == "PVP"
            else type
        )

        prompt_subs = {
            "game_short_name": "Connect Four",
            "readable_state_str": utils.convert_to_readable_state(
                game_short_name="connect_four",
                state_str=pyspiel_state.to_string(),
                current_player=pyspiel_state.current_player(),
            ),
            "move_history": utils.get_action_string_history(pyspiel_state) or "None",
            "player_name": notations.GAME_SPECIFIC_NOTATIONS['connect_four']['player_map'][pyspiel_state.current_player()],
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
                    colored(
                        f"Player {pyspiel_state.current_player()}"
                        f"randomly sampled action: {action_string}",
                        "light_cyan",
                    )
                )
                pyspiel_state.apply_action(action)
                action_list.append(action_string)
                summary_list.append("Random action")
            elif type == "INPUT":
                # TODO: Fix this unreachable ValueError
                if not options["input"]:
                    raise ValueError("Impossible")
                action = options["input"].popleft()
                if action not in pyspiel_state.legal_actions():
                    raise ValueError(f"The input of {action} is not in legal actions")
                action_string = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), action
                )
                print(
                    colored(
                        f"Player {pyspiel_state.current_player()},"
                        f"randomly sampled action: {action_string}",
                        "light_magenta",
                    )
                )
                pyspiel_state.apply_action(action)
                action_list.append(action_string)
                summary_list.append("Input action")
            else:
                raise ValueError(f"There is no such mode called {options['type']}")
        else:
            if pyspiel_state.current_player() == 0:
                model: model_generation.LLMModel = model_player_one
            else:
                if model_player_two is None:
                    raise ValueError("Player 2 model is None but game type requires it")
                model = model_player_two
            response = model.generate_with_text_input(prompt)
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
                status = "FAILED"
                error = (
                    "Parser produced no legal action string from model response. "
                    f"Move {move_number}, player {pyspiel_state.current_player()}."
                )
                summary_list.append(parser.summary or "Parser failed")
                break
            else:
                print(colored(f"Parser output: {parser_output}", "magenta"))
            action_list.append(parser_output)
            summary_list.append(parser.summary)
            try:
                parsed_action = pyspiel_state.string_to_action(parser_output)
                pyspiel_state.apply_action(parsed_action)
            except Exception as e:
                status = "FAILED"
                error = (
                    "Failed to apply parsed action. "
                    f"Move {move_number}, action={parser_output!r}, "
                    f"player {pyspiel_state.current_player()}: {e}"
                )
                print(colored(error, "red"))
                break

    returns = pyspiel_state.returns()
    # Redo for mmultiple models now (will inherently be different from different games though)
    winner = (
        str(options["models"][0])
        if returns[0] > returns[1]
        else str(options["models"][1])
        if returns[1] > returns[0] and len(options["models"]) > 1
        else "UNDETERMINED/TIE"
        if returns[1] == returns[0]
        else options["type"]
    )

    models_str = (
        [str(model) for model in options["models"]]
        if type == "PVP"
        else [str(options["models"][0])]
    )
    end_time = utils.datetime.now()
    return utils.EpisodeResult(
        episode_id=episode_id,
        env=ENV,
        game=GAME,
        arena_type=type,
        # Change to multiple soon, and index based on players
        # Ex: [GEM, GPT]; action_list: [x1,o2,...]
        # Ex: GEM is x, GPT is o
        # There should be a dictionary in the DB to show how many players per game in future
        agent=models_str,
        status=status,
        winner=winner,
        num_steps=NUM_STEPS,
        error=error,
        summary=summary_list,
        action=action_list,
        start_time=start_time,
        end_time=end_time,
    )
