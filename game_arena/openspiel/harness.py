"""
Harness/library for running OpenSpiel games
"""

import logging
import pyspiel
import random
import uuid
import os

from game_arena.openspiel import utils
from game_arena.openspiel import prompts
from game_arena.openspiel import model_generation
from game_arena.openspiel import parsers
from game_arena.openspiel import notations
from orchestrator.observability import increment_counter
from orchestrator.failures import FailureCategory, classify_exception

LOGGER = logging.getLogger("game_arena.openspiel.harness")

SUPPORTED_TYPES = ["INPUT", "RAND", "PVP"]
ENV = "OPENSPIEL"
HARNESS_RANDOM_SEED = os.environ.get("HARNESS_RANDOM_SEED")


def _seeded_rng() -> random.Random:
    if HARNESS_RANDOM_SEED is None:
        return random.Random()
    try:
        return random.Random(int(HARNESS_RANDOM_SEED))
    except ValueError:
        return random.Random(HARNESS_RANDOM_SEED)


# TODO: Specific featuers I want right now: AI models, turns, skills of the player (skills should be based upon prompt, so build later)
# AI_Model should be of class AIModel, will fix later
def harness(options: utils.HarnessOptions) -> utils.EpisodeResult:
    episode_id = str(uuid.uuid4())
    start_time = utils.datetime.now()
    requested_num_steps = options["num_steps"]
    game = options["game"]

    pyspiel_game = pyspiel.load_game(game)
    pyspiel_state = pyspiel_game.new_initial_state()

    game_max_steps = pyspiel_game.max_game_length()
    if requested_num_steps is None or requested_num_steps <= 0:
        max_steps = game_max_steps if game_max_steps and game_max_steps > 0 else 1000
    else:
        max_steps = requested_num_steps

    model_player_one = options["models"][0]
    type = options["type"]
    rng = _seeded_rng()
    increment_counter("harness.episode.started", tags={"run_type": type})

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
            parsers.SoftMoveParser(game_short_name=game),
        ]
    )

    action_list = []
    summary_list = []
    status = "SUCCESS"
    error = "No errors"
    for move_number in range(max_steps):
        LOGGER.debug(
            "pre_move_state",
            extra={"move_number": move_number, "state": pyspiel_state.to_string()},
        )
        if pyspiel_state.is_terminal():
            LOGGER.info("terminal_state_reached", extra={"move_number": move_number})
            break

        LOGGER.debug("move_started", extra={"move_number": move_number})
        # player_name = (
        #     str(model_player_one)
        #     if pyspiel_state.current_player() == 0
        #     else str(model_player_two)
        #     if type == "PVP"
        #     else type
        # )

        prompt_subs = {
            "game_short_name": game.replace("_", " ").title(),
            "readable_state_str": utils.convert_to_readable_state(
                game_short_name=game,
                state_str=pyspiel_state.to_string(),
                current_player=pyspiel_state.current_player(),
            ),
            "move_history": utils.get_action_string_history(pyspiel_state) or "None",
            "player_name": notations.GAME_SPECIFIC_NOTATIONS[game]["player_map"][
                pyspiel_state.current_player()
            ],
            "move_notation": notations.GAME_SPECIFIC_NOTATIONS[game]["move_notation"],
            "notation": notations.GAME_SPECIFIC_NOTATIONS[game]["state_notation"],
        }

        prompt = prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS.format(**prompt_subs)

        LOGGER.debug(
            "prompt_built",
            extra={
                "move_number": move_number,
                "current_player": pyspiel_state.current_player(),
                "prompt_chars": len(prompt),
            },
        )
        if pyspiel_state.current_player() != 0 and type != "PVP":
            if type == "RAND":
                action = rng.choice(
                    pyspiel_state.legal_actions(pyspiel_state.current_player())
                )
                action_string = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), action
                )
                LOGGER.debug(
                    "rand_action_applied",
                    extra={
                        "current_player": pyspiel_state.current_player(),
                        "action": action_string,
                    },
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
                LOGGER.debug(
                    "input_action_applied",
                    extra={
                        "current_player": pyspiel_state.current_player(),
                        "action": action_string,
                    },
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
            legal_actions = pyspiel_state.legal_actions(pyspiel_state.current_player())
            legal_action_strings = parsers.get_legal_action_strings(pyspiel_state)
            try:
                response = model.generate_with_text_input(prompt)
            except Exception as e:
                failure = classify_exception(
                    e,
                    stage="harness.model_call",
                    default_category=FailureCategory.MODEL_API,
                    default_code="model_call_failed",
                )
                increment_counter(
                    "harness.fallback.total", tags={"reason": "model_api_error"}
                )
                increment_counter(
                    "harness.failure.total",
                    tags={"category": failure.category.value, "code": failure.code},
                )
                fallback_action = rng.choice(legal_actions)
                fallback_action_str = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), fallback_action
                )
                pyspiel_state.apply_action(fallback_action)
                action_list.append(fallback_action_str)
                summary_list.append(
                    f"{failure.compact()} Random legal action fallback applied."
                )
                LOGGER.warning(
                    "model_api_fallback_action_applied",
                    extra={
                        "move_number": move_number,
                        "current_player": pyspiel_state.current_player(),
                        "fallback_action": fallback_action_str,
                        "error": str(e),
                    },
                )
                continue
            LOGGER.debug(
                "model_response_received",
                extra={
                    "move_number": move_number,
                    "current_player": pyspiel_state.current_player(),
                    "response_chars": len(response.main_response or ""),
                },
            )
            parser_input = parsers.TextParserInput(
                text=response.main_response,
                state_str=pyspiel_state.to_string(),
                legal_moves=legal_action_strings,
                player_number=pyspiel_state.current_player(),
            )
            parser_output = parser.parse(parser_input)
            if parser_output is None:
                increment_counter(
                    "harness.fallback.total", tags={"reason": "parser_none"}
                )
                increment_counter(
                    "harness.failure.total",
                    tags={
                        "category": FailureCategory.PARSER.value,
                        "code": "parser_none",
                    },
                )
                fallback_action = rng.choice(legal_actions)
                fallback_action_str = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), fallback_action
                )
                pyspiel_state.apply_action(fallback_action)
                action_list.append(fallback_action_str)
                summary_list.append(
                    "[category=parser code=parser_none stage=harness.parse "
                    "retryable=false] Parser produced no legal action; random legal action fallback."
                )
                LOGGER.warning(
                    "parser_none_fallback_action_applied",
                    extra={
                        "move_number": move_number,
                        "current_player": pyspiel_state.current_player(),
                        "fallback_action": fallback_action_str,
                    },
                )
                continue
            else:
                LOGGER.debug(
                    "parser_output",
                    extra={
                        "move_number": move_number,
                        "current_player": pyspiel_state.current_player(),
                        "parser_output": parser_output,
                    },
                )
            if parser_output not in legal_action_strings:
                increment_counter(
                    "harness.fallback.total", tags={"reason": "illegal_parser_output"}
                )
                increment_counter(
                    "harness.failure.total",
                    tags={
                        "category": FailureCategory.INVALID_ACTION.value,
                        "code": "illegal_parser_output",
                    },
                )
                fallback_action = rng.choice(legal_actions)
                fallback_action_str = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), fallback_action
                )
                pyspiel_state.apply_action(fallback_action)
                action_list.append(fallback_action_str)
                summary_list.append(
                    "[category=invalid_action code=illegal_parser_output "
                    "stage=harness.parse retryable=false] Parsed move not legal; random legal action fallback."
                )
                continue
            action_list.append(parser_output)
            summary_list.append(parser.summary)
            try:
                parsed_action = pyspiel_state.string_to_action(parser_output)
                pyspiel_state.apply_action(parsed_action)
            except Exception as e:
                failure = classify_exception(
                    e,
                    stage="harness.apply_action",
                    default_category=FailureCategory.INVALID_ACTION,
                    default_code="invalid_parsed_action",
                )
                increment_counter(
                    "harness.fallback.total", tags={"reason": "invalid_parsed_action"}
                )
                increment_counter(
                    "harness.failure.total",
                    tags={"category": failure.category.value, "code": failure.code},
                )
                # Keep episode alive by applying a random legal move.
                fallback_action = rng.choice(legal_actions)
                fallback_action_str = pyspiel_state.action_to_string(
                    pyspiel_state.current_player(), fallback_action
                )
                pyspiel_state.apply_action(fallback_action)
                # Replace bad parsed action in record with the applied fallback.
                action_list[-1] = fallback_action_str
                summary_list[-1] = (
                    f"{failure.compact()} "
                    "(invalid action applied; random legal action fallback)."
                )
                LOGGER.warning(
                    "invalid_parsed_action_fallback_applied",
                    extra={
                        "move_number": move_number,
                        "current_player": pyspiel_state.current_player(),
                        "parser_output": parser_output,
                        "fallback_action": fallback_action_str,
                        "error": str(e),
                    },
                )
                continue

    if not pyspiel_state.is_terminal() and requested_num_steps is not None:
        status = "TRUNCATED"
        error = (
            f"Reached configured num_steps={requested_num_steps} before terminal state."
        )

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
    increment_counter("harness.episode.completed", tags={"run_type": type})
    if status != "SUCCESS":
        increment_counter(
            "harness.episode.non_success_status",
            tags={"run_type": type, "status": status},
        )
    LOGGER.info(
        "episode_finished",
        extra={
            "episode_id": episode_id,
            "run_type": type,
            "status": status,
            "winner": winner,
            "num_steps": len(action_list),
            "fallback_events": sum(
                1
                for summary in summary_list
                if isinstance(summary, str) and "fallback" in summary.lower()
            ),
        },
    )
    return utils.EpisodeResult(
        episode_id=episode_id,
        env=ENV,
        game=game,
        arena_type=type,
        # Change to multiple soon, and index based on players
        # Ex: [GEM, GPT]; action_list: [x1,o2,...]
        # Ex: GEM is x, GPT is o
        # There should be a dictionary in the DB to show how many players per game in future
        agent=models_str,
        status=status,
        winner=winner,
        num_steps=len(action_list),
        error=error,
        summary=summary_list,
        action=action_list,
        start_time=start_time,
        end_time=end_time,
    )
