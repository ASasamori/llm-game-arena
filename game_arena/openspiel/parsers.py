import dataclasses
import re
from typing import Protocol, Sequence
from absl import logging
import pyspiel
import game_arena.openspiel.notations as notations
import game_arena.openspiel.prompts as prompts
@dataclasses.dataclass(frozen=True, kw_only=True)
class TextParserInput:
  text: str
  state_str: str | None = None
  legal_moves: Sequence[str] | None = None
  player_number: int | None = None


def get_legal_action_strings(state: pyspiel.State) -> Sequence[str]:
  action_ints = state.legal_actions()
  return [
      state.action_to_string(state.current_player(), action_int)
      for action_int in action_ints
  ]

def parse_move_from_response(
    response: str, action_tag: str = "Final Answer: "
) -> dict[str, str] | None:
  """Extracts move, and the text before the move from response."""
  if response is None:
    return None
  try:
    last_index = -1
    final_split_token = ""
    for split_token in [action_tag, ":", "is"]:
      tmp_index = response.rfind(split_token)
      if tmp_index > last_index:
        last_index = tmp_index
        final_split_token = split_token
    if last_index == -1:
      return None
    # Split the string
    suffix = response[last_index + len(final_split_token) :]
    if suffix is None:
      return None
    move_str = (
        suffix
        .strip(" .")
        .replace("$", "")
        .replace("\\boxed{", "")
        .replace("\\text{", "")
        .replace("\boxed{", "")
        .replace("\text{", "")
        .replace("}", "")
        .replace("*", "")
        .replace(" ", "")
        .replace("`", "")
        .replace("\n", "")
    )
    clean = re.compile("<.*?>")
    move_str = re.sub(clean, "", move_str)
    if not move_str:
      return None
    prefix = response[:last_index]
    return {
        "move": move_str,
        "prefix": "" if prefix is None else prefix.strip(),
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error(
        "parse_move_from_response failed with input response %s", response
    )
    raise e

class TextParser(Protocol):
  """Generic text parser."""

  # TODO(google-deepmind): should we define a parser output type?
  def parse(self, parser_input: TextParserInput) -> str | None:
    ...

class ChainedMoveParser(TextParser):
  """Parses text into a move by successively applying parsers.

  For example, parsing with a LLM followed by parsing of the structured output
  with a rule-based parser or with a soft-parser against legal moves.
  """

  def __init__(self, parsers: Sequence[TextParser]):
    self._parsers = parsers

  def parse(self, parser_input: TextParserInput) -> str | None:
    maybe_text_output = None
    for parser in self._parsers:
      maybe_text_output = parser.parse(parser_input)
      if maybe_text_output is None:
        return None
      parser_input = dataclasses.replace(parser_input, text=maybe_text_output)
    return maybe_text_output

class RuleBasedMoveParser(TextParser):
  """Parses text into a move with string processing built into Python."""

  def __init__(self, action_tag: str = "Final Answer: "):
    self._action_tag = action_tag

  def parse(self, parser_input: TextParserInput) -> str | None:
    maybe_move_and_prefix = parse_move_from_response(
        parser_input.text, action_tag=self._action_tag
    )
    if maybe_move_and_prefix is None:
      return None
    return maybe_move_and_prefix["move"]

class SoftMoveParser(TextParser):
  """Parses text into a move by soft matching against legal moves."""

  def __init__(self, game_short_name: str):
    self._game_short_name = game_short_name

  def parse(self, parser_input: TextParserInput) -> str | None:
    if (
        parser_input.state_str is None
        or parser_input.legal_moves is None
        or parser_input.player_number is None
    ):
      raise ValueError(
          "State as a string, legal moves, and player number are required for"
          " soft move parsing."
      )

    selected_action = parser_input.text

    if not selected_action:
      return None

    if (
        self._game_short_name.startswith("tic_tac_toe")
        or self._game_short_name.startswith("connect_four")
        or self._game_short_name.startswith("go")
    ):
      player_name = notations.GAME_SPECIFIC_NOTATIONS[
          self._game_short_name
      ]["player_map"][parser_input.player_number]
      if self._game_short_name.startswith("go"):
        # OpenSpiel Go action string is lowercased.
        selected_action = (
            player_name
            + " "
            + selected_action.strip("(),").replace(",", "").lower()
        )
      elif self._game_short_name.startswith("tic_tac_toe"):
        selected_action = (
            player_name
            + ("" if selected_action.startswith("(") else "(")
            + selected_action.replace(" ", "")
            + ("" if selected_action.endswith(")") else ")")
        )
      else:
        selected_action = player_name + selected_action
    elif self._game_short_name.startswith("universal_poker"):
      selected_action = "player={} move={}".format(
          parser_input.player_number, selected_action
      )
    if selected_action not in parser_input.legal_moves:
      return None
    return selected_action