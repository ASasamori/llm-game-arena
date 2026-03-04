import dataclasses
import re
from typing import Protocol, Sequence
from absl import logging
import pyspiel
import game_arena.openspiel.notations as notations

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")
SUMMARY_RE = re.compile(r"summary\s*:\s*", re.IGNORECASE)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TextParserInput:
    text: str
    summary: str | None = None  # TODO: Look into this, probably why NULL is fine
    state_str: str | None = None
    legal_moves: Sequence[str] | None = None
    player_number: int | None = None


def get_legal_action_strings(state: pyspiel.State) -> Sequence[str]:
    action_ints = state.legal_actions()
    return [
        state.action_to_string(state.current_player(), action_int)
        for action_int in action_ints
    ]


# TODO: Fix incorrect summary logic to be a string, if not found, currently returns NULL
def parse_move_from_response(
    response: str, action_tag: str = "Final Answer: ", want_summary: bool = True
) -> dict[str, str] | None:
    """Extracts move, and the text before the move from response."""
    if response is None:
        return None
    try:
        cleaned_response = ANSI_ESCAPE_RE.sub("", response)
        cleaned_response = MARKDOWN_EMPHASIS_RE.sub("", cleaned_response)
        action_tag_pattern = r"\s*".join(map(re.escape, action_tag.strip().split()))
        final_answer_re = re.compile(action_tag_pattern + r"\s*", re.IGNORECASE)

        # Find the last final-answer tag to handle repeated generations.
        action_matches = list(final_answer_re.finditer(cleaned_response))
        if not action_matches:
            return None
        action_match = action_matches[-1]
        action_idx = action_match.start()

        after_action = cleaned_response[action_match.end() :]

        # Extract move: everything up to newline or "Summary:"
        summary_match = SUMMARY_RE.search(after_action)
        summary_idx = summary_match.start() if summary_match else -1
        newline_idx = after_action.find("\n")

        # Find the end of the move (first of: Summary:, newline, or end of string)
        end_idx = len(after_action)
        if summary_idx != -1:
            end_idx = min(end_idx, summary_idx)
        if newline_idx != -1:
            end_idx = min(end_idx, newline_idx)

        move_str = after_action[:end_idx].strip()

        # Clean up the move string
        move_str = (
            move_str.strip(" .")
            .replace("$", "")
            .replace("\\boxed{", "")
            .replace("\\text{", "")
            .replace("\boxed{", "")
            .replace("\text{", "")
            .replace("}", "")
            .replace("*", "")
            .replace(" ", "")
            .replace("`", "")
        )
        clean = re.compile("<.*?>")
        move_str = re.sub(clean, "", move_str)

        if not move_str:
            return None

        prefix = cleaned_response[:action_idx]

        # Extract summary
        summary = None
        if want_summary:
            full_summary_match = SUMMARY_RE.search(cleaned_response)
            if full_summary_match is not None:
                after_summary = cleaned_response[full_summary_match.end() :]
                # Take up to 3 sentences
                sentences = re.findall(r"[^.]*\.", after_summary)
                summary = "".join(sentences[:3]).strip()

        return {
            "move": move_str,
            "prefix": "" if prefix is None else prefix.strip(),
            "summary": "" if summary is None else summary,
        }
    except Exception as e:
        logging.error(
            "parse_move_from_response failed with input response %s", response
        )
        raise e


class TextParser(Protocol):
    """Generic text parser."""

    def parse(self, parser_input: TextParserInput) -> str | None: ...


# TODO: Implement fallback parsing if it doesn't parse correctly
class ChainedMoveParser(TextParser):
    """Parses text into a move by successively applying parsers.

    For example, parsing with a LLM followed by parsing of the structured output
    with a rule-based parser or with a soft-parser against legal moves.
    """

    def __init__(self, parsers: Sequence[TextParser]):
        self._parsers = parsers
        self._last_summary: str | None = None

    def parse(self, parser_input: TextParserInput) -> str | None:
        maybe_text_output = None
        for parser in self._parsers:
            maybe_text_output = parser.parse(parser_input)
            if maybe_text_output is None:
                return None
            # Carry summary forward, update text for next parser
            parser_input = dataclasses.replace(parser_input, text=maybe_text_output)
        # Store the summary so the caller can access it
        self._last_summary = parser_input.summary
        if not self._last_summary:
            self._last_summary = "This summary was not supplied by the model."
        return maybe_text_output

    @property
    def summary(self) -> str | None:
        return self._last_summary


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
        # Stash summary on parser_input so the chain can carry it forward
        summary = maybe_move_and_prefix["summary"]
        if not summary:
            summary = "This summary was not supplied by the model."
        object.__setattr__(parser_input, "summary", summary)
        return maybe_move_and_prefix["move"]


class SoftMoveParser(TextParser):
    """Parses text into a move by soft matching against legal moves."""

    def __init__(self, game_short_name: str):
        self._game_short_name = game_short_name

    @staticmethod
    def _normalize_chess_move(text: str) -> str:
        selected_action = (text or "").strip()
        if not selected_action:
            return ""

        selected_action = selected_action.splitlines()[0].strip().strip("`")
        selected_action = re.sub(
            r"^(?:move|play|answer|final\s*answer)\s*[:=-]\s*",
            "",
            selected_action,
            flags=re.IGNORECASE,
        )
        selected_action = selected_action.rstrip(".;,")
        selected_action = selected_action.replace("0-0-0", "O-O-O").replace(
            "0-0", "O-O"
        )
        selected_action = selected_action.replace("o-o-o", "O-O-O").replace(
            "o-o", "O-O"
        )
        selected_action = re.sub(r"[!?+#]+$", "", selected_action)

        # Normalize piece-letter case for SAN-like text, e.g. "nf3" -> "Nf3".
        if selected_action and selected_action[0] in "kqrbn":
            selected_action = selected_action[0].upper() + selected_action[1:]

        # Normalize promotion suffix, e.g. "e8=q" -> "e8=Q".
        selected_action = re.sub(
            r"=([qrbn])$",
            lambda m: "=" + m.group(1).upper(),
            selected_action,
            flags=re.IGNORECASE,
        )
        return selected_action

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
            player_name = notations.GAME_SPECIFIC_NOTATIONS[self._game_short_name][
                "player_map"
            ][parser_input.player_number]
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
                if selected_action not in parser_input.legal_moves:
                    # Fallback for verbose responses like "column 4".
                    maybe_col = re.search(r"-?\d+", parser_input.text)
                    if maybe_col is not None:
                        selected_action = player_name + maybe_col.group(0)
        elif self._game_short_name.startswith("universal_poker"):
            selected_action = "player={} move={}".format(
                parser_input.player_number, selected_action
            )
        elif self._game_short_name.startswith("chess"):
            normalized_action = self._normalize_chess_move(selected_action)
            if normalized_action in parser_input.legal_moves:
                selected_action = normalized_action
            else:
                # Last-pass fallback: pick legal SAN token explicitly present in text.
                text = parser_input.text or ""
                for legal_move in sorted(
                    parser_input.legal_moves, key=len, reverse=True
                ):
                    if re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(legal_move)}(?![A-Za-z0-9])",
                        text,
                    ):
                        selected_action = legal_move
                        break
        if selected_action not in parser_input.legal_moves:
            return None
        return selected_action
