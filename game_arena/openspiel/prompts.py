import enum


@enum.unique
class PromptTemplate(enum.Enum):
    """Prompt template."""

    WITH_LEGAL_ACTIONS = "WITH_LEGAL_ACTIONS"
    NO_LEGAL_ACTIONS = "NO_LEGAL_ACTIONS"
    NO_LEGAL_ACTIONS_NO_HISTORY = "NO_LEGAL_ACTIONS_NO_HISTORY"
    NO_LEGAL_ACTIONS_RETHINK_APPENDED = "NO_LEGAL_ACTIONS_RETHINK_APPENDED"
    NO_LEGAL_ACTIONS_WITH_PIECE_DICT = "NO_LEGAL_ACTIONS_WITH_PIECE_DICT"
    NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED = (
        "NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED"
    )
    NO_LEGAL_ACTIONS_WITH_ASCII_BOARD = "NO_LEGAL_ACTIONS_WITH_ASCII_BOARD"
    NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED = (
        "NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED"
    )
    WITH_BOARD_IMAGE = "WITH_BOARD_IMAGE"
    WITH_SVG_RENDERED_IMAGE = "WITH_SVG_RENDERED_IMAGE"

    @classmethod
    def from_string(cls, value: str) -> "PromptTemplate":
        return PromptTemplate[value]


def is_image_text(prompt_template: PromptTemplate) -> bool:
    """Returns whether the prompt template is image and text."""
    return prompt_template in (
        PromptTemplate.WITH_BOARD_IMAGE,
        PromptTemplate.WITH_SVG_RENDERED_IMAGE,
    )


PROMPT_TEMPLATE_NO_LEGAL_ACTIONS = (
    PROMPT_TEMPLATE_NO_LEGAL_ACTIONS
) = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. 
Reason step by step to come up with your move, 
then output your final answer in the format "Final Answer: X" 
where X is your chosen move in {move_notation}.
Finally, give a 1-3 sentence summary explaining your thought process and next possible actions in the format "Summary: Z".
Remember, Summary and then Final Answer.

IMPORTANT: You MUST include both "Summary:" and "Final Answer:" in your response in that order, respectively."""
