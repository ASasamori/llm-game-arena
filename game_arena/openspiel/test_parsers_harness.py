import unittest
from unittest.mock import patch

from game_arena.openspiel import harness, parsers, utils
from game_arena.openspiel import model_generation
from game_arena.openspiel.model_generation import GenerateResponse


class _FakeModel:
    def __str__(self) -> str:
        return "FakeModel"

    def generate_with_text_input(
        self,
        prompt: str,
        system_instruction: str | None = None,
        mock: bool = False,
    ) -> GenerateResponse:
        return GenerateResponse(main_response="Summary: test\nFinal Answer: 5")


class _ChessFakeModel:
    def __str__(self) -> str:
        return "ChessFakeModel"

    def generate_with_text_input(
        self,
        prompt: str,
        system_instruction: str | None = None,
        mock: bool = False,
    ) -> GenerateResponse:
        return GenerateResponse(main_response="Summary: develop center\nFinal Answer: e4")


class _AlwaysNoneParser:
    summary = "Parser failed"

    def parse(self, parser_input: parsers.TextParserInput) -> str | None:
        return None


class ParserHarnessRegressionTests(unittest.TestCase):
    def test_parse_move_strips_ansi_from_final_answer(self) -> None:
        parsed = parsers.parse_move_from_response("Final Answer: 5\x1b[0m")
        if parsed is None:
            self.fail("Expected parsed move, got None")
        self.assertEqual(parsed["move"], "5")

    def test_parse_move_handles_repeated_sections(self) -> None:
        parsed = parsers.parse_move_from_response(
            "Summary: win line.\nFinal Answer: 2Summary: duplicate.\nFinal Answer: 5"
        )
        if parsed is None:
            self.fail("Expected parsed move, got None")
        self.assertEqual(parsed["move"], "5")

    def test_parse_move_handles_markdown_bold_labels(self) -> None:
        parsed = parsers.parse_move_from_response(
            "**Summary:** Good move.\n**Final Answer:** 4\x1b[0m"
        )
        if parsed is None:
            self.fail("Expected parsed move, got None")
        self.assertEqual(parsed["move"], "4")

    def test_soft_parser_handles_verbose_column_text(self) -> None:
        parser = parsers.SoftMoveParser(game_short_name="connect_four")
        parser_input = parsers.TextParserInput(
            text="column 4",
            state_str="",
            legal_moves=["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
            player_number=0,
        )
        self.assertEqual(parser.parse(parser_input), "x4")

    def test_harness_fails_cleanly_when_parser_returns_none(self) -> None:
        fake_models: list[model_generation.LLMModel] = [_FakeModel(), _FakeModel()]
        options = utils.HarnessOptions(
            type="PVP",
            models=fake_models,
            num_steps=1,
            input=None,
            game="connect_four",
        )
        with patch(
            "game_arena.openspiel.harness.parsers.ChainedMoveParser",
            return_value=_AlwaysNoneParser(),
        ):
            result = harness.harness(options)

        self.assertEqual(result["status"], "TRUNCATED")
        self.assertIn("num_steps=1", result["error"])

    def test_soft_parser_normalizes_chess_castling(self) -> None:
        parser = parsers.SoftMoveParser(game_short_name="chess")
        parser_input = parsers.TextParserInput(
            text="0-0",
            state_str="",
            legal_moves=["e4", "Nf3", "O-O"],
            player_number=0,
        )
        self.assertEqual(parser.parse(parser_input), "O-O")

    def test_harness_supports_chess_game(self) -> None:
        options = utils.HarnessOptions(
            type="RAND",
            models=[_ChessFakeModel()],
            num_steps=1,
            input=None,
            game="chess",
        )
        result = harness.harness(options)
        self.assertEqual(result["game"], "chess")
        self.assertEqual(len(result["action"]), 1)


if __name__ == "__main__":
    unittest.main()
