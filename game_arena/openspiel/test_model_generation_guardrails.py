import unittest

from game_arena.openspiel import model_generation


class ModelGenerationGuardrailTests(unittest.TestCase):
    def test_validate_accepts_summary_then_final_answer(self) -> None:
        text = "Summary: Choose center to maximize options.\nFinal Answer: 3"
        validated = model_generation._validate_model_response(text)
        self.assertEqual(validated, text)

    def test_validate_rejects_missing_summary(self) -> None:
        text = "Final Answer: 4"
        with self.assertRaises(model_generation.ModelResponseGuardrailError):
            model_generation._validate_model_response(text)

    def test_validate_rejects_wrong_section_order(self) -> None:
        text = "Final Answer: 4\nSummary: This is backwards."
        with self.assertRaises(model_generation.ModelResponseGuardrailError):
            model_generation._validate_model_response(text)

    def test_validate_rejects_empty_final_answer_value(self) -> None:
        text = "Summary: Win threat next move.\nFinal Answer:   "
        with self.assertRaises(model_generation.ModelResponseGuardrailError):
            model_generation._validate_model_response(text)


if __name__ == "__main__":
    unittest.main()
