import dataclasses
import datetime
import importlib
import json
import os
import re
from typing import Any, Protocol

from absl import logging
import tenacity

MAX_RETRY_SECONDS = int(os.environ.get("MODEL_API_RETRY_MAX_SECONDS", "90"))
if MAX_RETRY_SECONDS < 1:
    MAX_RETRY_SECONDS = 1


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


MODEL_API_MOCK_MODE = _is_truthy(os.environ.get("MODEL_API_MOCK_MODE", "false"))
MODEL_API_MOCK_CASE = os.environ.get("MODEL_API_MOCK_CASE", "default").strip().lower()
MODEL_API_MAX_RESPONSE_CHARS = int(
    os.environ.get("MODEL_API_MAX_RESPONSE_CHARS", "8000")
)
if MODEL_API_MAX_RESPONSE_CHARS < 100:
    MODEL_API_MAX_RESPONSE_CHARS = 100
MODEL_API_REQUIRE_STRUCTURED_OUTPUT = _is_truthy(
    os.environ.get("MODEL_API_REQUIRE_STRUCTURED_OUTPUT", "true")
)

MOCK_TEST_CASES: dict[str, str] = {
    # Valid shape expected by parser.
    "default": "Summary: Mock default response choosing center column.\nFinal Answer: 3",
    # Useful for parser fallback testing.
    "parser_none": "I think maybe column three but not sure.",
    # Useful for invalid parsed action fallback testing.
    "invalid_action": "Summary: Deliberately invalid move for testing.\nFinal Answer: 99",
    # Deterministic alternate legal response.
    "edge_col": "Summary: Mock response choosing edge column.\nFinal Answer: 0",
    # Deterministic legal opening move for chess.
    "chess_e4": "Summary: Mock response develops central control.\nFinal Answer: e4",
    # Explicit error path for guardrail/fallback testing.
    "raise_error": "",
}


def _mock_response_text(provider: str, prompt: str) -> str:
    case = MODEL_API_MOCK_CASE
    if case == "raise_error":
        raise RuntimeError(f"Injected mock failure for provider={provider}")
    if case == "default" and '"legal_actions"' in prompt:
        return MockModel(name=provider).generate_with_text_input(prompt).main_response
    base = MOCK_TEST_CASES.get(case, MOCK_TEST_CASES["default"])
    if case not in MOCK_TEST_CASES:
        logging.warning(
            "Unknown MODEL_API_MOCK_CASE=%s, using default case.", MODEL_API_MOCK_CASE
        )
    # Keep response deterministic but provider-tagged for traceability.
    return f"{base}\nMeta: provider={provider} prompt_chars={len(prompt)}"


# TODO: Setup retry logic for when I reach the rate limit, and can no longer use the API key
def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
    """Log a warning when retrying."""
    logging.warning(
        "Retrying API call (attempt %d) after error: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    )


def _retryable_model_error(exc: BaseException) -> bool:
    if isinstance(exc, (ValueError, TypeError)):
        return False
    status = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, ConnectionError))
        or status in {408, 429}
        or (isinstance(status, int) and status >= 500)
        or type(exc).__name__
        in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
    )


# Bound transient retries. Bad credentials and malformed output fail immediately.
_retry_decorator = tenacity.retry(
    retry=tenacity.retry_if_exception(_retryable_model_error),
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_delay(datetime.timedelta(seconds=MAX_RETRY_SECONDS))
    | tenacity.stop_after_attempt(3),
    before_sleep=_log_retry_warning,
    reraise=True,
)


@dataclasses.dataclass
class GenerateResponse:
    """Simple response container with the model's output."""

    main_response: str
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class ModelResponseGuardrailError(ValueError):
    """Raised when model output violates guardrails."""


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")
SUMMARY_LABEL_RE = re.compile(r"summary\s*:\s*", re.IGNORECASE)
FINAL_ANSWER_LABEL_RE = re.compile(r"final\s*answer\s*:\s*", re.IGNORECASE)


def _validate_model_response(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        raise ModelResponseGuardrailError("Model returned an empty response")
    if len(normalized) > MODEL_API_MAX_RESPONSE_CHARS:
        raise ModelResponseGuardrailError(
            "Model response exceeded max length "
            f"({len(normalized)} > {MODEL_API_MAX_RESPONSE_CHARS})"
        )
    if MODEL_API_REQUIRE_STRUCTURED_OUTPUT:
        cleaned = ANSI_ESCAPE_RE.sub("", normalized)
        cleaned = MARKDOWN_EMPHASIS_RE.sub("", cleaned)
        summary_matches = list(SUMMARY_LABEL_RE.finditer(cleaned))
        final_answer_matches = list(FINAL_ANSWER_LABEL_RE.finditer(cleaned))
        if not summary_matches:
            raise ModelResponseGuardrailError(
                'Model response missing required "Summary:" section'
            )
        if not final_answer_matches:
            raise ModelResponseGuardrailError(
                'Model response missing required "Final Answer:" section'
            )
        last_summary = summary_matches[-1]
        last_final = final_answer_matches[-1]
        if last_summary.start() > last_final.start():
            raise ModelResponseGuardrailError(
                'Model response must place "Summary:" before "Final Answer:"'
            )
        after_final = cleaned[last_final.end() :]
        next_summary = SUMMARY_LABEL_RE.search(after_final)
        answer_end = len(after_final) if next_summary is None else next_summary.start()
        final_answer_raw = after_final[:answer_end].strip()
        final_answer_lines = final_answer_raw.splitlines()
        final_answer_text = final_answer_lines[0].strip() if final_answer_lines else ""
        if not final_answer_text:
            raise ModelResponseGuardrailError(
                'Model response has an empty "Final Answer:" value'
            )
    return normalized


class LLMModel(Protocol):
    # Protocols are essentially an abstract base class that are not enforcing shared logic
    # Protocols are also faster than abstracts
    """Protocol for LLM model wrappers."""

    def generate_with_text_input(
        self,
        prompt: str,
        system_instruction: str | None = None,
        mock: bool = False,
    ) -> GenerateResponse: ...

    def __str__(self) -> str: ...


class MockModel:
    """Stateless offline model; no SDK import, environment lookup, key or network."""

    def __init__(self, name: str = "mock", *, model_name: str | None = None):
        self._model_name = model_name or name
        self._mock_mode = True

    def __str__(self) -> str:
        return f"MockModel({self._model_name})"

    def generate_with_text_input(
        self,
        prompt: str,
        system_instruction: str | None = None,
        mock: bool = False,
    ) -> GenerateResponse:
        # The prompt contains exactly one JSON object. raw_decode allows suffix instructions.
        start = prompt.rfind("\n{") + 1
        if not prompt[start:].startswith("{"):
            raise ValueError("MockModel requires a structured legal_actions prompt")
        payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
        legal = payload.get("legal_actions", [])
        if not legal:
            raise ValueError("MockModel requires at least one legal action")
        return GenerateResponse(
            main_response=f"Summary: Deterministic offline legal action.\nFinal Answer: {legal[0]['id']}",
            metadata={"mock": True, "model": self._model_name},
        )


class GeminiModel:
    """Simple wrapper for Google Gemini API."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        mock_mode: bool | None = None,
    ):
        self._model_name = model_name
        self._mock_mode = MODEL_API_MOCK_MODE if mock_mode is None else mock_mode
        self._client: Any | None = None
        if self._mock_mode:
            return
        # Use provided key, or fall back to GEMINI_API_KEY env var
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY environment variable or pass api_key argument."
            )
        google_genai = importlib.import_module("google.genai")
        self._client = google_genai.Client(api_key=key)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock or self._mock_mode:
            return GenerateResponse(
                main_response=_validate_model_response(
                    _mock_response_text("GEMINI", prompt)
                )
            )
        client = self._client
        if client is None:
            raise RuntimeError("Gemini client is not initialized")
        from google.genai import types as google_genai_types

        config = google_genai_types.GenerateContentConfig(
            system_instruction=system_instruction,  # type: ignore
        )

        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )

        main_response = ""
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.text:
                    main_response += part.text

        usage = response.usage_metadata
        return GenerateResponse(
            main_response=_validate_model_response(main_response),
            usage={
                "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
                "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
            },
        )

    def __str__(self) -> str:
        return f"GeminiModel({self._model_name})"


class GPTModel:
    """Simple wrapper for OpenAI GPT API."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        mock_mode: bool | None = None,
    ):
        self._model_name = model_name
        self._mock_mode = MODEL_API_MOCK_MODE if mock_mode is None else mock_mode
        self._client: Any | None = None
        if self._mock_mode:
            return
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable or pass api_key argument."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=60, max_retries=0)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock or self._mock_mode:
            return GenerateResponse(
                main_response=_validate_model_response(
                    _mock_response_text("GPT", prompt)
                )
            )
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI client is not initialized")
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
        )

        main_response = response.choices[0].message.content or ""
        usage = response.usage
        return GenerateResponse(
            main_response=_validate_model_response(main_response),
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )

    def __str__(self) -> str:
        return f"GPTModel({self._model_name})"


class ClaudeModel:
    """Simple wrapper for Anthropic Claude API."""

    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        mock_mode: bool | None = None,
    ):
        self._model_name = model_name
        self._mock_mode = MODEL_API_MOCK_MODE if mock_mode is None else mock_mode
        self._client: Any | None = None
        if self._mock_mode:
            return
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key argument."
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=key, timeout=60, max_retries=0)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock or self._mock_mode:
            return GenerateResponse(
                main_response=_validate_model_response(
                    _mock_response_text("CLAUDE", prompt)
                )
            )
        client = self._client
        if client is None:
            raise RuntimeError("Anthropic client is not initialized")

        kwargs = {
            "model": self._model_name,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        response = client.messages.create(**kwargs)

        main_response = ""
        for block in response.content:
            if block.type == "text":
                main_response += block.text

        return GenerateResponse(
            main_response=_validate_model_response(main_response),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def __str__(self) -> str:
        return f"ClaudeModel({self._model_name})"
