import dataclasses
import datetime
import os
from typing import Protocol

from absl import logging
from dotenv import find_dotenv, load_dotenv
from google import genai as google_genai
from google.genai import types as google_genai_types
from anthropic import Anthropic
import tenacity

load_dotenv(find_dotenv(usecwd=True), override=True)

MAX_RETRY_SECONDS = int(os.environ.get("MODEL_API_RETRY_MAX_SECONDS", "90"))
if MAX_RETRY_SECONDS < 1:
    MAX_RETRY_SECONDS = 1


# TODO: Setup retry logic for when I reach the rate limit, and can no longer use the API key
def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
    """Log a warning when retrying."""
    logging.warning(
        "Retrying API call (attempt %d) after error: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    )


# Retry decorator for handling rate limits and transient errors
_retry_decorator = tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_delay(datetime.timedelta(seconds=MAX_RETRY_SECONDS)),
    before_sleep=_log_retry_warning,
    reraise=True,
)


@dataclasses.dataclass
class GenerateResponse:
    """Simple response container with the model's output."""

    main_response: str


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


class GeminiModel:
    """Simple wrapper for Google Gemini API."""

    def __init__(
        self, model_name: str = "gemini-2.5-flash-lite", api_key: str | None = None
    ):
        self._model_name = model_name
        # Use provided key, or fall back to GEMINI_API_KEY env var
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY environment variable or pass api_key argument."
            )
        self._client = google_genai.Client(api_key=key)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock:
            return GenerateResponse(main_response="TestingGEMINI")
        config = google_genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
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

        return GenerateResponse(main_response=main_response)

    def __str__(self) -> str:
        return f"GeminiModel({self._model_name})"


class GPTModel:
    """Simple wrapper for OpenAI GPT API."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI

        self._model_name = model_name
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable or pass api_key argument."
            )
        self._client = OpenAI(api_key=key)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock:
            return GenerateResponse(main_response="TestingGPT")
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
        )

        main_response = response.choices[0].message.content or ""
        return GenerateResponse(main_response=main_response)

    def __str__(self) -> str:
        return f"GPTModel({self._model_name})"


class ClaudeModel:
    """Simple wrapper for Anthropic Claude API."""

    def __init__(
        self, model_name: str = "claude-3-haiku-20240307", api_key: str | None = None
    ):
        self._model_name = model_name
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key argument."
            )
        self._client = Anthropic(api_key=key)

    @_retry_decorator
    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        """Generate a response from a text prompt."""
        if mock:
            return GenerateResponse(main_response="TestingCLAUDE")

        kwargs = {
            "model": self._model_name,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        response = self._client.messages.create(**kwargs)

        main_response = ""
        for block in response.content:
            if block.type == "text":
                main_response += block.text

        return GenerateResponse(main_response=main_response)

    def __str__(self) -> str:
        return f"ClaudeModel({self._model_name})"
