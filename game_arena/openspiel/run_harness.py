import argparse
import random
import time
from typing import Callable

from game_arena.openspiel import harness
from game_arena.openspiel import model_generation
from game_arena.openspiel import utils


def _build_model(provider: str, model_name: str | None) -> model_generation.LLMModel:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "claude":
        return model_generation.ClaudeModel(
            model_name=model_name or "claude-3-haiku-20240307"
        )
    if provider_normalized == "gemini":
        return model_generation.GeminiModel(
            model_name=model_name or "gemini-2.5-flash-lite"
        )
    if provider_normalized == "gpt":
        return model_generation.GPTModel(model_name=model_name or "gpt-4o-mini")
    raise ValueError(f"Unsupported model provider: {provider}")


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, ValueError):
        return False
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    return True


def _retry_backoff_delay_seconds(
    attempt_number: int, initial: float, max_delay: float
) -> float:
    capped = min(max_delay, initial * (2 ** (attempt_number - 1)))
    # Add small jitter to reduce synchronized retries.
    return capped + random.uniform(0.0, capped * 0.2)


def run_with_backoff(
    options: utils.HarnessOptions,
    *,
    max_attempts: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> utils.EpisodeResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if initial_backoff_seconds <= 0:
        raise ValueError("initial_backoff_seconds must be > 0")
    if max_backoff_seconds <= 0:
        raise ValueError("max_backoff_seconds must be > 0")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return harness.harness(options)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_exception(exc) or attempt >= max_attempts:
                raise
            delay = _retry_backoff_delay_seconds(
                attempt, initial_backoff_seconds, max_backoff_seconds
            )
            print(
                "[run_harness] attempt "
                f"{attempt}/{max_attempts} failed ({type(exc).__name__}: {exc}). "
                f"Retrying in {delay:.2f}s."
            )
            sleep_fn(delay)
    if last_error is None:
        raise RuntimeError("Harness run failed with an unknown error")
    raise RuntimeError(
        f"Harness run failed after retries: {last_error}"
    ) from last_error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenSpiel harness with retries/backoff."
    )
    parser.add_argument("--mode", default="PVP", choices=["PVP", "RAND", "INPUT"])
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument(
        "--player-one-provider",
        default="claude",
        choices=["claude", "gemini", "gpt"],
    )
    parser.add_argument("--player-one-model", default=None)
    parser.add_argument(
        "--player-two-provider",
        default="gemini",
        choices=["claude", "gemini", "gpt"],
    )
    parser.add_argument("--player-two-model", default=None)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_one = _build_model(args.player_one_provider, args.player_one_model)
    models: list[model_generation.LLMModel] = [model_one]
    if args.mode == "PVP":
        model_two = _build_model(args.player_two_provider, args.player_two_model)
        models.append(model_two)

    options = utils.HarnessOptions(
        type=args.mode,
        num_steps=args.num_steps,
        models=models,
        input=None,
    )

    result = run_with_backoff(
        options,
        max_attempts=args.max_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
