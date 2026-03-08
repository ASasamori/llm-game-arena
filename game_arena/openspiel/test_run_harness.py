import unittest
from unittest.mock import patch

from game_arena.openspiel import run_harness
from game_arena.openspiel import utils


class RunHarnessBackoffTests(unittest.TestCase):
    def test_run_with_backoff_retries_and_succeeds(self) -> None:
        options = utils.HarnessOptions(
            type="RAND", models=[], num_steps=2, input=None, game="connect_four"
        )
        calls = {"count": 0}

        def flaky_run(_options: utils.HarnessOptions) -> utils.EpisodeResult:
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("transient API error")
            return utils.EpisodeResult(
                episode_id="ep",
                env="OPENSPIEL",
                agent=[],
                arena_type="RAND",
                game="connect_four",
                status="SUCCESS",
                winner="UNDETERMINED/TIE",
                num_steps=0,
                error="No errors",
                summary=[],
                action=[],
                start_time=utils.datetime.now(),
                end_time=utils.datetime.now(),
            )

        sleep_calls: list[float] = []
        with patch(
            "game_arena.openspiel.run_harness.harness.harness", side_effect=flaky_run
        ):
            result = run_harness.run_with_backoff(
                options,
                max_attempts=4,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.02,
                sleep_fn=lambda seconds: sleep_calls.append(seconds),
            )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(calls["count"], 3)
        self.assertEqual(len(sleep_calls), 2)

    def test_run_with_backoff_does_not_retry_non_retryable_value_error(self) -> None:
        options = utils.HarnessOptions(
            type="RAND", models=[], num_steps=2, input=None, game="connect_four"
        )
        with patch(
            "game_arena.openspiel.run_harness.harness.harness",
            side_effect=ValueError("bad config"),
        ):
            with self.assertRaises(ValueError):
                run_harness.run_with_backoff(
                    options,
                    max_attempts=4,
                    initial_backoff_seconds=0.01,
                    max_backoff_seconds=0.02,
                    sleep_fn=lambda _seconds: None,
                )


if __name__ == "__main__":
    unittest.main()
