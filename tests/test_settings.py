import os
import unittest
from unittest.mock import patch

from orchestrator.settings import get_database_target, resolve_database_url


class SettingsTests(unittest.TestCase):
    def test_resolve_database_url_prefers_direct_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql+psycopg2://direct-user:pw@localhost:5432/db",  # pragma: allowlist secret -- synthetic test credential or sentinel only
                "DATABASE_TARGET": "gcp",
                "DATABASE_URL_GCP": "postgresql+psycopg2://ignored:pw@host:5432/db",  # pragma: allowlist secret -- synthetic local test credential
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_database_url(),
                "postgresql+psycopg2://direct-user:pw@localhost:5432/db",  # pragma: allowlist secret -- synthetic local test credential
            )

    def test_resolve_database_url_uses_target_specific_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_TARGET": "raspberry_pi",
                "DATABASE_URL_RASPBERRY_PI": "postgresql+psycopg2://pi:pw@100.116.74.50:5432/llm_game_arena",  # pragma: allowlist secret -- synthetic local test credential
                "DATABASE_URL_GCP": "postgresql+psycopg2://gcp:pw@gcp-host:5432/llm_game_arena",  # pragma: allowlist secret -- synthetic local test credential
            },
            clear=True,
        ):
            self.assertIn("100.116.74.50", resolve_database_url())

    def test_resolve_database_url_builds_from_parts(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_TARGET": "gcp",
                "GCP_PG_HOST": "10.8.0.7",
                "GCP_PG_PORT": "5432",
                "GCP_PG_USER": "arena_user",
                "GCP_PG_PASSWORD": "arena pass",  # pragma: allowlist secret -- synthetic test credential or sentinel only
                "GCP_PG_DATABASE": "llm_game_arena",
            },
            clear=True,
        ):
            url = resolve_database_url()
            self.assertTrue(url.startswith("postgresql+psycopg2://"))
            self.assertIn("10.8.0.7:5432/llm_game_arena", url)

    def test_database_target_aliases(self) -> None:
        with patch.dict(os.environ, {"DATABASE_TARGET": "rpi"}, clear=True):
            self.assertEqual(get_database_target(), "raspberry_pi")
        with patch.dict(os.environ, {"DATABASE_TARGET": "google-cloud"}, clear=True):
            self.assertEqual(get_database_target(), "gcp")


if __name__ == "__main__":
    unittest.main()
