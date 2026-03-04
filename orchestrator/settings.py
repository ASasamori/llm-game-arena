"""
Shared environment-driven settings helpers.
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_database_target() -> str:
    raw = os.environ.get("DATABASE_TARGET", "raspberry_pi").strip().lower()
    aliases = {
        "pi": "raspberry_pi",
        "rpi": "raspberry_pi",
        "raspberrypi": "raspberry_pi",
        "cloud": "gcp",
        "google_cloud": "gcp",
        "google-cloud": "gcp",
    }
    return aliases.get(raw, raw)


def _build_database_url_for(prefix: str) -> str | None:
    host = os.environ.get(f"{prefix}_PG_HOST")
    socket_path = os.environ.get(f"{prefix}_PG_SOCKET")
    user = os.environ.get(f"{prefix}_PG_USER")
    password = os.environ.get(f"{prefix}_PG_PASSWORD")
    database = os.environ.get(f"{prefix}_PG_DATABASE", "llm_game_arena")
    port = os.environ.get(f"{prefix}_PG_PORT", "5432")
    driver = os.environ.get("DATABASE_DRIVER", "postgresql+psycopg2")

    if user and password and host:
        return (
            f"{driver}://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{database}"
        )
    if user and password and socket_path:
        return (
            f"{driver}://{quote_plus(user)}:{quote_plus(password)}@/{database}"
            f"?host={quote_plus(socket_path)}"
        )
    return None


def resolve_database_url() -> str:
    direct = os.environ.get("DATABASE_URL")
    if direct:
        return direct

    target = get_database_target()
    target_vars = {
        "raspberry_pi": "DATABASE_URL_RASPBERRY_PI",
        "gcp": "DATABASE_URL_GCP",
    }

    if target in target_vars:
        from_target_var = os.environ.get(target_vars[target])
        if from_target_var:
            return from_target_var

    if target == "raspberry_pi":
        built = _build_database_url_for("RASPBERRY_PI")
        if built:
            return built
    elif target == "gcp":
        built = _build_database_url_for("GCP")
        if built:
            return built
    else:
        unknown_var = f"DATABASE_URL_{target.upper()}"
        unknown_target = os.environ.get(unknown_var)
        if unknown_target:
            return unknown_target

    raise ValueError(
        "No database URL resolved. Set DATABASE_URL directly or configure "
        "DATABASE_TARGET with DATABASE_URL_RASPBERRY_PI / DATABASE_URL_GCP."
    )
