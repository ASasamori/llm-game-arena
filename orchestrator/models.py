#  - SQLAlchemy 2.0 style with DeclarativeBase and Mapped annotations
#  - Run model: run_id (PK), num_episodes, status, created_at
#  - Episode model: episode_id (PK), run_id (FK → runs), env, agent, game, status, winner, num_steps, error,
#  summary (JSON), action (JSON), start_time, end_time
#  - get_session() factory, create_tables() utility
#  - Run python -m orchestrator.models once to create tables
from typing import Any, Callable
import uuid
import os
import threading
from sqlalchemy import (
    Integer,
    Boolean,
    Text,
    String,
    Float,
    ForeignKey,
    DateTime,
    JSON,
    Index,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from datetime import UTC, datetime
from orchestrator.settings import get_database_target, resolve_database_url

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:
    # Keep runtime working in minimal environments where python-dotenv
    # is not installed but DATABASE_URL is already exported.
    def _load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


load_dotenv: Callable[..., bool] = _load_dotenv

load_dotenv()

DATABASE_URL = resolve_database_url()
DATABASE_TARGET = get_database_target()


# Base class
class Base(DeclarativeBase):
    pass


# TODO: Create Meta-run for higher level; several PVP_RR runs with different settings


# Run model
class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_type: Mapped[str] = mapped_column(
        String(10), nullable=False
    )  # "PVP", "RAND", "INPUT"
    num_episodes: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # For non-PVP runs
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    # Relationships
    matchups: Mapped[list["Matchup"]] = relationship("Matchup", back_populates="run")
    episodes: Mapped[list["Episode"]] = relationship("Episode", back_populates="run")


# Matchup model - tracks a specific model pair within a run
class Matchup(Base):
    __tablename__ = "matchups"

    matchup_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), nullable=False
    )
    models: Mapped[list] = mapped_column(JSON)  # ["GEMINI", "GPT"]
    num_episodes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    run: Mapped["Run"] = relationship("Run", back_populates="matchups")
    episodes: Mapped[list["Episode"]] = relationship(
        "Episode", back_populates="matchup"
    )


# Episode model
class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (
        Index("ix_episodes_run_status", "run_id", "status"),
        Index("ix_episodes_matchup_status", "matchup_id", "status"),
    )

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), nullable=False
    )
    matchup_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("matchups.matchup_id"),
        nullable=True,  # Null for INPUT/RAND runs
    )
    worker_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # Worker that executed this episode
    env: Mapped[str] = mapped_column(String(50))
    agent: Mapped[list] = mapped_column(JSON)
    type: Mapped[str] = mapped_column(String(50))
    game: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    winner: Mapped[str] = mapped_column(String(50))
    num_steps: Mapped[int] = mapped_column(Integer)
    error: Mapped[str] = mapped_column(String(255))
    summary: Mapped[list[str]] = mapped_column(JSON)
    action: Mapped[list[str]] = mapped_column(JSON)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime] = mapped_column(DateTime)

    run: Mapped["Run"] = relationship("Run", back_populates="episodes")
    matchup: Mapped["Matchup | None"] = relationship(
        "Matchup", back_populates="episodes"
    )
    steps: Mapped[list["EpisodeStep"]] = relationship(
        "EpisodeStep", back_populates="episode", cascade="all, delete-orphan"
    )


# EpisodeStep model - normalized step-level data from episodes
class EpisodeStep(Base):
    __tablename__ = "episode_steps"
    __table_args__ = (Index("ix_episode_steps_episode_id", "episode_id"),)

    step_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    episode_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("episodes.episode_id", ondelete="CASCADE"),
        nullable=False,
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    player: Mapped[str] = mapped_column(String(10), nullable=False)  # "x" or "o"
    action: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "x3"
    action_probs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    episode: Mapped["Episode"] = relationship("Episode", back_populates="steps")


class RunConfiguration(Base):
    """Immutable scheduling manifest, separate from the legacy runs schema."""

    __tablename__ = "run_configurations"
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), primary_key=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class RunDispatch(Base):
    """Durable outbox marker bridging database commit and Redis submission."""

    __tablename__ = "run_dispatches"
    __table_args__ = (Index("ix_run_dispatch_pending", "complete", "updated_at"),)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), primary_key=True
    )
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )


class RunProgress(Base):
    """Constant-time completion accounting for large runs."""

    __tablename__ = "run_progress"
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), primary_key=True
    )
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MatchupProgress(Base):
    __tablename__ = "matchup_progress"
    matchup_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("matchups.matchup_id"), primary_key=True
    )
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EpisodeTrace(Base):
    """Versioned result and provenance, excluding separately paginated transitions."""

    __tablename__ = "episode_traces"
    episode_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("episodes.episode_id", ondelete="CASCADE"),
        primary_key=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EpisodeTransition(Base):
    """One complete engine transition; player IDs are explicit, never inferred."""

    __tablename__ = "episode_transitions"
    episode_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("episodes.episode_id", ondelete="CASCADE"),
        primary_key=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ModelRating(Base):
    """
    Current Elo-style rating snapshot for a model in a specific game.
    """

    __tablename__ = "model_ratings"
    __table_args__ = (
        UniqueConstraint("model_name", "game", name="uq_model_ratings_model_game"),
        Index("ix_model_ratings_game_rating", "game", "rating"),
    )

    rating_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    game: Mapped[str] = mapped_column(String(50), nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)
    games_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_as_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    games_as_o: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins_as_x: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wins_as_o: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )


class RatingEvent(Base):
    """
    Immutable Elo update event for a single episode result.
    `episode_id` is unique to guarantee idempotent ingestion.
    """

    __tablename__ = "rating_events"
    __table_args__ = (
        UniqueConstraint("episode_id", name="uq_rating_events_episode_id"),
        Index("ix_rating_events_game_created", "game", "created_at"),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), nullable=False
    )
    episode_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("episodes.episode_id", ondelete="CASCADE"),
        nullable=False,
    )
    game: Mapped[str] = mapped_column(String(50), nullable=False)
    model_a: Mapped[str] = mapped_column(String(100), nullable=False)
    model_b: Mapped[str] = mapped_column(String(100), nullable=False)
    player_x_model: Mapped[str] = mapped_column(String(100), nullable=False)
    player_o_model: Mapped[str] = mapped_column(String(100), nullable=False)
    winner_model: Mapped[str] = mapped_column(String(100), nullable=False)

    score_a: Mapped[float] = mapped_column(Float, nullable=False)
    expected_a: Mapped[float] = mapped_column(Float, nullable=False)
    k_factor_a: Mapped[float] = mapped_column(Float, nullable=False)
    k_factor_b: Mapped[float] = mapped_column(Float, nullable=False)
    model_a_pre: Mapped[float] = mapped_column(Float, nullable=False)
    model_b_pre: Mapped[float] = mapped_column(Float, nullable=False)
    model_a_post: Mapped[float] = mapped_column(Float, nullable=False)
    model_b_post: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    run: Mapped["Run"] = relationship("Run")
    episode: Mapped["Episode"] = relationship("Episode")


# Engine and session factory
engine = create_engine(
    DATABASE_URL,
    echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
    future=True,
)
SessionLocal = sessionmaker(bind=engine, future=True)
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


def get_session():
    return SessionLocal()


# Utility to create tables
def create_tables():
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            # API and runner may bootstrap the same fresh database concurrently.
            # PostgreSQL serializes DDL discovery and creation across processes.
            connection.execute(text("SELECT pg_advisory_xact_lock(736274601)"))
        Base.metadata.create_all(connection)
        # create_all does not add indexes on already-existing legacy tables.
        for model in (Episode, RunDispatch):
            for index in model.__table__.indexes:
                index.create(connection, checkfirst=True)


def ensure_tables_created() -> None:
    """
    Idempotently initialize schema for this process.

    Enabled by default so fresh Cloud SQL instances can self-bootstrap.
    Set DB_AUTO_INIT_SCHEMA=false to disable.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    auto_init = os.environ.get("DB_AUTO_INIT_SCHEMA", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not auto_init:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        create_tables()
        _SCHEMA_READY = True


def migrate_actions_to_episode_steps():
    """
    Migration function to normalize action data from episodes into episode_steps.
    Iterates over existing episodes and creates one EpisodeStep row per action.
    Does not modify existing episode records.
    """
    session = get_session()
    try:
        # Get all episodes that have actions
        episodes = session.query(Episode).filter(Episode.action.isnot(None)).all()

        for episode in episodes:
            # Skip if already migrated (has steps)
            existing_steps = (
                session.query(EpisodeStep)
                .filter(EpisodeStep.episode_id == episode.episode_id)
                .first()
            )
            if existing_steps:
                continue

            actions = episode.action
            if not isinstance(actions, list):
                continue

            for step_index, action_str in enumerate(actions):
                if not action_str or not isinstance(action_str, str):
                    continue

                # Extract player from first character of action string
                player = action_str[0].lower() if action_str else ""

                step = EpisodeStep(
                    step_id=str(uuid.uuid4()),
                    episode_id=episode.episode_id,
                    step_index=step_index,
                    player=player,
                    action=action_str,
                    action_probs=None,
                    entropy=None,
                )
                session.add(step)

        session.commit()
        print(f"Migration complete. Processed {len(episodes)} episodes.")
    except Exception as e:
        session.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        session.close()


# Run this script to create tables
if __name__ == "__main__":
    create_tables()
