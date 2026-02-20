#  - SQLAlchemy 2.0 style with DeclarativeBase and Mapped annotations
#  - Run model: run_id (PK), num_episodes, status, created_at
#  - Episode model: episode_id (PK), run_id (FK → runs), env, agent, game, status, winner, num_steps, error,
#  summary (JSON), action (JSON), start_time, end_time
#  - get_session() factory, create_tables() utility
#  - Run python -m orchestrator.models once to create tables


# models.py
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    # Keep runtime working in minimal environments where python-dotenv
    # is not installed but DATABASE_URL is already exported.
    def load_dotenv(*args, **kwargs):
        return False
import uuid
from sqlalchemy import Integer, String, Float, ForeignKey, DateTime, JSON, Index, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from datetime import datetime

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set. Add it to your .env file."
    )


# Base class
class Base(DeclarativeBase):
    pass

#TODO: Create Meta-run for higher level; several PVP_RR runs with different settings

# Run model
class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "PVP", "RAND", "INPUT"
    num_episodes: Mapped[int] = mapped_column(Integer, nullable=False)  # For non-PVP runs
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["Run"] = relationship("Run", back_populates="matchups")
    episodes: Mapped[list["Episode"]] = relationship("Episode", back_populates="matchup")


# Episode model
class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.run_id"), nullable=False
    )
    matchup_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("matchups.matchup_id"), nullable=True  # Null for INPUT/RAND runs
    )
    worker_name: Mapped[str | None] = mapped_column(String(100), nullable=True)  # Worker that executed this episode
    env: Mapped[str] = mapped_column(String(50))
    agent: Mapped[list] = mapped_column(JSON)
    type: Mapped[str] = mapped_column(String(50))
    game: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    winner: Mapped[str] = mapped_column(String(50))
    num_steps: Mapped[int] = mapped_column(Integer)
    error: Mapped[str] = mapped_column(String(255))
    summary: Mapped[dict] = mapped_column(JSON)
    action: Mapped[dict] = mapped_column(JSON)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[datetime] = mapped_column(DateTime)

    run: Mapped["Run"] = relationship("Run", back_populates="episodes")
    matchup: Mapped["Matchup | None"] = relationship("Matchup", back_populates="episodes")
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    episode: Mapped["Episode"] = relationship("Episode", back_populates="steps")


# Engine and session factory
engine = create_engine(DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


def get_session():
    return SessionLocal()


# Utility to create tables
def create_tables():
    Base.metadata.create_all(engine)


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
