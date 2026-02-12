#  - SQLAlchemy 2.0 style with DeclarativeBase and Mapped annotations
#  - Run model: run_id (PK), num_episodes, status, created_at
#  - Episode model: episode_id (PK), run_id (FK → runs), env, agent, game, status, winner, num_steps, error,
#  summary (JSON), action (JSON), start_time, end_time
#  - get_session() factory, create_tables() utility
#  - Run python -m orchestrator.models once to create tables


# models.py
import os

from dotenv import load_dotenv
from sqlalchemy import Integer, String, ForeignKey, DateTime, JSON, create_engine
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


# Engine and session factory
engine = create_engine(DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


def get_session():
    return SessionLocal()


# Utility to create tables
def create_tables():
    Base.metadata.create_all(engine)


# Run this script to create tables
if __name__ == "__main__":
    create_tables()
