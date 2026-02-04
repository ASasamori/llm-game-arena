#  - SQLAlchemy 2.0 style with DeclarativeBase and Mapped annotations                                             
#  - Run model: run_id (PK), num_episodes, status, created_at                                                     
#  - Episode model: episode_id (PK), run_id (FK → runs), env, agent, game, status, winner, num_steps, error,      
#  summary (JSON), action (JSON), start_time, end_time                                                            
#  - get_session() factory, create_tables() utility                                                               
#  - Run python -m orchestrator.models once to create tables    


# models.py
import os

from dotenv import load_dotenv
from sqlalchemy import (
    Column, Integer, String, ForeignKey, DateTime, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from datetime import datetime

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Add it to your .env file.")

# Base class
class Base(DeclarativeBase):
    pass

# Run model
class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    num_episodes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship to episodes
    episodes: Mapped[list["Episode"]] = relationship("Episode", back_populates="run")

# Episode model
class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"), nullable=False)
    env: Mapped[str] = mapped_column(String(50))
    agent: Mapped[str] = mapped_column(String(50))
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