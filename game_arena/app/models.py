"""
Pydantic models and data structures for the Mafia game engine.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    """Player roles in the game."""
    MAFIA = "mafia"
    VILLAGER = "villager"
    DOCTOR = "doctor"
    SHERIFF = "sheriff"


class PlayerStatus(str, Enum):
    """Player status in the game."""
    ALIVE = "alive"
    DEAD = "dead"


class GamePhase(str, Enum):
    """Current phase of the game."""
    SETUP = "setup"
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    GAME_OVER = "game_over"


class Player(BaseModel):
    """Represents a player in the game."""
    id: int
    name: str
    model_id: str
    avatar: str
    role: Optional[Role] = None
    status: PlayerStatus = PlayerStatus.ALIVE
    private_memory: list[str] = Field(default_factory=list)

    def add_memory(self, memory: str) -> None:
        """Add a private memory note for this player."""
        self.private_memory.append(memory)


class TranscriptEntry(BaseModel):
    """A single entry in the public transcript."""
    speaker: Optional[str] = None  # None for system messages
    content: str
    is_system: bool = False
    is_secret: bool = False  # If True, players cannot see this, only spectators
    inner_monologue: Optional[str] = None  # Only shown to spectator, not other players


class AIResponse(BaseModel):
    """Structured response from the AI."""
    inner_monologue: str
    public_speech: str


class VoteResult(BaseModel):
    """Result of a voting round."""
    votes: dict[str, str]  # voter_name -> target_name or "skip"
    eliminated: Optional[str] = None
    was_tie: bool = False


class GameState(BaseModel):
    """The complete state of the game."""
    day_count: int = 0
    phase: GamePhase = GamePhase.SETUP
    players: list[Player] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    graveyard: list[str] = Field(default_factory=list)  # Names of dead players
    current_speaker_index: int = 0
    discussion_turns_remaining: int = 0
    
    # New fields for granular phase processing
    current_mafia_speaker_index: int = 0
    mafia_chat_rounds_remaining: int = 0
    current_voter_index: int = 0
    votes: dict[str, str] = Field(default_factory=dict) # voter_name -> target
    
    winner: Optional[str] = None  # "mafia", "town", or None

    def get_alive_players(self) -> list[Player]:
        """Return list of alive players."""
        return [p for p in self.players if p.status == PlayerStatus.ALIVE]

    def get_mafia_players(self) -> list[Player]:
        """Return list of mafia players (alive or dead)."""
        return [p for p in self.players if p.role == Role.MAFIA]

    def get_alive_mafia(self) -> list[Player]:
        """Return list of alive mafia players."""
        return [p for p in self.get_alive_players() if p.role == Role.MAFIA]

    def get_alive_town(self) -> list[Player]:
        """Return list of alive town players."""
        return [p for p in self.get_alive_players() if p.role != Role.MAFIA]

    def add_system_message(self, content: str, is_secret: bool = False, inner_monologue: Optional[str] = None) -> None:
        """Add a system message to the transcript."""
        self.transcript.append(TranscriptEntry(content=content, is_system=True, is_secret=is_secret, inner_monologue=inner_monologue))

    def add_player_message(self, speaker: str, content: str, inner_monologue: str) -> None:
        """Add a player message to the transcript."""
        self.transcript.append(TranscriptEntry(
            speaker=speaker,
            content=content,
            is_system=False,
            inner_monologue=inner_monologue
        ))

    def check_win_condition(self) -> Optional[str]:
        """Check if the game has ended. Returns 'mafia', 'town', or None."""
        alive_mafia = len(self.get_alive_mafia())
        alive_town = len(self.get_alive_town())

        if alive_mafia == 0:
            return "town"
        if alive_mafia >= alive_town:
            return "mafia"
        return None
