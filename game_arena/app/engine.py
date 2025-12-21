"""
Core game engine - manages game state, phases, and rules.
"""
import json
import random
import asyncio
from pathlib import Path
from collections import Counter

from app.models import (
    Player, Role, PlayerStatus, GamePhase, GameState, VoteResult
)
from app import agents
from app.llm_client import RateLimitError


def load_player_config() -> list[dict]:
    """Load player configuration from JSON file."""
    config_path = Path(__file__).parent.parent / "player_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["players"]


def initialize_game() -> GameState:
    """
    Initialize a new game with players and randomly assigned roles.
    """
    player_configs = load_player_config()

    # Create Player objects
    players = [
        Player(
            id=i,
            name=cfg["name"],
            model_id=cfg["model_id"],
            avatar=cfg["avatar"]
        )
        for i, cfg in enumerate(player_configs)
    ]

    # Assign roles: 3 Mafia, 1 Doctor, 1 Sheriff, 5 Villagers
    roles = (
        [Role.MAFIA] * 3 +
        [Role.DOCTOR] +
        [Role.SHERIFF] +
        [Role.VILLAGER] * 5
    )
    random.shuffle(roles)

    for player, role in zip(players, roles):
        player.role = role

    # Initialize private memory for Mafia (they know each other)
    mafia_names = [p.name for p in players if p.role == Role.MAFIA]
    for player in players:
        if player.role == Role.MAFIA:
            teammates = [n for n in mafia_names if n != player.name]
            player.add_memory(f"You are MAFIA. Your teammates are: {', '.join(teammates)}")

    game_state = GameState(
        day_count=0,
        phase=GamePhase.SETUP,
        players=players
    )

    game_state.add_system_message("🎭 Welcome to Mafia! The game is about to begin...")
    game_state.add_system_message(f"📋 {len(players)} players have gathered: {', '.join(p.name for p in players)}")

    return game_state


async def generate_mafia_chat_turn(game_state: GameState) -> dict:
    """
    Generate a single turn of Mafia private discussion.
    """
    alive_mafia = game_state.get_alive_mafia()
    if not alive_mafia or game_state.mafia_chat_rounds_remaining <= 0:
        return None

    # Get current speaker
    speaker_index = game_state.current_mafia_speaker_index % len(alive_mafia)
    speaker = alive_mafia[speaker_index]

    # Get chat history for context
    mafia_chat_history = [
        e.content.replace("🕵️ [MAFIA CHAT] ", "")
        for e in game_state.transcript 
        if e.is_secret and "[MAFIA CHAT]" in e.content
    ]

    # Get AI response
    statement = await agents.get_mafia_discussion_turn(speaker, game_state, mafia_chat_history)
    
    # Add to transcript (Spectator Only)
    entry_text = f"{speaker.name}: {statement}"
    game_state.add_system_message(f"🕵️ [MAFIA CHAT] {entry_text}", is_secret=True)

    # Add to private memory of all mafia immediately
    for m in alive_mafia:
        m.add_memory(f"Night {game_state.day_count} Chat - {entry_text}")

    # Update state
    game_state.current_mafia_speaker_index += 1
    
    # If we finished a full loop of alive mafia, decrement rounds
    if game_state.current_mafia_speaker_index % len(alive_mafia) == 0:
        game_state.mafia_chat_rounds_remaining -= 1

    return {
        "speaker": speaker.name,
        "avatar": speaker.avatar,
        "content": statement
    }


async def resolve_night_actions(game_state: GameState) -> dict:
    """
    Resolve night actions: Kill, Save, Investigate.
    """
    result = {
        "mafia_target": None,
        "doctor_save": None,
        "sheriff_target": None,
        "sheriff_result": None,
        "death": None,
        "saved": False
    }

    alive_mafia = game_state.get_alive_mafia()
    alive_town = game_state.get_alive_town()

    if not alive_mafia or not alive_town:
        return result

    # Get chat history for final decision context (if needed by agents)
    mafia_chat_history = [
        e.content.replace("🕵️ [MAFIA CHAT] ", "")
        for e in game_state.transcript 
        if e.is_secret and "[MAFIA CHAT]" in e.content
    ]

    # --- Sequential Action Resolution ---
    action_results = {}
    
    # 1. Mafia Kill
    mafia_leader = alive_mafia[0]
    action_results["mafia"] = await agents.get_mafia_kill_target(mafia_leader, game_state, mafia_chat_history)
    
    # 2. Doctor Save
    doctor = next((p for p in game_state.get_alive_players() if p.role == Role.DOCTOR), None)
    if doctor:
        action_results["doctor"] = await agents.get_doctor_save_target(doctor, game_state)
        
    # 3. Sheriff Investigation
    sheriff = next((p for p in game_state.get_alive_players() if p.role == Role.SHERIFF), None)
    if sheriff:
        action_results["sheriff"] = await agents.get_sheriff_investigate_target(sheriff, game_state)
    
    # Process Results & Log to Transcript (Spectator Mode)
    
    # 1. Log Actions (Spectator Only)
    if "mafia" in action_results:
        result["mafia_target"] = action_results["mafia"]
        game_state.add_system_message(f"🔫 [MAFIA TARGET] The Mafia decided to kill: {result['mafia_target']}", is_secret=True)
        
    if "doctor" in action_results:
        doc_resp = action_results["doctor"]
        if isinstance(doc_resp, dict):
            doctor_save = doc_resp.get("target")
            doc_monologue = doc_resp.get("monologue")
        else:
            doctor_save = doc_resp
            doc_monologue = None
            
        result["doctor_save"] = doctor_save
        if doctor:
            doctor.add_memory(f"Night {game_state.day_count}: You protected {doctor_save}")
            game_state.add_system_message(
                f"💉 [DOCTOR] {doctor.name} protected: {doctor_save}", 
                is_secret=True,
                inner_monologue=doc_monologue
            )
            
    if "sheriff" in action_results:
        sheriff_resp = action_results["sheriff"]
        if isinstance(sheriff_resp, dict):
            sheriff_target = sheriff_resp.get("target")
            sheriff_monologue = sheriff_resp.get("monologue")
        else:
            sheriff_target = sheriff_resp
            sheriff_monologue = None
            
        result["sheriff_target"] = sheriff_target
        if sheriff and sheriff_target:
            target_player = next((p for p in game_state.players if p.name == sheriff_target), None)
            if target_player:
                is_mafia = target_player.role == Role.MAFIA
                result["sheriff_result"] = "MAFIA" if is_mafia else "TOWN"
                sheriff.add_memory(
                    f"Night {game_state.day_count}: You investigated {sheriff_target} - they are {result['sheriff_result']}"
                )
                game_state.add_system_message(
                    f"🔎 [SHERIFF] {sheriff.name} investigated {sheriff_target}: {result['sheriff_result']}", 
                    is_secret=True,
                    inner_monologue=sheriff_monologue
                )

    # --- Resolve death ---
    mafia_target = result["mafia_target"]
    if result["doctor_save"] == mafia_target and mafia_target is not None:
        result["saved"] = True
        result["death"] = None
    elif mafia_target:
        result["death"] = mafia_target
        target_player = next((p for p in game_state.players if p.name == mafia_target), None)
        if target_player:
            target_player.status = PlayerStatus.DEAD
            game_state.graveyard.append(target_player.name)

    return result


async def generate_day_turn(game_state: GameState) -> dict:
    """
    Generate a single turn of day discussion.
    Returns the current speaker's response.
    """
    alive_players = game_state.get_alive_players()

    if not alive_players or game_state.discussion_turns_remaining <= 0:
        return None

    # Get current speaker
    speaker_index = game_state.current_speaker_index % len(alive_players)
    speaker = alive_players[speaker_index]

    # Prevent same speaker twice in a row (unless only 1 left)
    if len(alive_players) > 1 and game_state.transcript:
        last_entry = game_state.transcript[-1]
        if last_entry.speaker == speaker.name:
            # Skip to next player
            game_state.current_speaker_index += 1
            speaker_index = game_state.current_speaker_index % len(alive_players)
            speaker = alive_players[speaker_index]

    # Get AI response
    response = await agents.get_player_speech(speaker, game_state)

    # Add to transcript (only public speech goes to other players)
    game_state.add_player_message(
        speaker=speaker.name,
        content=response.public_speech,
        inner_monologue=response.inner_monologue
    )

    # Update state
    game_state.current_speaker_index += 1
    game_state.discussion_turns_remaining -= 1

    return {
        "speaker": speaker.name,
        "avatar": speaker.avatar,
        "role": speaker.role.value,
        "inner_monologue": response.inner_monologue,
        "public_speech": response.public_speech
    }


def get_next_speaker(game_state: GameState) -> Player | None:
    """
    Predict the next active player based on the current phase and sub-turn.
    """
    if game_state.phase == GamePhase.NIGHT:
        if game_state.mafia_chat_rounds_remaining > 0:
            alive_mafia = game_state.get_alive_mafia()
            if alive_mafia:
                return alive_mafia[game_state.current_mafia_speaker_index % len(alive_mafia)]
        return None  # No "speaker" during action resolution

    elif game_state.phase == GamePhase.DAY_DISCUSSION:
        alive_players = game_state.get_alive_players()
        if not alive_players or game_state.discussion_turns_remaining <= 0:
            return None

        speaker_index = game_state.current_speaker_index % len(alive_players)
        speaker = alive_players[speaker_index]

        if len(alive_players) > 1 and game_state.transcript:
            last_entry = game_state.transcript[-1]
            if last_entry.speaker == speaker.name:
                speaker_index = (game_state.current_speaker_index + 1) % len(alive_players)
                speaker = alive_players[speaker_index]
        return speaker

    elif game_state.phase == GamePhase.DAY_VOTING:
        alive_players = game_state.get_alive_players()
        if game_state.current_voter_index < len(alive_players):
            return alive_players[game_state.current_voter_index]
        return None

    return None


async def resolve_voting_turn(game_state: GameState) -> dict:
    """
    Get a single player's vote.
    """
    alive_players = game_state.get_alive_players()
    if game_state.current_voter_index >= len(alive_players):
        return None

    voter = alive_players[game_state.current_voter_index]
    
    # Get vote
    vote_target = await agents.get_player_vote(voter, game_state)
    game_state.votes[voter.name] = vote_target

    # Update state
    game_state.current_voter_index += 1

    return {
        "voter": voter.name,
        "target": vote_target
    }


def finalize_voting(game_state: GameState) -> VoteResult:
    """
    Tally the votes and determine the outcome.
    """
    vote_counts = Counter(v for v in game_state.votes.values() if v != "skip")
    total_alive = len(game_state.get_alive_players())

    result = VoteResult(votes=game_state.votes.copy())

    if vote_counts:
        top_voted, top_count = vote_counts.most_common(1)[0]
        if top_count > total_alive / 2:
            result.eliminated = top_voted
            target = next((p for p in game_state.players if p.name == top_voted), None)
            if target:
                target.status = PlayerStatus.DEAD
                game_state.graveyard.append(target.name)
        else:
            top_votes = [name for name, count in vote_counts.items() if count == top_count]
            if len(top_votes) > 1:
                result.was_tie = True

    return result


def advance_phase(game_state: GameState) -> None:
    """
    Advance the game to the next phase and initialize turn counters.
    """
    if game_state.phase == GamePhase.SETUP:
        game_state.phase = GamePhase.NIGHT
        game_state.day_count = 1
        game_state.current_mafia_speaker_index = 0
        game_state.mafia_chat_rounds_remaining = 2

    elif game_state.phase == GamePhase.NIGHT:
        game_state.phase = GamePhase.DAY_DISCUSSION
        game_state.current_speaker_index = 0
        game_state.discussion_turns_remaining = len(game_state.get_alive_players()) * 3

    elif game_state.phase == GamePhase.DAY_DISCUSSION:
        game_state.phase = GamePhase.DAY_VOTING
        game_state.current_voter_index = 0
        game_state.votes = {}

    elif game_state.phase == GamePhase.DAY_VOTING:
        winner = game_state.check_win_condition()
        if winner:
            game_state.winner = winner
            game_state.phase = GamePhase.GAME_OVER
        else:
            game_state.phase = GamePhase.NIGHT
            game_state.day_count += 1
            game_state.current_mafia_speaker_index = 0
            game_state.mafia_chat_rounds_remaining = 2
