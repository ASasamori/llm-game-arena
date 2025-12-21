"""
FastAPI application - main entry point and API routes.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.models import GameState, GamePhase
from app import engine

app = FastAPI(title="Mafia AI Simulation", version="1.0.0")

# Global game state
current_game: GameState | None = None


# --- API Routes ---

@app.post("/api/game/start")
async def start_game():
    """Initialize a new game."""
    global current_game
    current_game = engine.initialize_game()
    engine.advance_phase(current_game)  # Move from SETUP to NIGHT
    return {
        "status": "started",
        "phase": current_game.phase.value,
        "day": current_game.day_count,
        "players": [
            {
                "name": p.name,
                "avatar": p.avatar,
                "role": p.role.value,
                "status": p.status.value
            }
            for p in current_game.players
        ],
        "next_speaker": (ns := engine.get_next_speaker(current_game)) and ns.name or None
    }


@app.get("/api/game/state")
async def get_game_state():
    """Get the current game state."""
    if not current_game:
        raise HTTPException(status_code=404, detail="No game in progress")

    return {
        "phase": current_game.phase.value,
        "day": current_game.day_count,
        "winner": current_game.winner,
        "discussion_turns_remaining": current_game.discussion_turns_remaining,
        "players": [
            {
                "name": p.name,
                "avatar": p.avatar,
                "role": p.role.value,
                "status": p.status.value
            }
            for p in current_game.players
        ],
        "next_speaker": (ns := engine.get_next_speaker(current_game)) and ns.name or None,
        "graveyard": current_game.graveyard,
        "transcript": [
            {
                "speaker": e.speaker,
                "content": e.content,
                "is_system": e.is_system,
                "is_secret": e.is_secret,
                "inner_monologue": e.inner_monologue
            }
            for e in current_game.transcript
        ]
    }


@app.post("/api/game/next")
async def next_turn():
    """
    Advance the game by one step.
    - During NIGHT: Resolve all night actions
    - During DAY_DISCUSSION: Get one player's speech
    - During DAY_VOTING: Process the vote
    """
    global current_game
    if not current_game:
        raise HTTPException(status_code=404, detail="No game in progress")

    if current_game.phase == GamePhase.GAME_OVER:
        return {
            "phase": current_game.phase.value,
            "day": current_game.day_count,
            "winner": current_game.winner,
            "message": f"Game Over! {current_game.winner.upper()} wins!"
        }

    result = {
        "phase": current_game.phase.value,
        "day": current_game.day_count
    }

    try:
        if current_game.phase == GamePhase.SETUP:
            engine.advance_phase(current_game)
            message = "🌙 Night 1 falls. The roles are assigned, and the night begins..."
            current_game.add_system_message(message)
            result = {
                "phase": current_game.phase.value,
                "day": current_game.day_count,
                "turn_type": "phase_transition",
                "message": message
            }

        elif current_game.phase == GamePhase.NIGHT:
            # Check if Mafia discussion is still ongoing
            if current_game.mafia_chat_rounds_remaining > 0:
                chat_turn = await engine.generate_mafia_chat_turn(current_game)
                if chat_turn:
                    return {
                        "phase": current_game.phase.value,
                        "day": current_game.day_count,
                        "turn_type": "mafia_chat",
                        "turn": chat_turn,
                        "rounds_remaining": current_game.mafia_chat_rounds_remaining,
                        "next_speaker": (ns := engine.get_next_speaker(current_game)) and ns.name or None
                    }
            
            # If chat is done, resolve actions
            night_result = await engine.resolve_night_actions(current_game)

            if night_result["saved"]:
                message = f"☀️ Day {current_game.day_count} begins. The night was peaceful - no one died!"
            elif night_result["death"]:
                message = f"☀️ Day {current_game.day_count} begins. {night_result['death']} was found dead during the night! 💀"
            else:
                message = f"☀️ Day {current_game.day_count} begins."

            current_game.add_system_message(message)
            engine.advance_phase(current_game)  # Move to DAY_DISCUSSION

            # Check win condition after night kills
            winner = current_game.check_win_condition()
            if winner:
                current_game.winner = winner
                current_game.phase = GamePhase.GAME_OVER
                return {
                    "phase": current_game.phase.value,
                    "day": current_game.day_count,
                    "winner": winner,
                    "night_result": night_result,
                    "message": f"Game Over! {winner.upper()} wins!"
                }

            result = {
                "phase": current_game.phase.value,
                "day": current_game.day_count,
                "turn_type": "night_resolved",
                "night_result": night_result,
                "message": message
            }

        elif current_game.phase == GamePhase.DAY_DISCUSSION:
            # Get next player's speech
            turn_result = await engine.generate_day_turn(current_game)

            if turn_result:
                result = {
                    "phase": current_game.phase.value,
                    "day": current_game.day_count,
                    "turn_type": "discussion_turn",
                    "turn": turn_result,
                    "turns_remaining": current_game.discussion_turns_remaining
                }
            else:
                # No more discussion turns, move to voting
                current_game.add_system_message("⚖️ Discussion time is over. Time to vote!")
                engine.advance_phase(current_game)  # Move to DAY_VOTING
                result = {
                    "phase": current_game.phase.value,
                    "day": current_game.day_count,
                    "turn_type": "phase_transition",
                    "message": "Moving to voting phase..."
                }

        elif current_game.phase == GamePhase.DAY_VOTING:
            # Get next player's vote
            vote_turn = await engine.resolve_voting_turn(current_game)
            
            if vote_turn:
                # Add to transcript for spectator
                current_game.add_system_message(f"🗳️ [VOTE] {vote_turn['voter']} voted for: {vote_turn['target']}", is_secret=True)
                
                return {
                    "phase": current_game.phase.value,
                    "day": current_game.day_count,
                    "turn_type": "vote_turn",
                    "turn": vote_turn,
                    "next_speaker": (ns := engine.get_next_speaker(current_game)) and ns.name or None
                }
            
            # All votes gathered, finalize
            vote_result = engine.finalize_voting(current_game)

            if vote_result.eliminated:
                # Find the eliminated player to show their role
                eliminated_player = next(
                    (p for p in current_game.players if p.name == vote_result.eliminated),
                    None
                )
                role_reveal = eliminated_player.role.value.upper() if eliminated_player else "UNKNOWN"
                message = f"🗳️ The village has voted! {vote_result.eliminated} has been eliminated. They were {role_reveal}!"
            elif vote_result.was_tie:
                message = "🗳️ The vote was tied! No one is eliminated."
            else:
                message = "🗳️ Not enough votes to eliminate anyone. The day ends."

            current_game.add_system_message(message)

            # Check win condition
            winner = current_game.check_win_condition()
            if winner:
                current_game.winner = winner
                current_game.phase = GamePhase.GAME_OVER
                return {
                    "phase": current_game.phase.value,
                    "day": current_game.day_count,
                    "winner": winner,
                    "vote_result": {
                        "votes": vote_result.votes,
                        "eliminated": vote_result.eliminated,
                        "was_tie": vote_result.was_tie
                    },
                    "message": f"Game Over! {winner.upper()} wins!"
                }

            # Move to next night
            current_game.add_system_message(f"🌙 Night {current_game.day_count + 1} falls...")
            engine.advance_phase(current_game)  # Move to NIGHT

            result = {
                "phase": current_game.phase.value,
                "day": current_game.day_count,
                "turn_type": "voting_resolved",
                "vote_result": {
                    "votes": vote_result.votes,
                    "eliminated": vote_result.eliminated,
                    "was_tie": vote_result.was_tie
                },
                "message": message
            }
        
        # Add metadata and speaker info for UI
        result["day"] = current_game.day_count
        result["phase"] = current_game.phase.value
        
        if current_game.phase == GamePhase.DAY_DISCUSSION or current_game.phase == GamePhase.DAY_VOTING or current_game.phase == GamePhase.NIGHT:
            ns = engine.get_next_speaker(current_game)
            result["next_speaker"] = ns.name if ns else None

        return result

    except Exception as e:
        # Check if it's a rate limit error
        if type(e).__name__ == "RateLimitError":
             return {
                "status": "error",
                "error": "rate_limit",
                "retry_after": getattr(e, "retry_after", 5)
            }
        raise e


# --- Static Files ---
static_path = Path(__file__).parent.parent / "static"


@app.get("/")
async def serve_index():
    """Serve the main dashboard."""
    return FileResponse(static_path / "index.html")


# Mount static files (CSS, JS, assets)
app.mount("/static", StaticFiles(directory=static_path), name="static")
