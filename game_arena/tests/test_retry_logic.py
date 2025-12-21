import pytest
from unittest.mock import AsyncMock, patch
from app.models import GameState, GamePhase, Player, Role, PlayerStatus, TranscriptEntry
from app import engine
from app.llm_client import RateLimitError

@pytest.mark.asyncio
async def test_generate_day_turn_rate_limit_retry():
    """
    Test that a RateLimitError prevents the turn from advancing,
    and a subsequent successful call advances it exactly once.
    """
    # Setup state
    players = [
        Player(id=0, name="Alice", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👩"),
        Player(id=1, name="Bob", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👨")
    ]
    game_state = GameState(
        phase=GamePhase.DAY_DISCUSSION,
        day_count=1,
        players=players,
        current_speaker_index=0,
        discussion_turns_remaining=5
    )
    
    # Mock agents.get_player_speech
    # First call raises RateLimitError
    # Second call returns valid speech
    mock_speech = AsyncMock()
    mock_speech.side_effect = [
        RateLimitError("Rate limit hit"),
        type('obj', (object,), {"public_speech": "Hello", "inner_monologue": "Thinking"})
    ]
    
    with patch("app.agents.get_player_speech", mock_speech):
        # 1. First attempt: Should raise RateLimitError
        try:
            await engine.generate_day_turn(game_state)
            assert False, "Should have raised RateLimitError"
        except RateLimitError:
            pass
        
        # Verify state did NOT change
        assert game_state.current_speaker_index == 0
        assert game_state.discussion_turns_remaining == 5
        assert len(game_state.transcript) == 0
        
        # 2. Second attempt: Should succeed
        result = await engine.generate_day_turn(game_state)
        
        # Verify result
        assert result["speaker"] == "Alice"
        assert result["public_speech"] == "Hello"
        
        # Verify state DID change exactly once
        assert game_state.current_speaker_index == 1
        assert game_state.discussion_turns_remaining == 4
        assert len(game_state.transcript) == 1
        assert game_state.transcript[0].content == "Hello"

@pytest.mark.asyncio
async def test_no_consecutive_speaker():
    """
    Test that the same speaker is not selected twice in a row.
    """
    players = [
        Player(id=0, name="Alice", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👩"),
        Player(id=1, name="Bob", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👨")
    ]
    # Force transcript to show Alice spoke last
    game_state = GameState(
        phase=GamePhase.DAY_DISCUSSION,
        day_count=1,
        players=players,
        current_speaker_index=0, # Would normally pick Alice (index 0)
        discussion_turns_remaining=5,
        transcript=[TranscriptEntry(speaker="Alice", content="Prev")]
    )
    
    mock_speech = AsyncMock()
    mock_speech.return_value = type('obj', (object,), {"public_speech": "Hi", "inner_monologue": "..."})
    
    with patch("app.agents.get_player_speech", mock_speech):
        await engine.generate_day_turn(game_state)
        
        assert game_state.current_speaker_index == 2
        assert game_state.transcript[-1].speaker == "Bob"

@pytest.mark.asyncio
async def test_consecutive_rate_limit_retries():
    """
    Test that the game state remains stable even after multiple consecutive rate limits.
    Simulates the "Retrying..." loop scenario.
    """
    players = [
        Player(id=0, name="Alice", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👩"),
        Player(id=1, name="Bob", role=Role.VILLAGER, model_id="gpt-3.5-turbo", avatar="👨")
    ]
    game_state = GameState(
        phase=GamePhase.DAY_DISCUSSION,
        day_count=1,
        players=players,
        current_speaker_index=0,
        discussion_turns_remaining=5
    )
    
    # Mock: 3 Failures, then 1 Success
    mock_speech = AsyncMock()
    mock_speech.side_effect = [
        RateLimitError("Fail 1", retry_after=1),
        RateLimitError("Fail 2", retry_after=1),
        RateLimitError("Fail 3", retry_after=1),
        type('obj', (object,), {"public_speech": "Finally!", "inner_monologue": "Success"})
    ]
    
    with patch("app.agents.get_player_speech", mock_speech):
        # Attempt 1
        try:
            await engine.generate_day_turn(game_state)
        except RateLimitError:
            pass
        assert game_state.current_speaker_index == 0 # No change
        
        # Attempt 2
        try:
            await engine.generate_day_turn(game_state)
        except RateLimitError:
            pass
        assert game_state.current_speaker_index == 0 # Still no change
        
        # Attempt 3
        try:
            await engine.generate_day_turn(game_state)
        except RateLimitError:
            pass
        assert game_state.current_speaker_index == 0 # Still no change
        
        # Attempt 4 (Success)
        result = await engine.generate_day_turn(game_state)
        
        # Verify success
        assert result["public_speech"] == "Finally!"
        assert game_state.current_speaker_index == 1
        assert len(game_state.transcript) == 1

@pytest.mark.asyncio
async def test_parallel_voting_rate_limit():
    """
    Test parallel execution (Voting Phase) where multiple requests happen at once.
    Simulates: 
    1. 5 Players vote in parallel.
    2. All hit Rate Limit (or one hits it, failing the batch).
    3. Retry logic prevents state corruption.
    4. Successful retry collects all votes.
    """
    # 5 Players
    players = [
        Player(id=i, name=f"Player{i}", role=Role.VILLAGER, model_id="gpt-3.5", avatar="😐") 
        for i in range(5)
    ]
    
    game_state = GameState(
        phase=GamePhase.DAY_VOTING,
        day_count=1,
        players=players
    )
    
    # Mocking get_player_vote to fail first, then succeed
    # We want the FIRST BATCH (5 calls) to ALL fail
    
    side_effects = [RateLimitError("Limit")] * 5 + ["Player0"] * 5
    
    # We patch agents.get_player_vote directly
    mock_vote = AsyncMock()
    mock_vote.side_effect = side_effects
    
    with patch("app.agents.get_player_vote", mock_vote):
        # Attempt 1: Should Fail
        try:
            await engine.resolve_voting(game_state)
            assert False, "Should have raised RateLimitError from parallel batch"
        except RateLimitError:
            pass
            
        # Verify NO state change (nobody eliminated yet)
        assert game_state.phase == GamePhase.DAY_VOTING
        assert len(game_state.graveyard) == 0
        
        # Attempt 2: Should Succeed (consumes next 5 items in side_effects)
        result = await engine.resolve_voting(game_state)
        
        # Verify Result
        assert result.eliminated == "Player0"
        assert len(result.votes) == 5
        assert result.votes["Player1"] == "Player0"
