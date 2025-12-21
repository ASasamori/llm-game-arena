import pytest
from unittest.mock import AsyncMock, patch
from app.engine import (
    initialize_game, resolve_night_actions, resolve_voting_turn, 
    finalize_voting, advance_phase, generate_day_turn
)
from app.models import Role, GamePhase, PlayerStatus, AIResponse, GameState

@pytest.mark.asyncio
async def test_initialize_game():
    # Mock load_player_config since it reads from file
    with patch('app.engine.load_player_config') as mock_load:
        mock_load.return_value = [{"name": f"P{i}", "model_id": "m", "avatar": "a"} for i in range(10)]
        gs = initialize_game()
        
        assert len(gs.players) == 10
        assert gs.phase == GamePhase.SETUP
        
        # Check role counts
        roles = [p.role for p in gs.players]
        assert roles.count(Role.MAFIA) == 3
        assert roles.count(Role.DOCTOR) == 1
        assert roles.count(Role.SHERIFF) == 1
        assert roles.count(Role.VILLAGER) == 5

@pytest.mark.asyncio
async def test_night_phase_kill():
    gs = initialize_game()
    # Manually assign roles for deterministic testing
    for p in gs.players: p.role = Role.VILLAGER
    gs.players[0].name = "MafiaPlayer"
    gs.players[0].role = Role.MAFIA
    gs.players[1].name = "Target"
    gs.players[1].role = Role.VILLAGER
    
    with patch('app.agents.get_mafia_kill_target', new_callable=AsyncMock) as mock_kill, \
         patch('app.agents.get_doctor_save_target', new_callable=AsyncMock) as mock_save, \
         patch('app.agents.get_sheriff_investigate_target', new_callable=AsyncMock) as mock_investigate:
        
        mock_kill.return_value = "Target"
        mock_save.return_value = "SomeoneElse"
        mock_investigate.return_value = None
        
        result = await resolve_night_actions(gs)
        
        assert result["death"] == "Target"
        assert gs.players[1].status == PlayerStatus.DEAD
        assert "Target" in gs.graveyard

@pytest.mark.asyncio
async def test_night_phase_save():
    gs = initialize_game()
    # Manually assign roles for deterministic testing
    for p in gs.players: p.role = Role.VILLAGER
    gs.players[0].name = "MafiaPlayer"
    gs.players[0].role = Role.MAFIA
    gs.players[1].name = "DoctorPlayer"
    gs.players[1].role = Role.DOCTOR
    gs.players[2].name = "Target"
    gs.players[2].role = Role.VILLAGER
    
    with patch('app.agents.get_mafia_kill_target', new_callable=AsyncMock) as mock_kill, \
         patch('app.agents.get_doctor_save_target', new_callable=AsyncMock) as mock_save, \
         patch('app.agents.get_sheriff_investigate_target', new_callable=AsyncMock) as mock_investigate:
        
        mock_kill.return_value = "Target"
        mock_save.return_value = "Target"
        mock_investigate.return_value = None
        
        result = await resolve_night_actions(gs)
        
        assert result["saved"] is True
        assert result["death"] is None
        assert gs.players[2].status == PlayerStatus.ALIVE

@pytest.mark.asyncio
async def test_voting_elimination():
    gs = initialize_game()
    # Setup players to vote for 'Victim'
    for p in gs.players:
        p.status = PlayerStatus.ALIVE
    gs.players[0].name = "Victim"
    gs.current_voter_index = 0
    gs.votes = {}
    
    with patch('app.agents.get_player_vote', new_callable=AsyncMock) as mock_vote:
        mock_vote.return_value = "Victim"
        
        # Resolve all votes
        for _ in range(len(gs.get_alive_players())):
            await resolve_voting_turn(gs)
            
        result = finalize_voting(gs)
        
        assert result.eliminated == "Victim"
        assert gs.players[0].status == PlayerStatus.DEAD

@pytest.mark.asyncio
async def test_night_phase_sheriff():
    gs = initialize_game()
    for p in gs.players: p.role = Role.VILLAGER
    gs.players[0].name = "Sheriff"
    gs.players[0].role = Role.SHERIFF
    gs.players[1].name = "MafiaMember"
    gs.players[1].role = Role.MAFIA
    gs.players[1].status = PlayerStatus.ALIVE
    gs.players[2].name = "Townie"
    gs.players[2].role = Role.VILLAGER
    
    with patch('app.agents.get_mafia_kill_target', new_callable=AsyncMock) as mock_kill, \
         patch('app.agents.get_doctor_save_target', new_callable=AsyncMock) as mock_save, \
         patch('app.agents.get_sheriff_investigate_target', new_callable=AsyncMock) as mock_investigate:
        
        mock_kill.return_value = "Townie"
        mock_save.return_value = "One"
        mock_investigate.side_effect = ["MafiaMember", "Townie"]
        
        # Test Mafia investigation
        result1 = await resolve_night_actions(gs)
        assert result1["sheriff_result"] == "MAFIA"
        assert "investigated MafiaMember - they are MAFIA" in gs.players[0].private_memory[-1]
        
        # Reset for Town investigation
        gs.players[2].status = PlayerStatus.ALIVE # Revive for next test
        result2 = await resolve_night_actions(gs)
        assert result2["sheriff_result"] == "TOWN"
        assert "investigated Townie - they are TOWN" in gs.players[0].private_memory[-1]

@pytest.mark.asyncio
async def test_generate_day_turn_logic():
    gs = initialize_game()
    gs.phase = GamePhase.DAY_DISCUSSION
    gs.discussion_turns_remaining = 2
    
    with patch('app.agents.get_player_speech', new_callable=AsyncMock) as mock_speech:
        mock_speech.return_value = AIResponse(inner_monologue="I suspect P1", public_speech="Hello all")
        
        # Turn 1
        turn1 = await generate_day_turn(gs)
        assert turn1["speaker"] == gs.get_alive_players()[0].name
        assert gs.discussion_turns_remaining == 1
        assert len(gs.transcript) > 0
        
        # Turn 2
        turn2 = await generate_day_turn(gs)
        assert turn2["speaker"] == gs.get_alive_players()[1].name
        assert gs.discussion_turns_remaining == 0
        
        # Turn 3 (No turns left)
        turn3 = await generate_day_turn(gs)
        assert turn3 is None

@pytest.mark.asyncio
async def test_voting_tie_and_skip_logic():
    gs = initialize_game()
    # 4 players: A, B, C, D
    # Case 1: Tie (2 for A, 2 for B) -> Actually 10 players, A=2, B=8
    with patch('app.agents.get_player_vote', new_callable=AsyncMock) as mock_vote:
        mock_vote.side_effect = ["A", "A", "B", "B", "B", "B", "B", "B", "B", "B"]
        gs.current_voter_index = 0
        gs.votes = {}
        for _ in range(10):
            await resolve_voting_turn(gs)
        result = finalize_voting(gs)
        assert result.eliminated is not None 

    # Case 2: No majority (10 players, top voted has 4)
    with patch('app.agents.get_player_vote', new_callable=AsyncMock) as mock_vote:
        votes = ["A"]*4 + ["B"]*3 + ["C"]*3
        mock_vote.side_effect = votes
        gs.current_voter_index = 0
        gs.votes = {}
        for _ in range(10):
            await resolve_voting_turn(gs)
        result = finalize_voting(gs)
        assert result.eliminated is None
        assert result.was_tie is False 

    # Case 3: Skip majority
    with patch('app.agents.get_player_vote', new_callable=AsyncMock) as mock_vote:
        mock_vote.return_value = "skip"
        gs.current_voter_index = 0
        gs.votes = {}
        for _ in range(10):
            await resolve_voting_turn(gs)
        result = finalize_voting(gs)
        assert result.eliminated is None

def test_full_phase_transition_cycle():
    gs = initialize_game()
    assert gs.phase == GamePhase.SETUP
    
    # Setup -> Night
    advance_phase(gs)
    assert gs.phase == GamePhase.NIGHT
    
    # Night -> Day Discussion
    advance_phase(gs)
    assert gs.phase == GamePhase.DAY_DISCUSSION
    assert gs.discussion_turns_remaining == 12
    
    # Discussion -> Voting
    advance_phase(gs)
    assert gs.phase == GamePhase.DAY_VOTING
    
    # Voting -> Night (if no winner)
    with patch.object(GameState, 'check_win_condition', return_value=None):
        advance_phase(gs)
        assert gs.phase == GamePhase.NIGHT
        assert gs.day_count == 2

    # Voting -> Game Over (if winner)
    gs.phase = GamePhase.DAY_VOTING
    with patch.object(GameState, 'check_win_condition', return_value="mafia"):
        advance_phase(gs)
        assert gs.phase == GamePhase.GAME_OVER
        assert gs.winner == "mafia"
