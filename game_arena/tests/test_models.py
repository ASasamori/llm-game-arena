import pytest
from app.models import GameState, Player, Role, PlayerStatus, GamePhase

def test_game_state_alive_filtering():
    players = [
        Player(id=0, name="P1", model_id="m1", avatar="a1", role=Role.MAFIA, status=PlayerStatus.ALIVE),
        Player(id=1, name="P2", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.DEAD),
        Player(id=2, name="P3", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.ALIVE),
    ]
    gs = GameState(players=players)
    
    assert len(gs.get_alive_players()) == 2
    assert gs.get_alive_players()[0].name == "P1"
    assert gs.get_alive_players()[1].name == "P3"

def test_game_state_mafia_town_filtering():
    players = [
        Player(id=0, name="M1", model_id="m1", avatar="a1", role=Role.MAFIA),
        Player(id=1, name="V1", model_id="m1", avatar="a1", role=Role.VILLAGER),
        Player(id=2, name="D1", model_id="m1", avatar="a1", role=Role.DOCTOR),
    ]
    gs = GameState(players=players)
    
    assert len(gs.get_alive_mafia()) == 1
    assert len(gs.get_alive_town()) == 2
    assert gs.get_alive_mafia()[0].name == "M1"

def test_win_conditions():
    # Town wins (no mafia)
    gs_town = GameState(players=[
        Player(id=0, name="V1", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.ALIVE)
    ])
    assert gs_town.check_win_condition() == "town"
    
    # Mafia wins (Mafia >= Town)
    gs_mafia = GameState(players=[
        Player(id=0, name="M1", model_id="m1", avatar="a1", role=Role.MAFIA, status=PlayerStatus.ALIVE),
        Player(id=1, name="V1", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.ALIVE)
    ])
    assert gs_mafia.check_win_condition() == "mafia"
    
    # Game continues
    gs_cont = GameState(players=[
        Player(id=0, name="M1", model_id="m1", avatar="a1", role=Role.MAFIA, status=PlayerStatus.ALIVE),
        Player(id=1, name="V1", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.ALIVE),
        Player(id=2, name="V2", model_id="m1", avatar="a1", role=Role.VILLAGER, status=PlayerStatus.ALIVE)
    ])
    assert gs_cont.check_win_condition() is None
