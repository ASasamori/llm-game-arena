game_player_map = {0: "AI", 1: "Random"}

connect_four_state = """.......
.......
.......
.......
.......
......x

"""
connect_four_action = "o2; o6; x0. The game begins with player x."
connect_four_player_map = {0: "x", 1: "o"}
connect_four_notation = "zero-based column number"
connect_four_state_notation = "ASCII notation"


GAME_SPECIFIC_NOTATIONS = {
    "connect_four": {
        "state": connect_four_state,
        "action": connect_four_action,
        "player_map": connect_four_player_map,
        "move_notation": connect_four_notation,
        "state_notation": connect_four_state_notation,
    },
}
