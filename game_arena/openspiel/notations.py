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

chess_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
chess_action = "e4; Nf3; O-O"
chess_player_map = {0: "white", 1: "black"}
chess_notation = "Standard Algebraic Notation (SAN)"
chess_state_notation = "FEN string"


GAME_SPECIFIC_NOTATIONS = {
    "connect_four": {
        "state": connect_four_state,
        "action": connect_four_action,
        "player_map": connect_four_player_map,
        "move_notation": connect_four_notation,
        "state_notation": connect_four_state_notation,
    },
    "chess": {
        "state": chess_state,
        "action": chess_action,
        "player_map": chess_player_map,
        "move_notation": chess_notation,
        "state_notation": chess_state_notation,
    },
}
