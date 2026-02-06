import pyspiel

game = pyspiel.load_game("connect_four")
state = game.new_initial_state()

# Play out a win for 'x'.
print(state.legal_actions(state.current_player()))
action_string = state.action_to_string(state.current_player(),0)
print(action_string)
state.apply_action(0)
print(state.legal_actions(state.current_player()))
action_string = state.action_to_string(state.current_player(),0)
print(action_string)