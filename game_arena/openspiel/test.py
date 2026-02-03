import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Play out a win for 'x'.
state.apply_action(4)
state.apply_action(1)
state.apply_action(2)
state.apply_action(5)
print(state)
print(state.returns())
