from game_arena.openspiel import harness
from game_arena.openspiel import utils

from collections import deque

test = utils.HarnessOptions(
    type="PVP",
    num_moves=3,
    models=["GPT","GEMINI"],
    input=None
)

res = harness.harness(test)

print(f"Result of res is {res}")