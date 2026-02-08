from game_arena.openspiel import harness
from game_arena.openspiel import model_generation
from game_arena.openspiel import utils

from collections import deque

test = utils.HarnessOptions(
    type="INPUT",
    num_moves=4,
    models = [model_generation.GeminiModel()],
    input=deque([0,1,2,4])
)

res: utils.EpisodeResult = harness.harness(test)

print(f"Result of res is {res}")