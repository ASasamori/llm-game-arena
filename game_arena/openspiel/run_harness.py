from game_arena.openspiel import harness
from game_arena.openspiel import model_generation
from game_arena.openspiel import utils


test = utils.HarnessOptions(
    type="PVP",
    num_steps=4,
    models=[model_generation.ClaudeModel(), model_generation.GeminiModel()],
    input=None,
)

res: utils.EpisodeResult = harness.harness(test)

print(f"Result of res is {res}")
