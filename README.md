# LLM Game Arena

A reproducible OpenSpiel evaluation harness with a local experiment planner and
queued workers. Games are discovered from the installed OpenSpiel registry rather
than maintained as a two-game allowlist. The engine follows each state's node
kind, supports arbitrary player counts, and records replayable transition data.

**No model API permissions are needed for the offline workflow.** Planning is the
default. Local execution requires `--execute-mock` and constructs only mock models.
No provider SDK client is created by that path.

## Start without credentials or services

Use Python 3.12+ and the checked-in dependency lock:

```bash
uv sync --frozen

# List games compiled/registered in this installation.
uv run python -m game_arena.openspiel catalog

# Inspect an exact configuration before scheduling it.
uv run python -m game_arena.openspiel catalog --inspect 'kuhn_poker(players=3)'

# Dry-run: inspect capabilities, calculate episode count, preview stable jobs.
uv run python -m game_arena.openspiel plan \
  --games tic_tac_toe kuhn_poker matrix_rps matching_pennies_3p catch \
  --episodes 4 --seed 123 --output analytics/offline-demo

# Explicit mock-only build test. Redis, Postgres, and API keys are unnecessary.
uv run python -m game_arena.openspiel plan \
  --games tic_tac_toe kuhn_poker matrix_rps matching_pennies_3p catch \
  --episodes 4 --seed 123 --output analytics/offline-demo --execute-mock
```

A single model label (the default `mock`) enables selfplay for any player count.
For a tournament, provide distinct labels with `--models alpha beta gamma`; the
planner generates combinations of the game’s player count. Episode count is per
matchup. Seat assignments rotate across episodes; use a multiple of the player
count for equal seat exposure. `--no-seat-rotation` preserves fixed assignments.
Offline labels always refer to local mock models, including labels such as GPT.

`--max-steps` caps state transitions, including chance and mean field transitions;
its default is 1000. An episode that reaches the cap is `TRUNCATED` and does not
contribute a fabricated win. Terminal return vectors are the primary outcome for
solo, multiplayer, cooperative, and general-sum games.

## Game support and capability boundaries

The catalog reports the registry’s player bounds, observation APIs, dynamics,
chance mode, parameter requirements, and reproducibility requirements. A registry
entry is a declaration, not proof that every parameterization works. Exact
configuration inspection runs in a subprocess because invalid native game
constructors can abort a process. Python games are registered when their optional
upstream dependencies are available; missing dependencies are reported.

The generic execution path handles:

- Sequential decisions, including games where the same player acts repeatedly.
- Simultaneous decisions collected before applying a joint action.
- Explicit chance outcomes sampled from a separate, seeded RNG stream.
- Any supported player count, including one-player games.
- Imperfect information through player-scoped information/observation APIs.
- Sampled stochastic games with a reproducibly seeded underlying environment.
- Mean field updates when an explicit population distribution is provided.

Games requiring a wrapped `game=...`, a file, or other required parameters must be
specified with those parameters. External environments that cannot be initialized
or reproduced locally are rejected with a reason. Adding a game to OpenSpiel does
not require adding a notation dictionary entry to this repository.

For a mean field experiment, explicitly select a distribution:

```bash
uv run python -m game_arena.openspiel plan \
  --games mfg_crowd_modelling --mean-field-distribution uniform \
  --episodes 2 --seed 123 --output analytics/mean-field-demo --execute-mock
```

The uniform distribution is an exogenous experimental assumption, **not a solved
mean field equilibrium**. The chosen distribution and population are recorded.
Programmatic configuration also accepts a support-state-to-probability mapping.

Player prompts contain only that player’s scoped view and legal action IDs. The
full evaluator trace can include private observations for several players and
should be treated as evaluation data. Generic text/tensor views provide legal
play; game-specific strategy explanations and image observations are not supplied
automatically.

## Reproduction and replay

Each plan records the exact configuration, inspected game metadata, Python and
OpenSpiel versions, prompt/parser schema versions, source revision, and a hash of
the working source. A different build or configuration needs a new output
directory. Actual episodes also record their effective OpenSpiel specification,
seed, assigned agents, raw model output, selected legal actions, fallbacks,
chance events, reward/return vectors, and state hashes.

A seed controls the harness’s chance, random opponent, and fallback streams. Real
remote model responses may vary even when a provider accepts a seed. Exact
trajectory replay uses recorded actions and never calls a provider.

```bash
# Use an episode_id from the plan preview or an episode summary file.
uv run python -m game_arena.openspiel replay \
  --dataset analytics/offline-demo --episode <EPISODE_ID>
```

Replay checks legal transitions, recorded state hashes, and resulting utilities.
A mismatch identifies incompatible game behavior or modified evidence. Preserving
the dependency lock, source build, and any external files referenced by a game
specification at their recorded absolute paths is necessary for long-term
reproduction. File-backed definitions record SHA-256 hashes and byte sizes;
their contents are not embedded. Replay requires the retained files and checks
their hashes before loading the game. The source hash covers repository Python
code; it does not archive external game assets.

## Large experiments and datasets

Plans generate jobs lazily. Mock execution uses at most `2 * workers` pending jobs
by default and a process pool for `--workers > 1`; each worker holds at most one
episode trace in memory. Increase `--max-pending` only when needed. Memory does not
grow with the total number of episodes, while episode size still depends on its
step cap and observation/action-space size.

```bash
uv run python -m game_arena.openspiel plan \
  --games tic_tac_toe kuhn_poker matrix_rps --episodes 10000 --seed 123 \
  --output analytics/batch --execute-mock --workers 4
```

Independent machines can share a dataset directory through disjoint shards:

```bash
# Run indexes 0, 1, 2, and 3 independently with the same plan/build.
uv run python -m game_arena.openspiel plan \
  --games tic_tac_toe kuhn_poker matrix_rps --episodes 10000 --seed 123 \
  --output analytics/sharded --execute-mock --workers 2 \
  --shard-index 0 --shard-count 4
```

Sharding and worker count do not change episode IDs, seeds, or seat assignments.
Run each shard once at a time. Reissuing the same command resumes committed
episodes after checking their job identity and checksums; corrupt records are
reported rather than silently skipped. An interrupted episode can be rerun.
Execution exceptions are recorded as `FAILED` evidence with no invented return
vector and are retried on resume. The CLI exits with a nonzero status when any
episode fails.

Dataset layout:

```text
manifest.json
 episodes/<bucket>/<episode-id>.json
 episodes/<bucket>/<episode-id>.steps.jsonl
```

Transition streams are flushed and atomically replaced before the corresponding
summary is committed. Temporary/orphan transition files are not completion
markers. One episode is the persistence/retry unit.

Export normalized, streamed JSONL files for downstream analysis or ingestion:

```bash
uv run python -m game_arena.openspiel export \
  --dataset analytics/offline-demo --output analytics/offline-export
```

The exporter verifies the source and writes `episodes.jsonl.gz`,
`steps.jsonl.gz`, and `decisions.jsonl.gz`, plus an export manifest. Use
`--uncompressed` for plain JSONL. Join decisions to steps with `episode_id` and
`step_index`; `decision_index` distinguishes simultaneous players. Row order is
not a semantic identifier. Export and resume verification stream files rather
than accumulating the run in memory.

## API, queue, and runner

The service path remains `FastAPI → Redis/RQ → orchestrator.runner → Postgres`.
The runner creates runtime workers listening for independent episode jobs.
Per-episode seeds remove the need to chain episodes for deterministic randomness.
Queued submission validates game capabilities and player counts before creating
run records, batches queue writes, and uses stable job/episode identities for
retries. Persistence retains actual player IDs instead of inferring players by
turn parity. Versioned run configuration, episode summary, and transition tables
are added alongside the existing tables; existing rows remain readable.

A durable dispatch marker preserves partially submitted runs through Redis
outages. The runner retries pending dispatches at startup and every 60 seconds
with the same job IDs. A submission `503` includes the run ID and polling URL;
status responses report dispatch completion separately from episode completion.

```bash
docker compose up --build
```

The compose configuration defaults to mock behavior. For explicit mock jobs, use
`MOCK` in API `model_names`; this is independent of provider credentials:

```bash
curl -X POST http://localhost:8000/v1/runs \
  -H 'Authorization: Bearer local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"run_type":"PVP","env":"kuhn_poker(players=3)","num_episodes":3,"num_steps":100,"model_names":["MOCK","MOCK","MOCK"],"seed":123,"rotate_seats":true}'
```

`env` accepts native OpenSpiel specifications. `connect_four_v1` and `chess_v1`
remain compatibility aliases. `PVP` needs one model per seat, `PVP_RR` generates
n-player combinations, `RAND` places one model against random policies in other
seats, and `INPUT` supplies a fixed opponent action sequence for one episode.
Server-local filename/path parameters, including nested game specifications, are
disabled for API and queue submissions. Use the trusted offline CLI for file-backed
definitions; their recorded hashes are checked against the plan before execution.

Read [CLIENT_GUIDE.md](CLIENT_GUIDE.md) for endpoint details. Generic metrics expose
clean terminal returns per player/agent, with fallback-assisted returns reported
separately. Fallback-assisted episodes are excluded from Elo. Existing Connect Four
plots and two-player Elo are specialized views; they do not rank cooperative or
multiplayer games with an arbitrary winner rule.

Database and deployment configuration remains in
[infra/gcp/README.md](infra/gcp/README.md). `DATABASE_URL` overrides the configured
target; `DATABASE_TARGET=raspberry_pi` or `gcp` selects the corresponding URL. The
Pi target can be reached at `100.116.74.50` over the configured Tailscale network.
Local offline planning, execution, export, and replay never connect to that DB.

## Validation

```bash
uv run ruff check --fix
uv run ruff format
uv run pyright
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python -m unittest discover -s game_arena/openspiel -p 'test_*.py'

# Bounded, isolated checks for all default-loadable installed games.
uv run python -m game_arena.openspiel.conformance --max-steps 96 --timeout 20

# With disposable PostgreSQL and Redis explicitly configured:
uv run python scripts/generic_e2e_ci.py
```

Use the mock-only command above as the feature build test. The queued integration
smoke additionally needs Redis/Postgres and is exercised by the GitHub workflow.
Before pushing, run the repository workflows through `act` with Docker as required
by the project instructions. No live provider benchmark is required for a build
test.

The old `game_arena/openspiel/prompts.py` template and `data/collect.py` placeholder
are cleanup candidates. The generic engine uses `views.py`; notation helpers
remain for compatibility with existing board-game responses.

## Upstream contracts

The engine follows [OpenSpiel’s state API](https://openspiel.readthedocs.io/en/stable/api_reference.html)
and [upstream implementation](https://github.com/google-deepmind/open_spiel).
[Google DeepMind Game Arena](https://github.com/google-deepmind/game_arena) is a
reference for model competition and recording decisions. This repository provides
its own orchestration, reproduction, and dataset layer rather than copying a
fixed set of Game Arena game adapters.
