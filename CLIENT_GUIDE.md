# Arena client guide

For a no-service, no-credential workflow, use the offline planner and explicit
mock execution commands in [README.md](README.md). This guide covers the queued
HTTP API, which requires the configured Redis and Postgres services.

## Start locally

```bash
docker compose up --build
```

The API listens at `http://localhost:8000`. Local compose auth defaults to
`local-dev-key`; send `Authorization: Bearer <key>` or `x-api-key: <key>`.
Swagger and the OpenAPI schema are available at `/docs` and `/openapi.json`.
Use model identifier `MOCK` to ensure no provider client is constructed.

## Discover a game

```bash
curl http://localhost:8000/v1/games \
  -H 'Authorization: Bearer local-dev-key'

curl --get http://localhost:8000/v1/games/inspect \
  --data-urlencode 'game=kuhn_poker(players=3)' \
  -H 'Authorization: Bearer local-dev-key'
```

The catalog declares capabilities and requirements from this installation.
Inspection checks an exact game configuration in an isolated process. Supply all
required game parameters. Mean field games need an explicit distribution and
population; sampled stochastic environments need a reproducible underlying seed.
Some games require optional upstream dependencies absent from the installation.

## Submit and poll

```bash
curl -X POST http://localhost:8000/v1/runs \
  -H 'Authorization: Bearer local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"run_type":"PVP","env":"kuhn_poker(players=3)","num_episodes":3,"num_steps":100,"model_names":["MOCK","MOCK","MOCK"],"seed":123,"rotate_seats":true}'
```

The response contains `run_id`, job count, and status URL. Each game’s number of
players determines the seat count. Accepted modes are:

- `PVP`: exactly one model per game seat.
- `PVP_RR`: n-player combinations from the supplied model pool.
- `RAND`: one model in seat zero, random policies in the remaining seats.
- `INPUT`: one model and a fixed `input_actions` list, with `num_episodes=1`.

`env` accepts native game specifications such as `matrix_rps`, `chess`, or
`kuhn_poker(players=3)`. The existing `connect_four_v1` and `chess_v1` aliases work.
`seed` controls harness randomness. `rotate_seats` defaults to true for PVP games;
use a multiple of the player count for equal seat exposure. `num_steps` is a
transition cap. Chance, simultaneous, and mean field updates each count as one
transition; a simultaneous transition contains one decision per acting player.
`failure_policy` accepts `random` (the compatibility default) or `raise` for
strict evaluation. Fallback-assisted episodes are marked degraded, reported
separately in metrics, and excluded from Elo updates.

For mean field games include `"mean_field_distribution":"uniform"` or a support
probability mapping, plus `"mean_field_population":0`. This records an exogenous
population assumption, not an equilibrium solver.

```bash
curl http://localhost:8000/v1/runs/<RUN_ID> \
  -H 'Authorization: Bearer local-dev-key'

curl 'http://localhost:8000/v1/runs/<RUN_ID>/episodes?limit=50&offset=0' \
  -H 'Authorization: Bearer local-dev-key'

curl http://localhost:8000/v1/runs/<RUN_ID>/metrics \
  -H 'Authorization: Bearer local-dev-key'

curl 'http://localhost:8000/v1/episodes/<EPISODE_ID>/trace?limit=100&offset=0' \
  -H 'Authorization: Bearer local-dev-key'
```

The trace endpoint returns episode provenance and paginated transition records.
Records include actual player IDs, legal actions, player observations, prompts,
responses, fallback reasons, rewards, returns, and state hashes. Legacy episodes
created before trace recording may not have a trace. Retain source and dependency
versions with these records for replay. Treat traces as evaluator data: private
observations for different players can appear in the same dataset.

The metrics endpoint reports status counts, terminal returns by player/agent,
transition counts, and fallback counts. A `TRUNCATED` episode is incomplete and is
not counted as a win. Solo, cooperative, and general-sum utilities are preserved;
they are not converted into two-player Elo. Existing `/analytics` is specialized
for Connect Four and `/leaderboard` is for eligible two-player rated games.

## Service configuration

Important environment settings:

- `ARENA_API_KEYS`: comma-separated accepted API keys.
- `DATABASE_URL`: direct SQLAlchemy database URL override.
- `DATABASE_TARGET`: `raspberry_pi` or `gcp`, selecting its configured URL.
- `REDIS_HOST` / `REDIS_PORT`: queue and rate-limit store.
- `RUNNER_NUM_WORKERS`: runtime worker concurrency in `orchestrator.runner`.
- `MODEL_API_MOCK_MODE=true`: mock behavior for legacy provider wrappers.

Request/token rate limits apply to each API key. `/v1/usage/me` reports the current
minute’s usage. The runner remains the entry point for long-lived local workers:

```bash
uv run python -m orchestrator.api_gateway
uv run python -m orchestrator.runner
```

A pending run needs a listening worker and reachable database/queue. Authentication
failures return 401; invalid game/player configurations are rejected before
scheduling. Queue/DB outages return a service error. For deployment instructions,
read [infra/gcp/README.md](infra/gcp/README.md).

When submission is interrupted after its database commit, a `503` includes the
run ID and polling URL. The runner retries pending dispatches at startup and
every 60 seconds. Run status exposes `dispatch_complete` and `dispatch_error`.
Workers must match the submitted build's source, dependencies, and OpenSpiel
version; mismatches are recorded before any model client is created.
