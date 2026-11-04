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

The response contains `run_id`, job count, status URL, and report links. Each
game's number of players determines the seat count. Accepted modes are:

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

For base Catan, use `catan(players=3)` or `catan(players=4)` and supply one model
per seat. This short mock smoke run uses declared game safety caps:

```bash
curl -sS -X POST http://localhost:8000/v1/runs -H 'Authorization: Bearer local-dev-key' -H 'Content-Type: application/json' -d '{"run_type":"PVP","env":"catan(players=3,max_turns=20)","num_episodes":1,"num_steps":5000,"model_names":["MOCK:a","MOCK:b","MOCK:c"],"seed":123,"failure_policy":"raise"}'
```

Trading is part of the environment's legal action protocol: choose a partner,
draft resource quantities, send, then accept/reject/counter. Bank/port trades
are separate legal actions. The recorded viewer shows the island and offered
bundles; it is a replay viewer, not an interactive human game client. See the
[environment guide](game_arena/environments/README.md) for rules and parameters.

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

## Inspect a run in the browser or terminal

Open `http://localhost:8000/runs/RUN_ID` for a specific run, or
`http://localhost:8000/reports` to enter its ID. The API serves this website
directly. The public HTML shell contains no run results or API key. Enter the key
in the page to load authenticated report data; the page keeps it in memory and
does not put it in the URL or browser storage. Reloading the page requires the
key again.

Run creation and status responses include these absolute URLs:

- `report_url`: the browser page.
- `text_report_url`: the authenticated ASCII report endpoint.
- `report_data_url`: the authenticated JSON report endpoint.

The browser lets you choose an episode, jump to a numbered step, select a phase,
and step through transitions using buttons or arrow keys. Reports
show outcomes and return vectors, moves, recorded player observations, summaries,
and raw model responses. New native traces include before/after public board
snapshots, with private player views shown separately. Older Avalon traces can
display public fields extracted from stored observations, labelled as such;
missing after-state evidence is left unavailable. Generic OpenSpiel games retain
their existing evaluator boards and observations.
A summary is text the model supplied during play; it does not expose hidden
internal reasoning. Mock responses are placeholders and should not be interpreted
as strategic reasoning. State views and private observations are evaluator
evidence, so share report access with trusted evaluators. Failed and legacy
episodes retain their recorded status and show a notice when evidence is unavailable.

These examples are single-line commands. Replace `RUN_ID` and `EPISODE_ID` with
IDs from your run, and use your configured API key:

```bash
curl -sS 'http://localhost:8000/v1/runs/RUN_ID/report?format=text' -H 'Authorization: Bearer local-dev-key'
curl -sS 'http://localhost:8000/v1/runs/RUN_ID/report?episode_id=EPISODE_ID&step_limit=20&step_offset=20' -H 'Authorization: Bearer local-dev-key'
curl -sS 'http://localhost:8000/v1/runs/RUN_ID/report?format=text&episode_id=EPISODE_ID&include_prompt=true&width=120' -H 'Authorization: Bearer local-dev-key'
```

`GET /v1/runs/{run_id}/report` accepts:

- `format`: `json` (default) or `text` (`text/plain`, ASCII).
- `episode_limit`: 1–100, default 20; `episode_offset`: nonnegative, default 0.
- `episode_id`: optional episode selection, which must belong to this run.
- `step_limit`: 1–200, default 20; `step_offset`: nonnegative, default 0.
- `width`: text report width from 60–160, default 100.
- `include_prompt`: include recorded prompts in text output; default `false`.

The JSON payload has `schema_version`, `run`, `episodes`, `selected_episode`,
`transitions`, `transition_index`, and `notices` fields. Both page collections include their items,
total, limit, and offset. JSON transition records include the saved model data;
`include_prompt` controls only the plain-text rendering. Report requests use the
usual authentication and rate limits. Reading a report does not request new model
generations or incur provider API fees. The website needs no separate hosting.

`transition_index` projects small navigation fields for at most 5,000 transitions:
zero-based `index`, `kind`, public `phase`, `quest`, `turn`, and a short `label`.
It excludes prompts, observations, and responses. Its `total` and `truncated`
fields describe the full episode. Numeric jumps still reach any later step using
`step_offset`; transition payload pages remain limited to 200 items. The browser
uses human-readable step numbers starting at 1 and saves `episode_id` and `step`
in its URL. New native transition records have `public_state_before` and
`public_state_after`; these contain spectator facts rather than hidden game state.

## Jev typed decision agents

`JEV:jev-1.14.0` and `JEV:jev-1.13.0` use the official
[TypeSafe API](https://docs.typesafe.ai/introduction/quickstart). `JEV` defaults to
the pinned 1.14.0 version; `jev-latest` and `jev-preview` aliases are also accepted.
Each action is a typed Choice question over the current player's legal actions.
Only their observation, their own action history, and game instructions are sent.
The provider supports up to 255 choices; this includes all Avalon team sizes.
The adapter also applies a conservative 32 KB request bound. A forced single
legal action is applied locally without an API call or classifier confidence.

This is a classifier policy, not a prose-generating model. Reports and exports
retain the original typed JSON, probabilities, confidence, token usage, and
pricing. Any Summary/Final Answer envelope is labelled as an adapter summary;
it is not presented as generated reasoning. Mock mode is explicitly marked and
does not fabricate a provider response.

TypeSafe currently charges [USD 0.042 per million input tokens](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost),
with free output and no free/trial credits. A key must be issued by an
authenticated, funded [TypeSafe console](https://console.typesafe.ai/) account.
[Keys have no individual spending or permission limits](https://help.typesafe.ai/articles/8310295754-who-can-create-and-manage-api-keys).
A random local string is not a valid provider key. The arena API key, such as
`arena-test-key`, authenticates only to this gateway.

Configure the provider credential in the server environment, never in a report
URL, browser storage, or a committed file:

```bash
export TYPESAFE_API_KEY='YOUR_TYPESAFE_ISSUED_KEY'
export JEV_BUDGET_LEDGER='analytics/jev-budget.sqlite'
export JEV_MAX_COST_USD='0.10'
export JEV_MAX_CALLS='100'
```

`JEV_API_KEY` is accepted as an alternative credential variable. Set the same
`JEV_BUDGET_LEDGER` path for all workers to share an aggregate cap. Without that
setting, a standalone Jev model gets its own private temporary ledger; its cap
is per instance, not per run. Existing ledgers keep immutable limits and pricing.
Calls are admitted using a conservative input bound before HTTP, with no retry
or redirect. Failed/ambiguous calls retain their reservation. Confirmed usage
reconciles input cost; output tokens remain free.

For the six-seat Avalon trial, use the existing local database, Redis, and
`runner.py` with `RUNNER_QUEUE_NAME=avalon_trial`. This command shares a **$0.10
cap across the entire trial**, stops after the third quest result, and requires
explicit paid execution:

```bash
uv run python -m orchestrator.live_trial --provider jev --directory analytics/jev-trial --allow-paid
```

It uses six independent player views of `jev-1.14.0`, not six different models.
The recorded build and provider configuration are frozen before queueing. A
durable once-only episode claim prevents a worker restart from repeating paid
execution. A budget/provider stop preserves the existing partial trace and is
reported as truncated, with no fabricated win. Ordinary API runs can also use
`"model_names":["JEV:jev-1.14.0", ...]`, one entry per seat. `MODEL_API_MOCK_MODE=true`
keeps Jev calls offline like the other providers.

## Export a collection

Download all recorded data for a run, without report pagination:

```bash
curl -fsS "$ARENA_URL/v1/runs/$RUN_ID/export" -H "Authorization: Bearer $ARENA_KEY" -o "arena-$RUN_ID.zip"
```

Creation/status responses include `export_url`. The ZIP contains complete gzip
JSONL database rows, normalized `decisions.jsonl.gz`, and `manifest.json` with
row counts, file sizes, and SHA-256 checksums. It includes all episodes and
transitions for the run, not only the currently selected report page. Global
ratings are omitted from single-run exports. Empty or incomplete runs retain
their actual stored status; exports do not manufacture results.

For the entire configured database, explicitly supply its connection URL through
the environment. The utility does not guess a Raspberry Pi or cloud target:

```bash
export DATABASE_URL='postgresql+psycopg2://USER:PASSWORD@HOST:5432/DATABASE'
uv run python -m orchestrator.exports --output analytics/collection.zip
uv run python -m orchestrator.exports --run-id "$RUN_ID" --output analytics/single-run.zip
uv run python -m orchestrator.exports --postgres-dump --output analytics/backup.zip
```

`--postgres-dump` adds `arena-postgres.dump` for `pg_restore` into a PostgreSQL
server you choose. It requires a compatible installed `pg_dump`; on Homebrew,
use `--pg-dump-binary /opt/homebrew/opt/postgresql@17/bin/pg_dump` when needed.
It is available only for whole PostgreSQL databases. SQLite supports JSONL ZIP
exports. PostgreSQL exports share a repeatable-read snapshot, including the
optional database backup, so the files describe the same point in time while
workers continue collecting. Rows are streamed with bounded memory and archives
are published atomically without overwriting existing output. No database rows
are changed. API download files are removed after the response finishes.

The archive contains private player observations, prompts, and raw responses.
No API keys or database connection settings are added by the exporter. It is a
local/private collection artifact; moving it to another server is an explicit
operation. A temporary tunnel's database is not automatically your permanent
Raspberry Pi collection. Run export against the intended database while it is
available; this command does not restart stopped temporary services.

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
