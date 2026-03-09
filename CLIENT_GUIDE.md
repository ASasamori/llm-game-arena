# Client Guide

This guide is for clients who want to submit runs and read results without digging into internals.

## 1) What this API does

- Submit game-eval runs (`connect_four_v1`, `chess_v1`)
- Queue work for background workers
- Persist run/episode data
- Return run status, episodes, analytics, and leaderboard JSON

Core endpoint:
- `POST /v1/runs` (returns quickly with `run_id`)

## 2) Auth

Use one of:
- `Authorization: Bearer <API_KEY>`
- `x-api-key: <API_KEY>`

## 3) Fast local start (Docker)

```bash
docker compose up --build
```

Use:
- API: `http://localhost:8000`
- API key (default local): `local-dev-key`

Create a run:

```bash
curl -X POST http://localhost:8000/v1/runs \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "run_type":"PVP",
    "env":"connect_four_v1",
    "num_episodes":1,
    "num_steps":42,
    "model_names":["GEMINI","GPT"]
  }'
```

## 4) GCP managed stack (on/off)

Single-command lifecycle:

```bash
bash scripts/gcp_stack_toggle.sh up
bash scripts/gcp_stack_toggle.sh down
```

After `up`, get API URL and key:

```bash
export API_URL="$(gcloud run services describe llm-arena-api --region us-central1 --format='value(status.url)')"
export API_KEY="$(gcloud secrets versions access latest --secret arena-api-keys | cut -d',' -f1)"
```

## 5) Client request flow

1. Create run
2. Poll run status
3. Fetch episodes/analytics

Create run:

```bash
curl -sS -X POST "$API_URL/v1/runs" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"run_type":"PVP","env":"chess_v1","num_episodes":1,"num_steps":120,"model_names":["GPT","CLAUDE"]}'
```

Poll:

```bash
curl -sS "$API_URL/v1/runs/<RUN_ID>" -H "Authorization: Bearer $API_KEY"
```

Episodes:

```bash
curl -sS "$API_URL/v1/runs/<RUN_ID>/episodes?limit=50&offset=0" \
  -H "Authorization: Bearer $API_KEY"
```

Analytics:

```bash
curl -sS "$API_URL/v1/runs/<RUN_ID>/analytics?action_space=7" \
  -H "Authorization: Bearer $API_KEY"
```

## 6) Health/useful checks

```bash
curl -sS "$API_URL/v1/usage/me" -H "Authorization: Bearer $API_KEY"
```

```bash
curl -sS "$API_URL/v1/leaderboard?game=connect_four&limit=25" \
  -H "Authorization: Bearer $API_KEY"
```

## 7) Common issues

- `401 Invalid API key`: wrong key/header.
- `503 Failed to submit run`: queue/DB dependency issue; check API logs.
- Run stuck `PENDING`: worker not triggered/running.

Manual worker kick (GCP):

```bash
gcloud run jobs execute llm-arena-worker --region us-central1 --wait
```
