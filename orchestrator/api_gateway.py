"""
REST API gateway for arena run orchestration.

Capabilities:
- API key auth
- Redis-backed request/token rate limiting
- Run creation with retry/backoff
- Run status polling
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import time
import math
from itertools import zip_longest
from collections.abc import Awaitable
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypeVar, cast
from uuid import UUID, uuid4

import redis
import tenacity
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import func

from orchestrator.models import Episode, EpisodeStep, Matchup, ModelRating, Run, get_session
from orchestrator.observability import (
    build_dashboard_view,
    configure_json_logging,
    get_metrics_snapshot,
    increment_counter,
    observe_timing_ms,
    time_now_ms,
)
from orchestrator.failures import classify_exception
from orchestrator.settings import get_database_target

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None


LOGGER = logging.getLogger("orchestrator.api_gateway")
configure_json_logging(
    service="api_gateway", level=os.environ.get("API_LOG_LEVEL", "INFO")
)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

REQUESTS_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
TOKENS_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_TOKENS_PER_MINUTE", "30000"))
RATE_LIMIT_TTL_SECONDS = int(os.environ.get("API_RATE_LIMIT_WINDOW_SECONDS", "90"))

API_GATEWAY_NAME = os.environ.get("API_GATEWAY_NAME", "llm-game-arena-gateway")
API_KEYS = {
    item.strip()
    for item in os.environ.get("ARENA_API_KEYS", "local-dev-key").split(",")
    if item.strip()
}

redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# TODO: Read over this and explain what this means
_RATE_LIMIT_SCRIPT = """
local req_key = KEYS[1]
local tok_key = KEYS[2]
local req_inc = tonumber(ARGV[1])
local tok_inc = tonumber(ARGV[2])
local req_lim = tonumber(ARGV[3])
local tok_lim = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local req_curr = tonumber(redis.call("GET", req_key) or "0")
local tok_curr = tonumber(redis.call("GET", tok_key) or "0")

if req_curr + req_inc > req_lim then
  return {0, req_curr, tok_curr}
end

if tok_curr + tok_inc > tok_lim then
  return {0, req_curr, tok_curr}
end

local req_new = redis.call("INCRBY", req_key, req_inc)
local tok_new = redis.call("INCRBY", tok_key, tok_inc)

if req_new == req_inc then
  redis.call("EXPIRE", req_key, ttl)
end
if tok_new == tok_inc then
  redis.call("EXPIRE", tok_key, ttl)
end

return {1, req_new, tok_new}
"""

T = TypeVar("T")


def _resolve_sync_result(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        raise RuntimeError("Received awaitable from sync Redis client")
    return cast(T, value)


class RunCreateRequest(BaseModel):
    run_type: Literal["INPUT", "RAND", "PVP", "PVP_RR"] = Field(
        default="PVP",
        description="Run execution mode.",
    )
    num_episodes: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Episode count. INPUT mode must be 1.",
    )
    num_steps: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Optional per-episode max step cap. Null means play to terminal.",
    )
    model_names: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=16,
        description="Model identifiers (e.g., GEMINI, GPT, CLAUDE).",
    )
    input_actions: list[int] | None = Field(
        default=None,
        description="Only for INPUT mode: fixed opponent action sequence.",
    )
    env: str = Field(
        default="connect_four_v1",
        description="Environment version: connect_four_v1 or chess_v1.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "run_type": "PVP",
                    "num_episodes": 5,
                    "num_steps": 42,
                    "model_names": ["GEMINI", "GPT"],
                    "env": "connect_four_v1",
                },
                {
                    "run_type": "RAND",
                    "num_episodes": 3,
                    "num_steps": 120,
                    "model_names": ["GPT"],
                    "env": "chess_v1",
                }
            ]
        }
    }

    @model_validator(mode="after")
    def validate_mode_constraints(self) -> "RunCreateRequest":
        if self.run_type == "PVP" and len(self.model_names) != 2:
            raise ValueError("PVP requires exactly 2 model_names")
        if self.run_type == "PVP_RR" and len(self.model_names) < 2:
            raise ValueError("PVP_RR requires at least 2 model_names")
        if self.run_type == "INPUT":
            if not self.input_actions:
                raise ValueError("INPUT requires input_actions")
            if self.num_episodes != 1:
                raise ValueError("INPUT supports only num_episodes=1")
        if self.run_type in {"RAND", "INPUT"} and len(self.model_names) < 1:
            raise ValueError(f"{self.run_type} requires at least one model name")
        if self.env not in {"connect_four_v1", "chess_v1"}:
            raise ValueError("env must be one of: connect_four_v1, chess_v1")
        return self


class RunCreateResponse(BaseModel):
    run_id: str = Field(description="Created run UUID.")
    status: str = Field(description="Initial run status.")
    job_count: int = Field(description="Number of jobs enqueued.")
    status_url: str = Field(description="Relative URL for polling run status.")
    submitted_at: datetime = Field(description="Submission timestamp (UTC).")


class MatchupStatus(BaseModel):
    matchup_id: str = Field(description="Matchup UUID.")
    models: list[str] = Field(description="Competing model names.")
    num_episodes: int = Field(description="Episodes scheduled for this matchup.")
    status: str = Field(description="Matchup status.")


class RunStatusResponse(BaseModel):
    run_id: str = Field(description="Run UUID.")
    run_type: str = Field(description="Run type: INPUT, RAND, PVP, or PVP_RR.")
    status: str = Field(description="Run status.")
    num_episodes: int = Field(description="Configured run episode count.")
    episodes_saved: int = Field(description="Episodes persisted so far.")
    created_at: datetime = Field(description="Run creation timestamp (UTC).")
    matchups: list[MatchupStatus] = Field(
        description="Matchup details for PVP/PVP_RR runs.",
    )


class UsageResponse(BaseModel):
    request_count: int = Field(
        description="Requests consumed in current minute bucket."
    )
    token_count: int = Field(description="Token budget consumed in current bucket.")
    requests_per_minute_limit: int = Field(description="Configured request/min limit.")
    tokens_per_minute_limit: int = Field(description="Configured token/min limit.")
    bucket_epoch_minute: int = Field(description="Current UTC epoch minute bucket.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    gateway: str
    database_target: str
    ts: datetime


class ErrorResponse(BaseModel):
    error: str | dict[str, Any] | list[Any]
    status_code: int
    code: str | None = None


class ObservabilitySnapshotResponse(BaseModel):
    exported_at: datetime
    counter_count: int
    timing_series_count: int
    counters: dict[str, int]
    timings: dict[str, dict[str, float | int]]


class ObservabilityDashboardResponse(BaseModel):
    generated_at: datetime
    summary: dict[str, int]
    top_counters: list[dict[str, Any]]
    hottest_timings: list[dict[str, Any]]


class RunEpisodeItem(BaseModel):
    episode_id: str
    matchup_id: str | None
    worker_name: str | None
    status: str
    winner: str
    num_steps: int
    error: str
    agent: list[str]
    start_time: datetime
    end_time: datetime


class RunEpisodesResponse(BaseModel):
    run_id: str
    total: int
    limit: int
    offset: int
    episodes: list[RunEpisodeItem]


class StepSummaryItem(BaseModel):
    step_index: int
    player: str | None
    action: str | None
    summary: str


class EpisodeStepSummariesItem(BaseModel):
    episode_id: str
    matchup_id: str | None
    status: str
    winner: str
    num_steps: int
    agent: list[str]
    steps: list[StepSummaryItem]


class RunStepSummariesResponse(BaseModel):
    run_id: str
    total: int
    limit: int
    offset: int
    episodes: list[EpisodeStepSummariesItem]


class StepEntropyPoint(BaseModel):
    step_index: int
    entropy: float


class OpponentDistribution(BaseModel):
    opponent_model: str
    distribution: list[float]


class ModelAnalyticsSummary(BaseModel):
    model: str
    total_actions: int
    action_distribution: list[float]
    step_entropy: list[StepEntropyPoint]
    opponents: list[OpponentDistribution]


class RunAnalyticsResponse(BaseModel):
    run_id: str
    run_type: str
    status: str
    action_space: int
    episode_status_counts: dict[str, int]
    model_summaries: list[ModelAnalyticsSummary]


class LeaderboardRow(BaseModel):
    game: str
    model: str
    rating: float
    games_played: int
    wins: int
    draws: int
    losses: int
    win_rate: float | None
    x_win_rate: float | None
    o_win_rate: float | None
    updated_at: datetime


class LeaderboardResponse(BaseModel):
    game: str | None
    limit: int
    rows: list[LeaderboardRow]


class ClientStylizeRequest(BaseModel):
    title: str = Field(default="Arena Client Request")
    summary: str = Field(min_length=1, max_length=5000)
    highlights: list[str] = Field(default_factory=list, max_length=12)
    tone: Literal["technical", "executive", "compact"] = Field(default="technical")


class ClientStylizeResponse(BaseModel):
    title: str
    tone: str
    styled_markdown: str
    styled_text: str
    generated_at: datetime


bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")
api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False)


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _estimate_tokens(payload: Any) -> int:
    text = str(payload)
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text)))
    # Fallback approximation if tiktoken is unavailable.
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _rate_limit_keys(api_key_hash: str, minute_bucket: int) -> tuple[str, str]:
    return (
        f"api:rl:req:{api_key_hash}:{minute_bucket}",
        f"api:rl:tok:{api_key_hash}:{minute_bucket}",
    )


# TODO: Rework the bucket rate limiting? We can try running this to see if it would actually work
def _enforce_rate_limit(api_key: str, token_cost: int) -> tuple[int, int, int]:
    minute_bucket = int(time.time() // 60)
    req_key, tok_key = _rate_limit_keys(_hash_api_key(api_key), minute_bucket)
    result = _resolve_sync_result(
        redis_conn.eval(
            _RATE_LIMIT_SCRIPT,
            2,
            req_key,
            tok_key,
            "1",
            str(token_cost),
            str(REQUESTS_PER_MINUTE),
            str(TOKENS_PER_MINUTE),
            str(RATE_LIMIT_TTL_SECONDS),
        )
    )
    if not isinstance(result, list) or len(result) != 3:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter returned an unexpected response",
        )
    allowed, req_count, tok_count = result
    if int(allowed) != 1:
        increment_counter("api.rate_limit.exceeded", tags={"reason": "quota"})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Rate limit exceeded. "
                f"Limits: {REQUESTS_PER_MINUTE} req/min and {TOKENS_PER_MINUTE} tokens/min."
            ),
        )
    return int(req_count), int(tok_count), minute_bucket


def _authenticate(
    bearer_credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(bearer_scheme)
    ],
    header_api_key: Annotated[str | None, Security(api_key_scheme)],
) -> str:
    bearer_token = bearer_credentials.credentials if bearer_credentials else None
    key = header_api_key or bearer_token
    if not key or key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return key


def _build_run_status(run_id: str) -> RunStatusResponse:
    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        episodes_saved = (
            session.query(func.count(Episode.episode_id))
            .filter(Episode.run_id == run_id)
            .scalar()
        )
        rows = session.query(Matchup).filter(Matchup.run_id == run_id).all()
        matchup_payload = [
            MatchupStatus(
                matchup_id=row.matchup_id,
                models=list(row.models) if isinstance(row.models, list) else [],
                num_episodes=row.num_episodes,
                status=row.status,
            )
            for row in rows
        ]
        return RunStatusResponse(
            run_id=run.run_id,
            run_type=run.run_type,
            status=run.status,
            num_episodes=run.num_episodes,
            episodes_saved=int(episodes_saved or 0),
            created_at=run.created_at,
            matchups=matchup_payload,
        )
    finally:
        session.close()


def _parse_action_column(action: str) -> int | None:
    parsed = re.match(r"^[a-zA-Z](\d+)$", (action or "").strip())
    if parsed is None:
        return None
    return int(parsed.group(1))


def _safe_entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _build_run_analytics(run_id: str, action_space: int) -> RunAnalyticsResponse:
    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        episode_rows = (
            session.query(Episode.status, Episode.agent, Episode.episode_id)
            .filter(Episode.run_id == run_id)
            .all()
        )
        if not episode_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No episodes found for run {run_id}",
            )

        step_rows = (
            session.query(
                Episode.agent,
                EpisodeStep.step_index,
                EpisodeStep.player,
                EpisodeStep.action,
            )
            .join(EpisodeStep, Episode.episode_id == EpisodeStep.episode_id)
            .filter(Episode.run_id == run_id)
            .all()
        )

        status_counts: dict[str, int] = {}
        for status_value, _, _ in episode_rows:
            status_counts[status_value] = status_counts.get(status_value, 0) + 1

        model_action_counts: dict[str, Counter[int]] = defaultdict(Counter)
        model_step_action_counts: dict[str, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        model_opponent_counts: dict[str, dict[str, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

        for agent_list, step_index, player, action in step_rows:
            if not isinstance(agent_list, list) or len(agent_list) < 2:
                continue
            action_col = _parse_action_column(action)
            if action_col is None or action_col < 0 or action_col >= action_space:
                continue
            if player == "x":
                acting_model = str(agent_list[0]).strip()
                opponent_model = str(agent_list[1]).strip()
            elif player == "o":
                acting_model = str(agent_list[1]).strip()
                opponent_model = str(agent_list[0]).strip()
            else:
                continue

            model_action_counts[acting_model][action_col] += 1
            model_step_action_counts[acting_model][step_index][action_col] += 1
            model_opponent_counts[acting_model][opponent_model][action_col] += 1

        summaries: list[ModelAnalyticsSummary] = []
        all_models = set(model_action_counts.keys()) | {
            str(model).strip()
            for _, agents, _ in episode_rows
            for model in (agents if isinstance(agents, list) else [])
        }
        for model in sorted(all_models):
            overall = model_action_counts.get(model, Counter())
            total = sum(overall.values())
            if total > 0:
                distribution = [
                    overall.get(action, 0) / total for action in range(action_space)
                ]
            else:
                distribution = [0.0 for _ in range(action_space)]

            step_entropy: list[StepEntropyPoint] = []
            for step_index in sorted(model_step_action_counts.get(model, {}).keys()):
                step_counts = model_step_action_counts[model][step_index]
                step_entropy.append(
                    StepEntropyPoint(
                        step_index=step_index,
                        entropy=round(_safe_entropy(step_counts), 6),
                    )
                )

            opponents: list[OpponentDistribution] = []
            for opponent in sorted(model_opponent_counts.get(model, {}).keys()):
                opp_counter = model_opponent_counts[model][opponent]
                opp_total = sum(opp_counter.values())
                if opp_total <= 0:
                    opp_distribution = [0.0 for _ in range(action_space)]
                else:
                    opp_distribution = [
                        opp_counter.get(action, 0) / opp_total
                        for action in range(action_space)
                    ]
                opponents.append(
                    OpponentDistribution(
                        opponent_model=opponent,
                        distribution=opp_distribution,
                    )
                )

            summaries.append(
                ModelAnalyticsSummary(
                    model=model,
                    total_actions=total,
                    action_distribution=distribution,
                    step_entropy=step_entropy,
                    opponents=opponents,
                )
            )

        return RunAnalyticsResponse(
            run_id=run.run_id,
            run_type=run.run_type,
            status=run.status,
            action_space=action_space,
            episode_status_counts=status_counts,
            model_summaries=summaries,
        )
    finally:
        session.close()


def _render_stylized_client_payload(
    request: ClientStylizeRequest,
) -> tuple[str, str]:
    summary_line = request.summary.strip()
    highlights = [item.strip() for item in request.highlights if item.strip()]
    if request.tone == "executive":
        header = f"## {request.title}\n\n{summary_line}\n"
        bullets = "".join(f"- {item}\n" for item in highlights)
        markdown = f"{header}\n### Highlights\n{bullets or '- None provided\\n'}"
        text_lines = [
            f"{request.title.upper()}",
            "=" * max(10, len(request.title)),
            summary_line,
            "",
            "Highlights:",
            *(f" * {item}" for item in highlights),
        ]
    elif request.tone == "compact":
        pills = " | ".join(highlights) if highlights else "none"
        markdown = f"**{request.title}**: {summary_line}\n\n`{pills}`"
        text_lines = [f"[{request.title}] {summary_line}", f"Highlights: {pills}"]
    else:
        bullets = "".join(f"- `{idx + 1:02d}` {item}\n" for idx, item in enumerate(highlights))
        markdown = (
            f"### {request.title}\n\n"
            f"> {summary_line}\n\n"
            f"#### Structured Highlights\n{bullets or '- `00` none\\n'}"
        )
        text_lines = [
            f"{request.title}",
            "-" * max(8, len(request.title)),
            f"Summary: {summary_line}",
            "Highlights:",
            *(f" - [{idx + 1:02d}] {item}" for idx, item in enumerate(highlights)),
        ]

    text = "\n".join(text_lines).strip()
    return markdown.strip(), text


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=0.5, max=8),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(
        (RedisConnectionError, RedisTimeoutError, OSError)
    ),
    reraise=True,
)
def _enqueue_with_backoff(request: RunCreateRequest) -> tuple[str, list[str]]:
    # Lazy import prevents requiring all model API keys at process startup.
    from orchestrator.jobs import enqueue_run

    return enqueue_run(
        run_type=request.run_type,
        num_episodes=request.num_episodes,
        num_steps=request.num_steps,
        model_names=request.model_names,
        input_actions=request.input_actions,
        env=request.env,
    )


app = FastAPI(
    title="LLM Game Arena API Gateway",
    version="0.1.0",
    description=(
        "API gateway for creating and polling OpenSpiel arena runs. "
        "Includes API-key auth, Redis-backed rate limiting, token-budget accounting, "
        "and enqueue retry/backoff."
    ),
    openapi_tags=[
        {"name": "health", "description": "Gateway health and readiness checks."},
        {"name": "runs", "description": "Run submission and run status polling."},
        {"name": "usage", "description": "Per-API-key rate limit usage snapshots."},
        {"name": "client", "description": "Client-facing response formatting helpers."},
        {
            "name": "observability",
            "description": "Process-local counters/timing snapshots and dashboard views.",
        },
    ],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    start_ms = time_now_ms()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = time_now_ms() - start_ms
        increment_counter(
            "api.requests.total",
            tags={
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
            },
        )
        observe_timing_ms(
            "api.request.duration_ms",
            duration_ms=duration_ms,
            tags={"method": request.method, "path": request.url.path},
        )
        LOGGER.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise
    duration_ms = time_now_ms() - start_ms
    increment_counter(
        "api.requests.total",
        tags={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
        },
    )
    observe_timing_ms(
        "api.request.duration_ms",
        duration_ms=duration_ms,
        tags={"method": request.method, "path": request.url.path},
    )
    LOGGER.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers["x-request-id"] = request_id
    response.headers["x-api-gateway"] = API_GATEWAY_NAME
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    failure = classify_exception(exc, stage="api.unhandled", default_code="internal_error")
    increment_counter(
        "api.requests.error.total",
        tags={"category": failure.category.value, "code": failure.code},
    )
    LOGGER.exception(
        "unhandled_exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "failure_code": failure.code,
            "failure_category": failure.category.value,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": failure.compact(),
            "status_code": 500,
            "code": failure.code,
        },
    )


@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["health"],
    operation_id="getHealth",
    summary="Gateway health check",
    description="Returns gateway health, identity, and current UTC timestamp.",
)
def healthz():
    now = datetime.now(timezone.utc)
    return HealthResponse(
        status="ok",
        gateway=API_GATEWAY_NAME,
        database_target=get_database_target(),
        ts=now,
    )


@app.post(
    "/v1/client/stylize",
    response_model=ClientStylizeResponse,
    tags=["client"],
    operation_id="stylizeClientPayload",
    summary="Render stylized client response text",
    description=(
        "Takes a simple client payload and returns stylized markdown/text output "
        "for direct UI or terminal display."
    ),
    responses={401: {"model": ErrorResponse, "description": "Authentication failed"}},
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def stylize_client_payload(
    request: ClientStylizeRequest,
    _: str = Depends(_authenticate),
):
    markdown, text = _render_stylized_client_payload(request)
    return ClientStylizeResponse(
        title=request.title,
        tone=request.tone,
        styled_markdown=markdown,
        styled_text=text,
        generated_at=datetime.now(timezone.utc),
    )


@app.post(
    "/v1/runs",
    response_model=RunCreateResponse,
    status_code=202,
    tags=["runs"],
    operation_id="createRun",
    summary="Submit a run",
    description=(
        "Creates a new run and enqueues episode jobs into the orchestrator queue. "
        "Rate limits are enforced per API key by request count and tokenized payload size."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Queue submission unavailable"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def create_run(request: RunCreateRequest, api_key: str = Depends(_authenticate)):
    increment_counter("api.runs.create.attempt")
    token_cost = _estimate_tokens(request.model_dump())
    _enforce_rate_limit(api_key, token_cost)

    try:
        run_id, job_ids = _enqueue_with_backoff(request)
    except Exception as e:
        increment_counter("api.runs.create.error", tags={"reason": "enqueue_failed"})
        LOGGER.exception("Failed to enqueue run after retries: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Failed to submit run after retry/backoff attempts",
        ) from e

    increment_counter("api.runs.create.success")
    LOGGER.info(
        "run_created",
        extra={
            "run_id": run_id,
            "run_type": request.run_type,
            "job_count": len(job_ids),
            "num_episodes": request.num_episodes,
        },
    )
    now = datetime.now(timezone.utc)
    return RunCreateResponse(
        run_id=run_id,
        status="PENDING",
        job_count=len(job_ids),
        status_url=f"/v1/runs/{run_id}",
        submitted_at=now,
    )


@app.get(
    "/v1/runs/{run_id}",
    response_model=RunStatusResponse,
    tags=["runs"],
    operation_id="getRunStatus",
    summary="Poll run status",
    description="Returns run metadata, current status, saved episode count, and matchup status.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid run ID format"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_run_status(run_id: str, api_key: str = Depends(_authenticate)):
    _enforce_rate_limit(api_key, token_cost=1)
    try:
        UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="run_id must be a UUID") from e
    return _build_run_status(run_id)


@app.get(
    "/v1/runs/{run_id}/episodes",
    response_model=RunEpisodesResponse,
    tags=["runs"],
    operation_id="getRunEpisodes",
    summary="List persisted episodes for a run",
    description="Returns paginated episodes persisted for a run.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_run_episodes(
    run_id: str,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(_authenticate),
):
    _enforce_rate_limit(api_key, token_cost=1)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    try:
        UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="run_id must be a UUID") from e

    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        total = (
            session.query(func.count(Episode.episode_id))
            .filter(Episode.run_id == run_id)
            .scalar()
        )
        rows = (
            session.query(Episode)
            .filter(Episode.run_id == run_id)
            .order_by(Episode.start_time.asc(), Episode.episode_id.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        episodes = [
            RunEpisodeItem(
                episode_id=row.episode_id,
                matchup_id=row.matchup_id,
                worker_name=row.worker_name,
                status=row.status,
                winner=row.winner,
                num_steps=row.num_steps,
                error=row.error,
                agent=list(row.agent) if isinstance(row.agent, list) else [],
                start_time=row.start_time,
                end_time=row.end_time,
            )
            for row in rows
        ]
        return RunEpisodesResponse(
            run_id=run_id,
            total=int(total or 0),
            limit=limit,
            offset=offset,
            episodes=episodes,
        )
    finally:
        session.close()


@app.get(
    "/v1/runs/{run_id}/step-summaries",
    response_model=RunStepSummariesResponse,
    tags=["runs"],
    operation_id="getRunStepSummaries",
    summary="List per-step model summaries for a run",
    description=(
        "Returns step-aligned action + summary text for each persisted episode in a run, "
        "so model reasoning can be inspected turn-by-turn."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_run_step_summaries(
    run_id: str,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(_authenticate),
):
    _enforce_rate_limit(api_key, token_cost=1)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    try:
        UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="run_id must be a UUID") from e

    session = get_session()
    try:
        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        total = (
            session.query(func.count(Episode.episode_id))
            .filter(Episode.run_id == run_id)
            .scalar()
        )
        rows = (
            session.query(Episode)
            .filter(Episode.run_id == run_id)
            .order_by(Episode.start_time.asc(), Episode.episode_id.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        episodes_payload: list[EpisodeStepSummariesItem] = []
        for row in rows:
            actions = row.action if isinstance(row.action, list) else []
            summaries = row.summary if isinstance(row.summary, list) else []
            steps: list[StepSummaryItem] = []
            for idx, pair in enumerate(zip_longest(actions, summaries, fillvalue="")):
                action, summary = pair
                action_value = action if isinstance(action, str) and action else None
                summary_value = (
                    summary if isinstance(summary, str) and summary else "No summary."
                )
                if row.game == "chess":
                    player = "w" if idx % 2 == 0 else "b"
                elif action_value:
                    player = action_value[0].lower()
                else:
                    player = None
                steps.append(
                    StepSummaryItem(
                        step_index=idx,
                        player=player,
                        action=action_value,
                        summary=summary_value,
                    )
                )

            episodes_payload.append(
                EpisodeStepSummariesItem(
                    episode_id=row.episode_id,
                    matchup_id=row.matchup_id,
                    status=row.status,
                    winner=row.winner,
                    num_steps=row.num_steps,
                    agent=list(row.agent) if isinstance(row.agent, list) else [],
                    steps=steps,
                )
            )

        return RunStepSummariesResponse(
            run_id=run_id,
            total=int(total or 0),
            limit=limit,
            offset=offset,
            episodes=episodes_payload,
        )
    finally:
        session.close()


@app.get(
    "/v1/runs/{run_id}/analytics",
    response_model=RunAnalyticsResponse,
    tags=["runs"],
    operation_id="getRunAnalytics",
    summary="Get run analytics as JSON",
    description=(
        "Returns model-level action distributions, step entropy, and episode status counts "
        "for frontend chart rendering."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        404: {"model": ErrorResponse, "description": "Run not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_run_analytics(
    run_id: str,
    action_space: int = 7,
    api_key: str = Depends(_authenticate),
):
    _enforce_rate_limit(api_key, token_cost=1)
    if action_space < 2 or action_space > 64:
        raise HTTPException(status_code=400, detail="action_space must be between 2 and 64")
    try:
        UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="run_id must be a UUID") from e
    return _build_run_analytics(run_id=run_id, action_space=action_space)


@app.get(
    "/v1/leaderboard",
    response_model=LeaderboardResponse,
    tags=["runs"],
    operation_id="getLeaderboard",
    summary="Get model leaderboard",
    description="Returns Elo-style ratings from model_ratings table.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_leaderboard(
    game: str | None = None,
    limit: int = 100,
    api_key: str = Depends(_authenticate),
):
    _enforce_rate_limit(api_key, token_cost=1)
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    session = get_session()
    try:
        query = session.query(ModelRating)
        if game:
            query = query.filter(ModelRating.game == game.strip())
        rows = (
            query.order_by(
                ModelRating.game.asc(),
                ModelRating.rating.desc(),
                ModelRating.games_played.desc(),
            )
            .limit(limit)
            .all()
        )
        payload_rows: list[LeaderboardRow] = []
        for row in rows:
            win_rate = (
                float(row.wins) / float(row.games_played) if row.games_played else None
            )
            x_win_rate = (
                float(row.wins_as_x) / float(row.games_as_x) if row.games_as_x else None
            )
            o_win_rate = (
                float(row.wins_as_o) / float(row.games_as_o) if row.games_as_o else None
            )
            payload_rows.append(
                LeaderboardRow(
                    game=row.game,
                    model=row.model_name,
                    rating=float(row.rating),
                    games_played=int(row.games_played),
                    wins=int(row.wins),
                    draws=int(row.draws),
                    losses=int(row.losses),
                    win_rate=win_rate,
                    x_win_rate=x_win_rate,
                    o_win_rate=o_win_rate,
                    updated_at=row.updated_at,
                )
            )
        return LeaderboardResponse(game=game, limit=limit, rows=payload_rows)
    finally:
        session.close()


@app.get(
    "/v1/usage/me",
    response_model=UsageResponse,
    tags=["usage"],
    operation_id="getUsage",
    summary="Get current API key usage",
    description=(
        "Returns request/token usage counters for the caller in the active UTC minute bucket."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
    },
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_my_usage(api_key: str = Depends(_authenticate)):
    minute_bucket = int(time.time() // 60)
    req_key, tok_key = _rate_limit_keys(_hash_api_key(api_key), minute_bucket)
    req_count_raw = _resolve_sync_result(redis_conn.get(req_key))
    tok_count_raw = _resolve_sync_result(redis_conn.get(tok_key))
    req_count = int(req_count_raw or 0)
    tok_count = int(tok_count_raw or 0)
    return UsageResponse(
        request_count=req_count,
        token_count=tok_count,
        requests_per_minute_limit=REQUESTS_PER_MINUTE,
        tokens_per_minute_limit=TOKENS_PER_MINUTE,
        bucket_epoch_minute=minute_bucket,
    )


@app.get(
    "/v1/observability/metrics",
    response_model=ObservabilitySnapshotResponse,
    tags=["observability"],
    operation_id="getObservabilityMetrics",
    summary="Get process-local metrics snapshot",
    description="Returns in-memory counters and timing aggregates for this API process.",
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_observability_metrics(_: str = Depends(_authenticate)):
    return get_metrics_snapshot()


@app.get(
    "/v1/observability/dashboard",
    response_model=ObservabilityDashboardResponse,
    tags=["observability"],
    operation_id="getObservabilityDashboard",
    summary="Get dashboard-oriented observability view",
    description="Returns top counters and highest-latency timing series for quick inspection.",
    openapi_extra={"security": [{"BearerAuth": []}, {"APIKeyHeader": []}]},
)
def get_observability_dashboard(_: str = Depends(_authenticate)):
    return build_dashboard_view()


def main() -> None:
    uvicorn.run(
        "orchestrator.api_gateway:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
