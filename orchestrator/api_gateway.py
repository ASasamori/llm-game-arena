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
from collections.abc import Awaitable
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

from orchestrator.models import Episode, Matchup, Run, get_session
from orchestrator.observability import (
    configure_json_logging,
    increment_counter,
    observe_timing_ms,
    time_now_ms,
)

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
        description="Environment version.",
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
        if self.env != "connect_four_v1":
            raise ValueError("Only env='connect_four_v1' is supported right now")
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
    ts: datetime


class ErrorResponse(BaseModel):
    error: str | dict[str, Any] | list[Any]
    status_code: int


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
    )


app = FastAPI(
    title="LLM Game Arena API Gateway",
    version="0.1.0",
    description=(
        "API gateway for creating and polling OpenSpiel Connect Four arena runs. "
        "Includes API-key auth, Redis-backed rate limiting, token-budget accounting, "
        "and enqueue retry/backoff."
    ),
    openapi_tags=[
        {"name": "health", "description": "Gateway health and readiness checks."},
        {"name": "runs", "description": "Run submission and run status polling."},
        {"name": "usage", "description": "Per-API-key rate limit usage snapshots."},
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
    return HealthResponse(status="ok", gateway=API_GATEWAY_NAME, ts=now)


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


def main() -> None:
    uvicorn.run(
        "orchestrator.api_gateway:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
