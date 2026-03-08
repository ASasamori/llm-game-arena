from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCategory(str, Enum):
    VALIDATION = "validation"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    MODEL_API = "model_api"
    PARSER = "parser"
    INVALID_ACTION = "invalid_action"
    QUEUE = "queue"
    DB = "db"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


@dataclass(frozen=True)
class FailureInfo:
    category: FailureCategory
    code: str
    stage: str
    retryable: bool
    message: str

    def compact(self) -> str:
        retryable = "true" if self.retryable else "false"
        return (
            f"[category={self.category.value} code={self.code} "
            f"stage={self.stage} retryable={retryable}] {self.message}"
        )


def classify_exception(
    exc: Exception,
    *,
    stage: str,
    default_category: FailureCategory = FailureCategory.INTERNAL,
    default_code: str = "unhandled_exception",
) -> FailureInfo:
    msg = str(exc) or exc.__class__.__name__
    lowered = msg.lower()
    cls_name = exc.__class__.__name__.lower()

    if "invalid api key" in lowered:
        return FailureInfo(FailureCategory.AUTH, "invalid_api_key", stage, False, msg)
    if "rate limit" in lowered or "too many requests" in lowered:
        return FailureInfo(
            FailureCategory.RATE_LIMIT, "rate_limit_exceeded", stage, True, msg
        )
    if "timeout" in lowered:
        return FailureInfo(FailureCategory.TIMEOUT, "timeout", stage, True, msg)
    if "redis" in lowered or "queue" in lowered:
        return FailureInfo(FailureCategory.QUEUE, "queue_unavailable", stage, True, msg)
    if "sql" in lowered or "database" in lowered or "psycopg" in lowered:
        return FailureInfo(FailureCategory.DB, "db_error", stage, True, msg)
    if "parser" in lowered:
        return FailureInfo(FailureCategory.PARSER, "parser_error", stage, False, msg)
    if "legal action" in lowered or "invalid action" in lowered:
        return FailureInfo(
            FailureCategory.INVALID_ACTION, "illegal_action", stage, False, msg
        )
    if "validation" in lowered or cls_name == "valueerror":
        return FailureInfo(
            FailureCategory.VALIDATION, "validation_error", stage, False, msg
        )
    if "openai" in lowered or "anthropic" in lowered or "gemini" in lowered:
        return FailureInfo(
            FailureCategory.MODEL_API, "model_api_error", stage, True, msg
        )

    return FailureInfo(default_category, default_code, stage, False, msg)
