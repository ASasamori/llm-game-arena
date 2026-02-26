FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build tools are required if any dependency falls back to source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md uv.lock ./
COPY game_arena ./game_arena
COPY orchestrator ./orchestrator
COPY data ./data
COPY utils ./utils
COPY config ./config

RUN pip install --no-cache-dir --upgrade pip uv && \
    uv sync --frozen --no-install-project && \
    .venv/bin/pip install --no-cache-dir fastapi uvicorn tiktoken
