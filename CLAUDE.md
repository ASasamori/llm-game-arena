# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llm-game-arena is a pipeline for evaluating LLMs in game environments, primarily using OpenSpiel. The system follows a distributed architecture:

**Client Request → Run Orchestrator → Workers → Evals → Client Polling**

The orchestrator spawns parallelized episode executions (episodes are independent), workers execute game episodes with LLM agents, and evaluations aggregate metrics across runs.

## Core Architecture

### Episode Execution Flow (game_arena/openspiel/)
- `harness.py`: Main episode execution harness. The `main()` function runs a single episode and returns an `EpisodeResult`
- `model_generation.py`: LLM client wrappers (currently `GeminiModel` with retry logic using tenacity)
- `prompts.py`: Prompt templates for different game scenarios (e.g., `PROMPT_TEMPLATE_NO_LEGAL_ACTIONS`)
- `parsers.py`: Parse LLM responses to extract moves (supports chained parsers via `ChainedMoveParser`)
- `notations.py`: Game-specific notation mappings and state representations
- `utils.py`: Convert PySpiel states to readable formats, get action histories

### Data Types
- `EpisodeResult` (defined in both `harness.py` and `orchestrator/worker.py`): Contains `episode_id`, `env`, `agent`, `game`, `status`, `winner`, `num_steps`, `error`

### Planned Infrastructure (see orchestrator/next_steps.md)
The project is architected for eventual distributed execution with:
- FastAPI for the control plane (Run Orchestrator)
- Redis + RQ for job queuing
- PostgreSQL with 3 tables: Runs, Episodes, Episode Steps
- Async execution with multiple workers for failure isolation

Currently, the harness runs episodes synchronously but is structured to plug into this queue-based architecture.

## Commands

### Setup
```bash
# Install dependencies (Python 3.12+)
python -m pip install -e .

# Environment variables required
export GEMINI_API_KEY=<your-key>
```

### Development
```bash
# Run type checking
pyright

# Run linting/formatting
ruff check .
ruff format .

# Run a single test/example
python -m game_arena.openspiel.test
```

### Running Episodes
The harness (`game_arena/openspiel/harness.py`) can be imported and called programmatically:
```python
from game_arena.openspiel.harness import main
result = main(NUM_MOVES=10, AI_MODEL="gemini-2.5-flash")
```

## Development Notes

### PySpiel Integration
- Games are loaded via `pyspiel.load_game("connect_four")`
- State management uses `pyspiel.State` with methods like `is_terminal()`, `legal_actions()`, `apply_action()`
- Current player accessed via `state.current_player()` (0 or 1)
- Returns accessed via `state.returns()` to determine winner

### LLM Response Parsing
The parser pipeline is crucial for extracting valid moves from LLM outputs:
1. LLMs generate text responses with reasoning followed by "Final Answer: X"
2. `parsers.py` extracts moves using regex patterns
3. Supports fallback parsing strategies via `ChainedMoveParser`

### Game Configuration
- Currently hardcoded for Connect Four
- Opponent is a random player (player 1)
- Future: Support multiple games, opponent policies, and AI vs AI matches

### Current Limitations
- Only Connect Four is implemented
- No persistent storage yet (DB schema planned but not implemented)
- No API endpoints yet (FastAPI planned)
- Episodes run synchronously (worker/queue system planned)
- Only Gemini model wrapper exists (OpenAI integration available via dependencies)

## Next Development Steps (from orchestrator/next_steps.md)

1. Implement `run_ep()` function that returns structured `EpisodeResult`
2. Set up database schema (Runs, Episodes, Episode Steps tables)
3. Create FastAPI control plane (Run Orchestrator with immediate returns)
4. Add Redis + RQ for local queuing
5. Write worker to consume episode jobs
6. Add aggregation/analytics (win rate, average steps)
