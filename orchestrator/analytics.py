"""
CLI analytics for arena runs.

Commands:
1) Generate all available plots for a completed competitive run (PVP or PVP_RR):
   uv run python -m orchestrator.analytics --run-id <COMPETITIVE_RUN_ID>

2) Generate plots with explicit RAND baselines for delta charts:
   uv run python -m orchestrator.analytics \
     --run-id <COMPETITIVE_RUN_ID> \
     --baseline-run 'GPTModel(gpt-4o-mini)'=<RAND_RUN_ID_FOR_GPT_4O_MINI> \
     --baseline-run 'GeminiModel(gemini-2.5-flash-lite)'=<RAND_RUN_ID_FOR_GEMINI_2_5_FLASH_LITE> \
     --baseline-run 'ClaudeModel(claude-3-haiku-20240307)'=<RAND_RUN_ID_FOR_CLAUDE_HAIKU>

3) Generate plots into a custom directory:
   uv run python -m orchestrator.analytics \
     --run-id <COMPETITIVE_RUN_ID> \
     --output-dir analytics/custom_report

4) Analyze before run completion (if needed):
   uv run python -m orchestrator.analytics \
     --run-id <COMPETITIVE_RUN_ID> \
     --allow-incomplete

Outputs:
- heatmap_<MODEL>.png       (x=step, y=action, color=P(action|step))
- entropy_<MODEL>.png       (x=step, y=entropy)
- delta_<MODEL>.png         (x=action, y=probability difference vs baseline)
"""

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from orchestrator.models import Episode, EpisodeStep, Run, get_session


ACTION_RE = re.compile(r"^[a-zA-Z](\d+)$")
_PLOT_MODULE = None


def _get_plt():
    global _PLOT_MODULE
    if _PLOT_MODULE is None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "matplotlib is required for plotting. Install dependencies first (e.g. `uv sync`)."
            ) from e
        _PLOT_MODULE = plt
    return _PLOT_MODULE


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    action: int
    acting_model: str
    opponent_model: str


def _normalize_model_name(model_name: str) -> str:
    return (model_name or "").strip()


def _model_aliases(model_name: str) -> set[str]:
    full = _normalize_model_name(model_name)
    upper = full.upper()
    aliases = {full, upper}
    if "GPT" in upper:
        aliases.add("GPT")
    if "GEMINI" in upper:
        aliases.add("GEMINI")
    if "CLAUDE" in upper:
        aliases.add("CLAUDE")
    return aliases


def _resolve_model_key(model_key: str, known_models: set[str]) -> str:
    normalized = _normalize_model_name(model_key)
    if normalized in known_models:
        return normalized

    matches = [m for m in known_models if normalized.upper() in _model_aliases(m)]
    if len(matches) == 1:
        return matches[0]
    return normalized


def _parse_action_column(action: str) -> int | None:
    if not action:
        return None
    parsed = ACTION_RE.match(action.strip())
    if not parsed:
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


def _build_step_records(run_id: str) -> list[StepRecord]:
    session = get_session()
    try:
        rows = (
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
    finally:
        session.close()

    records: list[StepRecord] = []
    for agent_list, step_index, player, action in rows:
        if not isinstance(agent_list, list) or len(agent_list) < 2:
            continue
        action_col = _parse_action_column(action)
        if action_col is None:
            continue

        if player == "x":
            acting_model = _normalize_model_name(str(agent_list[0]))
            opponent_model = _normalize_model_name(str(agent_list[1]))
        elif player == "o":
            acting_model = _normalize_model_name(str(agent_list[1]))
            opponent_model = _normalize_model_name(str(agent_list[0]))
        else:
            continue

        records.append(
            StepRecord(
                step_index=step_index,
                action=action_col,
                acting_model=acting_model,
                opponent_model=opponent_model,
            )
        )

    return records


def _build_model_step_action_counts(
    records: list[StepRecord],
) -> dict[str, dict[int, Counter[int]]]:
    counts: dict[str, dict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for record in records:
        counts[record.acting_model][record.step_index][record.action] += 1
    return counts


def _build_model_opponent_step_action_counts(
    records: list[StepRecord],
) -> dict[str, dict[str, dict[int, Counter[int]]]]:
    counts: dict[str, dict[str, dict[int, Counter[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for record in records:
        counts[record.acting_model][record.opponent_model][record.step_index][
            record.action
        ] += 1
    return counts


def _distribution_from_records(
    records: list[StepRecord], action_space: int
) -> dict[str, dict[str, list[float]]]:
    model_vs_opponent_counts: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for record in records:
        if 0 <= record.action < action_space:
            model_vs_opponent_counts[record.acting_model][record.opponent_model][
                record.action
            ] += 1

    distributions: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for model, by_opp in model_vs_opponent_counts.items():
        for opponent, counter in by_opp.items():
            total = sum(counter.values())
            if total <= 0:
                continue
            distributions[model][opponent] = [
                counter.get(action, 0) / total for action in range(action_space)
            ]

    return distributions


def _auto_find_baseline_runs(models: set[str]) -> dict[str, str]:
    session = get_session()
    baseline_map: dict[str, str] = {}
    try:
        for model in models:
            rows = (
                session.query(Episode.run_id)
                .join(Run, Episode.run_id == Run.run_id)
                .filter(Run.run_type == "RAND", Run.status == "COMPLETED")
                .order_by(Run.created_at.desc())
                .all()
            )
            for (run_id,) in rows:
                ep = (
                    session.query(Episode.agent)
                    .filter(Episode.run_id == run_id)
                    .first()
                )
                if (
                    ep
                    and isinstance(ep.agent, list)
                    and len(ep.agent) >= 1
                    and _normalize_model_name(str(ep.agent[0]))
                    == _normalize_model_name(model)
                ):
                    baseline_map[model] = run_id
                    break
    finally:
        session.close()
    return baseline_map


def _parse_baseline_args(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --baseline-run value '{item}'. Expected MODEL=RUN_ID"
            )
        model, run_id = item.split("=", 1)
        model = model.strip()
        run_id = run_id.strip()
        if not model or not run_id:
            raise ValueError(
                f"Invalid --baseline-run value '{item}'. Expected MODEL=RUN_ID"
            )
        parsed[_normalize_model_name(model)] = run_id
    return parsed


def _fetch_baseline_distributions(
    baseline_runs: dict[str, str], action_space: int
) -> dict[str, list[float]]:
    baseline_distributions: dict[str, list[float]] = {}
    session = get_session()
    try:
        for model, run_id in baseline_runs.items():
            rows = (
                session.query(Episode.agent, EpisodeStep.player, EpisodeStep.action)
                .join(EpisodeStep, Episode.episode_id == EpisodeStep.episode_id)
                .filter(Episode.run_id == run_id)
                .all()
            )

            counts: Counter[int] = Counter()
            for agent_list, player, action in rows:
                if not isinstance(agent_list, list) or len(agent_list) < 1:
                    continue

                action_col = _parse_action_column(action)
                if action_col is None or action_col < 0 or action_col >= action_space:
                    continue

                # In RAND runs, model is expected to be player x.
                if player == "x" and _normalize_model_name(
                    str(agent_list[0])
                ) == _normalize_model_name(model):
                    counts[action_col] += 1

            total = sum(counts.values())
            if total > 0:
                baseline_distributions[model] = [
                    counts.get(a, 0) / total for a in range(action_space)
                ]
    finally:
        session.close()

    return baseline_distributions


def _plot_heatmaps(
    counts: dict[str, dict[int, Counter[int]]],
    action_space: int,
    output_dir: str,
    dpi: int,
) -> None:
    plt = _get_plt()
    for model, by_step in counts.items():
        if not by_step:
            continue

        max_step = max(by_step.keys())
        # Build matrix as [action][step] so axes are x=step, y=action.
        matrix: list[list[float]] = [
            [0.0 for _ in range(max_step + 1)] for _ in range(action_space)
        ]
        for step in range(max_step + 1):
            row_counts = by_step.get(step, Counter())
            total = sum(row_counts.values())
            if total <= 0:
                continue
            for action in range(action_space):
                matrix[action][step] = row_counts.get(action, 0) / total

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"Action Preference Heatmap | {model}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action (column)")
        ax.set_xticks(list(range(max_step + 1)))
        ax.set_yticks(list(range(action_space)))
        fig.colorbar(im, ax=ax, label="P(action | step)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_{model}.png"), dpi=dpi)
        plt.close(fig)


def _plot_entropy_lines(
    counts: dict[str, dict[str, dict[int, Counter[int]]]],
    output_dir: str,
    dpi: int,
) -> None:
    plt = _get_plt()
    for model, by_opponent in counts.items():
        if not by_opponent:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False

        for opponent, by_step in sorted(by_opponent.items()):
            if not by_step:
                continue

            max_step = max(by_step.keys())
            x_vals = list(range(max_step + 1))
            y_vals = [_safe_entropy(by_step.get(step, Counter())) for step in x_vals]
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=f"vs {opponent}")
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_title(f"Step Entropy by Opponent | {model}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy (bits)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"entropy_{model}.png"), dpi=dpi)
        plt.close(fig)


def _plot_delta_vs_baseline(
    current_distributions: dict[str, dict[str, list[float]]],
    baseline_distributions: dict[str, list[float]],
    action_space: int,
    output_dir: str,
    dpi: int,
) -> None:
    plt = _get_plt()
    for model, by_opponent in current_distributions.items():
        baseline = baseline_distributions.get(model)
        if baseline is None:
            continue

        opponents = sorted(by_opponent.keys())
        if not opponents:
            continue

        width = 0.8 / max(1, len(opponents))
        x_positions = [float(i) for i in range(action_space)]

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, opponent in enumerate(opponents):
            cur = by_opponent[opponent]
            deltas = [cur[action] - baseline[action] for action in range(action_space)]
            offsets = [
                x + (idx - (len(opponents) - 1) / 2.0) * width for x in x_positions
            ]
            ax.bar(offsets, deltas, width=width, label=f"vs {opponent}")

        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Delta Action Probability vs RAND Baseline | {model}")
        ax.set_xlabel("Action (column)")
        ax.set_ylabel("Delta P(action)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in range(action_space)])
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"delta_{model}.png"), dpi=dpi)
        plt.close(fig)


def _validate_run(run_id: str, allow_incomplete: bool) -> Run:
    session = get_session()
    try:
        run = session.query(Run).filter_by(run_id=run_id).first()
    finally:
        session.close()

    if run is None:
        raise ValueError(f"Run {run_id} not found")
    if run.run_type not in {"PVP", "PVP_RR"}:
        raise ValueError(
            f"Analytics CLI expects run_type in {{'PVP', 'PVP_RR'}}. Got run_type={run.run_type}"
        )
    if not allow_incomplete and run.status != "COMPLETED":
        raise ValueError(
            f"Run {run_id} status is {run.status}. Wait for COMPLETED or pass --allow-incomplete"
        )
    return run


def _fetch_run_models(run_id: str) -> set[str]:
    session = get_session()
    try:
        rows = session.query(Episode.agent).filter(Episode.run_id == run_id).all()
    finally:
        session.close()

    models: set[str] = set()
    for (agent_list,) in rows:
        if isinstance(agent_list, list):
            for model in agent_list:
                models.add(str(model))
    return {_normalize_model_name(model) for model in models}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate run-level analytics plots from Postgres episode tables"
    )
    parser.add_argument(
        "--run-id", required=True, help="Competitive run_id (PVP or PVP_RR) to analyze"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output PNG files (default: analytics/<run_id>)",
    )
    parser.add_argument(
        "--action-space",
        type=int,
        default=7,
        help="Number of discrete actions (Connect Four columns = 7)",
    )
    parser.add_argument(
        "--baseline-run",
        action="append",
        default=[],
        help="Manual baseline mapping as MODEL=RUN_ID. Repeat per model.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow analytics for runs not marked COMPLETED",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output image DPI")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    _validate_run(args.run_id, args.allow_incomplete)

    output_dir = args.output_dir or os.path.join("analytics", "runs", args.run_id)
    os.makedirs(output_dir, exist_ok=True)

    records = _build_step_records(args.run_id)
    if not records:
        raise ValueError(
            f"No usable step records found for run {args.run_id}. "
            "Confirm episodes and episode_steps are populated."
        )

    model_step_counts = _build_model_step_action_counts(records)
    model_opp_step_counts = _build_model_opponent_step_action_counts(records)
    current_distributions = _distribution_from_records(records, args.action_space)

    manual_baselines = _parse_baseline_args(args.baseline_run)
    run_models = _fetch_run_models(args.run_id)

    baseline_run_map: dict[str, str] = {}
    for model_key, baseline_run_id in manual_baselines.items():
        resolved_key = _resolve_model_key(model_key, run_models)
        baseline_run_map[resolved_key] = baseline_run_id
    missing_models = run_models - set(baseline_run_map.keys())
    if missing_models:
        baseline_run_map.update(_auto_find_baseline_runs(missing_models))

    baseline_distributions = _fetch_baseline_distributions(
        baseline_run_map, args.action_space
    )

    _plot_heatmaps(
        counts=model_step_counts,
        action_space=args.action_space,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    _plot_entropy_lines(
        counts=model_opp_step_counts,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    _plot_delta_vs_baseline(
        current_distributions=current_distributions,
        baseline_distributions=baseline_distributions,
        action_space=args.action_space,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print(f"[ANALYTICS] Generated plots in: {output_dir}")
    print("[ANALYTICS] Heatmaps: heatmap_<MODEL>.png")
    print("[ANALYTICS] Entropy lines: entropy_<MODEL>.png")
    if baseline_distributions:
        print("[ANALYTICS] Delta bars: delta_<MODEL>.png")
    else:
        print("[ANALYTICS] Delta bars skipped (no RAND baselines found/provided)")


if __name__ == "__main__":
    main()
