"""
Elo ratings CLI for OpenSpiel arena episodes.

Examples:
1) Update ratings from one completed PVP or PVP_RR run:
   uv run python -m orchestrator.ratings --run-id <RUN_ID>

2) Rebuild ratings from all completed competitive (PVP/PVP_RR) episodes:
   uv run python -m orchestrator.ratings --rebuild-all

3) Restrict rebuild/update to one game:
   uv run python -m orchestrator.ratings --rebuild-all --game connect_four

4) Export leaderboard summary JSON:
   uv run python -m orchestrator.ratings --rebuild-all --summary-json ratings_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_

from orchestrator.models import (
    Episode,
    ModelRating,
    RatingEvent,
    Run,
    create_tables,
    get_session,
)


DRAW_LABEL = "UNDETERMINED/TIE"


@dataclass
class EloConfig:
    initial_rating: float
    k_provisional: float
    k_established: float
    provisional_games: int


def _normalize_model_name(model_name: str) -> str:
    return (model_name or "").strip()


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _k_factor(games_played: int, config: EloConfig) -> float:
    return (
        config.k_provisional
        if games_played < config.provisional_games
        else config.k_established
    )


def _winner_to_score_a(winner: str, model_a: str, model_b: str) -> float:
    normalized = _normalize_model_name(winner)
    if normalized == model_a:
        return 1.0
    if normalized == model_b:
        return 0.0
    return 0.5


def _get_or_create_rating(
    session,
    cache: dict[tuple[str, str], ModelRating],
    *,
    model_name: str,
    game: str,
    initial_rating: float,
) -> ModelRating:
    key = (model_name, game)
    if key in cache:
        return cache[key]

    row = (
        session.query(ModelRating)
        .filter(
            and_(
                ModelRating.model_name == model_name,
                ModelRating.game == game,
            )
        )
        .first()
    )
    if row is None:
        row = ModelRating(
            rating_id=str(uuid.uuid4()),
            model_name=model_name,
            game=game,
            rating=initial_rating,
            games_played=0,
            wins=0,
            draws=0,
            losses=0,
            games_as_x=0,
            games_as_o=0,
            wins_as_x=0,
            wins_as_o=0,
            updated_at=datetime.utcnow(),
        )
        session.add(row)
    cache[key] = row
    return row


def _validate_run(session, run_id: str, allow_incomplete: bool) -> None:
    run = session.query(Run).filter(Run.run_id == run_id).first()
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    if run.run_type not in {"PVP", "PVP_RR"}:
        raise ValueError(
            f"Run {run_id} is run_type={run.run_type}, expected PVP or PVP_RR"
        )
    if not allow_incomplete and run.status != "COMPLETED":
        raise ValueError(
            f"Run {run_id} status is {run.status}. Wait for COMPLETED or pass --allow-incomplete"
        )


def _fetch_target_episodes(
    session,
    *,
    run_id: str | None,
    game: str | None,
    allow_incomplete: bool,
) -> list[Episode]:
    query = (
        session.query(Episode)
        .join(Run, Episode.run_id == Run.run_id)
        .filter(
            Run.run_type.in_(["PVP", "PVP_RR"]),
            Episode.status == "SUCCESS",
        )
    )

    if not allow_incomplete:
        query = query.filter(Run.status == "COMPLETED")
    if run_id is not None:
        query = query.filter(Episode.run_id == run_id)
    if game is not None:
        query = query.filter(Episode.game == game)

    return query.order_by(
        Episode.end_time.asc(), Episode.start_time.asc(), Episode.episode_id.asc()
    ).all()


def _update_record_counts(row: ModelRating, *, score: float, as_x: bool) -> None:
    row.games_played += 1
    if as_x:
        row.games_as_x += 1
    else:
        row.games_as_o += 1

    if score == 1.0:
        row.wins += 1
        if as_x:
            row.wins_as_x += 1
        else:
            row.wins_as_o += 1
    elif score == 0.5:
        row.draws += 1
    else:
        row.losses += 1


def _process_episodes(
    session,
    episodes: list[Episode],
    *,
    config: EloConfig,
    skip_existing_events: bool,
) -> tuple[int, int]:
    cache: dict[tuple[str, str], ModelRating] = {}
    processed = 0
    skipped = 0

    existing_episode_ids: set[str] = set()
    if skip_existing_events and episodes:
        candidate_ids = [ep.episode_id for ep in episodes]
        rows = (
            session.query(RatingEvent.episode_id)
            .filter(RatingEvent.episode_id.in_(candidate_ids))
            .all()
        )
        existing_episode_ids = {episode_id for (episode_id,) in rows}

    for ep in episodes:
        if skip_existing_events and ep.episode_id in existing_episode_ids:
            skipped += 1
            continue

        if not isinstance(ep.agent, list) or len(ep.agent) < 2:
            skipped += 1
            continue

        player_x_model = _normalize_model_name(str(ep.agent[0]))
        player_o_model = _normalize_model_name(str(ep.agent[1]))
        if not player_x_model or not player_o_model:
            skipped += 1
            continue
        if player_x_model == player_o_model:
            skipped += 1
            continue

        row_a = _get_or_create_rating(
            session,
            cache,
            model_name=player_x_model,
            game=ep.game,
            initial_rating=config.initial_rating,
        )
        row_b = _get_or_create_rating(
            session,
            cache,
            model_name=player_o_model,
            game=ep.game,
            initial_rating=config.initial_rating,
        )

        rating_a_pre = float(row_a.rating)
        rating_b_pre = float(row_b.rating)
        expected_a = _expected_score(rating_a_pre, rating_b_pre)
        score_a = _winner_to_score_a(ep.winner, player_x_model, player_o_model)
        score_b = 1.0 - score_a

        k_a = _k_factor(int(row_a.games_played), config)
        k_b = _k_factor(int(row_b.games_played), config)

        row_a.rating = rating_a_pre + k_a * (score_a - expected_a)
        row_b.rating = rating_b_pre + k_b * (score_b - (1.0 - expected_a))

        _update_record_counts(row_a, score=score_a, as_x=True)
        _update_record_counts(row_b, score=score_b, as_x=False)

        now = datetime.utcnow()
        row_a.updated_at = now
        row_b.updated_at = now

        winner_model = _normalize_model_name(ep.winner)
        if winner_model not in {player_x_model, player_o_model}:
            winner_model = DRAW_LABEL

        event = RatingEvent(
            event_id=str(uuid.uuid4()),
            run_id=ep.run_id,
            episode_id=ep.episode_id,
            game=ep.game,
            model_a=player_x_model,
            model_b=player_o_model,
            player_x_model=player_x_model,
            player_o_model=player_o_model,
            winner_model=winner_model,
            score_a=score_a,
            expected_a=expected_a,
            k_factor_a=k_a,
            k_factor_b=k_b,
            model_a_pre=rating_a_pre,
            model_b_pre=rating_b_pre,
            model_a_post=float(row_a.rating),
            model_b_post=float(row_b.rating),
            created_at=now,
        )
        session.add(event)
        processed += 1

    session.commit()
    return processed, skipped


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _print_leaderboard(session, *, game: str | None, limit: int) -> list[ModelRating]:
    query = session.query(ModelRating)
    if game is not None:
        query = query.filter(ModelRating.game == game)

    rows = (
        query.order_by(
            ModelRating.game.asc(),
            ModelRating.rating.desc(),
            ModelRating.games_played.desc(),
        )
        .limit(limit)
        .all()
    )

    if not rows:
        print("[RATINGS] No ratings found.")
        return []

    print("[RATINGS] Leaderboard")
    print(
        "{:<14} {:<24} {:>8} {:>7} {:>8} {:>10} {:>10}".format(
            "Game", "Model", "Rating", "Games", "Win%", "X Win%", "O Win%"
        )
    )
    print("-" * 90)

    for row in rows:
        win_rate = row.wins / row.games_played if row.games_played else None
        x_win_rate = row.wins_as_x / row.games_as_x if row.games_as_x else None
        o_win_rate = row.wins_as_o / row.games_as_o if row.games_as_o else None

        print(
            "{:<14} {:<24} {:>8.2f} {:>7} {:>8} {:>10} {:>10}".format(
                row.game,
                row.model_name,
                float(row.rating),
                int(row.games_played),
                _format_pct(win_rate),
                _format_pct(x_win_rate),
                _format_pct(o_win_rate),
            )
        )

    return rows


def _write_summary_json(rows: list[ModelRating], path: str) -> None:
    payload = []
    for row in rows:
        payload.append(
            {
                "game": row.game,
                "model": row.model_name,
                "rating": float(row.rating),
                "games": int(row.games_played),
                "wins": int(row.wins),
                "draws": int(row.draws),
                "losses": int(row.losses),
                "win_rate": (row.wins / row.games_played) if row.games_played else None,
                "x_win_rate": (row.wins_as_x / row.games_as_x)
                if row.games_as_x
                else None,
                "o_win_rate": (row.wins_as_o / row.games_as_o)
                if row.games_as_o
                else None,
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[RATINGS] Wrote summary JSON: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Elo ratings from competitive (PVP/PVP_RR) episode data"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-id", help="Incrementally rate one PVP or PVP_RR run")
    mode.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Rebuild ratings from all competitive (PVP/PVP_RR) episodes",
    )

    parser.add_argument(
        "--game", default=None, help="Optional game filter (e.g., connect_four)"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include runs not marked COMPLETED",
    )

    parser.add_argument("--initial-rating", type=float, default=1500.0)
    parser.add_argument("--k-provisional", type=float, default=32.0)
    parser.add_argument("--k-established", type=float, default=16.0)
    parser.add_argument("--provisional-games", type=int, default=30)

    parser.add_argument("--leaderboard-limit", type=int, default=200)
    parser.add_argument(
        "--summary-json", default=None, help="Optional output file for leaderboard JSON"
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # Ensure new ratings tables exist before querying them.
    create_tables()

    config = EloConfig(
        initial_rating=args.initial_rating,
        k_provisional=args.k_provisional,
        k_established=args.k_established,
        provisional_games=args.provisional_games,
    )

    session = get_session()
    try:
        if args.run_id:
            _validate_run(session, args.run_id, args.allow_incomplete)

        if args.rebuild_all:
            deleted_events = session.query(RatingEvent).delete()
            deleted_ratings = session.query(ModelRating).delete()
            session.commit()
            print(
                f"[RATINGS] Rebuild mode: cleared {deleted_events} events and {deleted_ratings} rating rows"
            )

        episodes = _fetch_target_episodes(
            session,
            run_id=args.run_id,
            game=args.game,
            allow_incomplete=args.allow_incomplete,
        )

        if not episodes:
            print("[RATINGS] No qualifying episodes found.")
            return

        processed, skipped = _process_episodes(
            session,
            episodes,
            config=config,
            skip_existing_events=not args.rebuild_all,
        )
        print(f"[RATINGS] Processed {processed} episodes, skipped {skipped}")

        rows = _print_leaderboard(session, game=args.game, limit=args.leaderboard_limit)
        if args.summary_json and rows:
            _write_summary_json(rows, args.summary_json)
    finally:
        session.close()


if __name__ == "__main__":
    main()
