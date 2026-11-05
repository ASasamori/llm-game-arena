"""Bounded, source-backed run reports and a terminal-safe ASCII presentation.

Decision text is a record of what a model or policy reported. It is not access
to a model's hidden reasoning. JSON transition pages preserve original stored
payloads; only the terminal presentation removes terminal control sequences.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import re
import textwrap
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from orchestrator.models import (
    Episode,
    EpisodeTrace,
    EpisodeTransition,
    Run,
    RunConfiguration,
)


class ReportNotFound(ValueError):
    """The requested run or an episode within that run does not exist."""


class ReportRequestError(ValueError):
    """A requested report page or rendering option is invalid."""


_RUNTIME_KEYS = (
    "schema_version",
    "prompt_version",
    "parser_version",
    "open_spiel_version",
    "python_version",
    "source_revision",
    "source_sha256",
    "dependency_lock_sha256",
)
_CONFIG_KEYS = (
    "game",
    "run_type",
    "num_episodes",
    "num_steps",
    "model_names",
    "seed",
    "rotate_seats",
    "failure_policy",
    "mean_field_population",
)
_PROVENANCE_KEYS = (
    *_RUNTIME_KEYS,
    "game",
    "max_steps",
    "failure_policy",
    "initial_state_hash",
    "mean_field_population",
    "environment_backend",
    "stop_after_quests",
    "recorded_features",
)
_TRACE_KEYS = (
    "game",
    "game_spec",
    "agent",
    "player_agents",
    "arena_type",
    "status",
    "winner",
    "winner_players",
    "stop_reason",
    "stop_message",
    "game_progress",
    "returns",
    "terminal",
    "seed",
    "num_steps",
    "decision_count",
    "pending_decision_count",
    "pending_fallback_count",
    "failed_attempt_count",
    "fallback_count",
    "degraded",
    "error",
    "final_state_hash",
)
_EPISODE_COLUMNS = (
    Episode.episode_id,
    Episode.game,
    Episode.agent,
    Episode.status,
    Episode.num_steps,
    Episode.winner,
    Episode.start_time,
    Episode.end_time,
)


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC).isoformat()


def _page(limit: int, offset: int, maximum: int, name: str) -> None:
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ReportRequestError(
            f"{name}_limit must be an integer between 1 and {maximum}"
        )
    if type(offset) is not int or offset < 0:
        raise ReportRequestError(f"{name}_offset must be a nonnegative integer")


def _runtime_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in _RUNTIME_KEYS if key in value}
    versions = value.get("dependency_versions")
    if isinstance(versions, dict):
        result["dependency_versions"] = {
            name: versions[name]
            for name in ("openai", "anthropic", "google-genai", "numpy")
            if name in versions
        }
    return result


def _episode_summary(row: Any) -> dict[str, Any]:
    result = {
        key: row[key]
        for key in ("episode_id", "game", "agent", "status", "num_steps", "winner")
    }
    result.update(
        start_time=_iso_utc(row["start_time"]), end_time=_iso_utc(row["end_time"])
    )
    return result


def _configuration(session: Session, run_id: str) -> dict[str, Any] | None:
    payload = RunConfiguration.payload
    row = (
        session.execute(
            select(
                RunConfiguration.run_id,
                *[payload[key].label(key) for key in _CONFIG_KEYS],
                payload["runtime"].label("runtime"),
            )
            .where(RunConfiguration.run_id == run_id)
            .limit(1)
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    result = {key: row[key] for key in _CONFIG_KEYS}
    result["runtime"] = _runtime_projection(row["runtime"])
    if result["rotate_seats"] in (0, 1):
        result["rotate_seats"] = bool(result["rotate_seats"])
    return result


def _trace_metadata(session: Session, episode_id: str) -> dict[str, Any] | None:
    payload = EpisodeTrace.payload
    manifest = payload["manifest"]
    row = (
        session.execute(
            select(
                EpisodeTrace.episode_id,
                *[payload[key].label(key) for key in _TRACE_KEYS],
                *[manifest[key].label("manifest_" + key) for key in _PROVENANCE_KEYS],
                manifest["model_configs"].label("model_configs"),
                manifest["dependency_versions"].label("dependency_versions"),
                manifest["game_sources"].label("game_sources"),
                manifest["trial_budget"].label("trial_budget"),
                *[
                    manifest["metadata"][key].label("metadata_" + key)
                    for key in ("episode_num", "matchup_index", "run_seed")
                ],
            )
            .where(EpisodeTrace.episode_id == episode_id)
            .limit(1)
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    result = {key: row[key] for key in _TRACE_KEYS}
    provenance = {
        key: row["manifest_" + key]
        for key in _PROVENANCE_KEYS
        if row["manifest_" + key] is not None
    }
    budget = row["trial_budget"]
    if isinstance(budget, dict):
        provenance["trial_budget"] = {
            key: budget[key]
            for key in (
                "budget_usd",
                "confirmed_cost_usd",
                "reserved_uncertain_usd",
                "accounted_cost_usd",
                "requests",
                "max_calls",
                "blocked",
                "pricing_source",
            )
            if key in budget
        }
    configs = row["model_configs"]
    provenance["model_configs"] = (
        [
            {
                key: entry[key]
                for key in ("identity", "model_name", "mock")
                if key in entry
            }
            for entry in configs
            if isinstance(entry, dict)
        ]
        if isinstance(configs, list)
        else []
    )
    provenance.update(
        {
            key: row["metadata_" + key]
            for key in ("episode_num", "matchup_index", "run_seed")
            if row["metadata_" + key] is not None
        }
    )
    if result["final_state_hash"] is not None:
        provenance["final_state_hash"] = result["final_state_hash"]
    versions = row["dependency_versions"]
    if isinstance(versions, dict):
        provenance["dependency_versions"] = {
            name: versions[name]
            for name in ("openai", "anthropic", "google-genai", "numpy")
            if name in versions
        }
    sources = row["game_sources"]
    if isinstance(sources, list):
        provenance["game_sources"] = [
            {
                key: source[key]
                for key in ("parameter_path", "sha256", "size_bytes", "format")
                if key in source
            }
            for source in sources
            if isinstance(source, dict)
        ]
    for key in ("terminal", "degraded"):
        if result[key] in (0, 1):
            result[key] = bool(result[key])
    result["provenance"] = provenance
    return result


def _transition_index(
    session: Session, episode_id: str, total: int, game: str
) -> dict[str, Any]:
    """Read small navigation fields, never prompts, responses or private views.

    The overview is capped independently of transition payload pages. Numeric
    jumps still address every transition in episodes larger than this overview.
    """
    payload = EpisodeTransition.payload
    board = payload["public_state_before"]
    node = payload["review_state_before"]["node"]
    rows = session.execute(
        select(
            EpisodeTransition.step_index,
            payload["kind"].label("kind"),
            board["phase"].label("phase"),
            board["quest_number"].label("quest"),
            board["turn"].label("turn"),
            board["turn_number"].label("turn_number"),
            payload["action_strings"].label("actions"),
            payload["players"].label("players"),
            node["kind"].label("node_kind"),
            node["move_number"].label("move_number"),
            node["active_players"].label("actors"),
        )
        .where(EpisodeTransition.episode_id == episode_id)
        .order_by(EpisodeTransition.step_index)
        .limit(5000)
    ).mappings()
    items = []
    for row in rows:
        actions = row["actions"]
        label = (
            ", ".join(str(value) for value in actions)
            if isinstance(actions, list)
            else ""
        )
        phase = row["phase"]
        # Older Avalon traces predate public snapshots. Labels identify only
        # the public phase; role assignments and individual ballots stay out.
        if phase is None and game.split("(", 1)[0] == "avalon":
            first = str(actions[0]) if isinstance(actions, list) and actions else ""
            phase = next(
                (
                    name
                    for name, prefixes in (
                        ("deal", ("deal_role:",)),
                        ("leader", ("initial_leader:",)),
                        ("proposal", ("propose_team:",)),
                        ("vote", ("approve_team", "reject_team")),
                        ("quest", ("quest_success", "quest_fail")),
                        ("assassinate", ("assassinate_player:",)),
                        (
                            "discussion",
                            ("claim_", "accuse_player:", "trust_player:", "pass"),
                        ),
                    )
                    if first.startswith(prefixes)
                ),
                None,
            )
        # Labels can contain private chance outcomes or ballots. Navigation
        # names the phase only for hidden-information native games.
        if row["kind"] == "chance":
            label = "Chance event"
        elif phase in {"deal", "quest", "vote"}:
            label = str(phase).replace("_", " ").title()
        items.append(
            {
                "index": row["step_index"],
                "kind": row["kind"],
                "phase": phase,
                "quest": row["quest"],
                "turn": row["turn_number"]
                if row["turn_number"] is not None
                else row["turn"],
                "label": label[:200],
                "node_kind": row["node_kind"] or row["kind"],
                "move_number": row["move_number"],
                "actors": row["actors"]
                if isinstance(row["actors"], list)
                else [
                    player
                    for player in (row["players"] or [])
                    if type(player) is int and player >= 0
                ],
            }
        )
    return {"items": items, "total": total, "truncated": total > len(items)}


def _pending_transition(
    session: Session, episode_id: str, limit: int, offset: int
) -> dict[str, Any] | None:
    """Project a bounded page of attempts that were never applied to the game."""
    pending = EpisodeTrace.payload["pending_transition"]
    fields = (
        "index",
        "kind",
        "applied",
        "state_hash_before",
        "failure",
        "review_state_before",
        "public_state_before",
        "state_before",
        "failed_attempt",
        "decision_count",
    )
    row = (
        session.execute(
            select(
                *[pending[key].label(key) for key in fields],
                *[
                    pending["decisions"][index].label(f"decision_{index}")
                    for index in range(offset, offset + limit)
                ],
            )
            .where(EpisodeTrace.episode_id == episode_id)
            .limit(1)
        )
        .mappings()
        .first()
    )
    if row is None or row["index"] is None:
        return None
    result = {key: row[key] for key in fields if row[key] is not None}
    result["applied"] = False
    result["decisions"] = {
        "items": [
            row[f"decision_{index}"]
            for index in range(offset, offset + limit)
            if row[f"decision_{index}"] is not None
        ],
        "total": int(row["decision_count"] or 0),
        "limit": limit,
        "offset": offset,
    }
    return result


def build_run_report(
    session: Session,
    run_id: str,
    *,
    episode_id: str | None = None,
    episode_limit: int = 20,
    episode_offset: int = 0,
    step_limit: int = 20,
    step_offset: int = 0,
    pending_limit: int = 20,
    pending_offset: int = 0,
) -> dict[str, Any]:
    """Read bounded pages without loading episode action/summary arrays.

    Every selected-episode query is constrained to the requested run. A legacy
    episode without a trace remains inspectable, with unavailable facts marked
    unknown rather than reconstructed from turn parity or action labels.
    """
    _page(episode_limit, episode_offset, 100, "episode")
    _page(step_limit, step_offset, 200, "step")
    _page(pending_limit, pending_offset, 200, "pending")
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise ReportRequestError("episode_id must be a nonempty string")
    run = (
        session.execute(
            select(
                Run.run_id,
                Run.status,
                Run.run_type,
                Run.num_episodes,
                Run.created_at,
            )
            .where(Run.run_id == run_id)
            .limit(1)
        )
        .mappings()
        .first()
    )
    if run is None:
        raise ReportNotFound(f"Run not found: {run_id}")
    status_counts = {
        str(status): int(count)
        for status, count in session.execute(
            select(Episode.status, func.count(Episode.episode_id))
            .where(Episode.run_id == run_id)
            .group_by(Episode.status)
        )
    }
    total = sum(status_counts.values())
    configuration = _configuration(session, run_id)
    rows = (
        session.execute(
            select(*_EPISODE_COLUMNS)
            .where(Episode.run_id == run_id)
            .order_by(Episode.start_time, Episode.episode_id)
            .offset(episode_offset)
            .limit(episode_limit)
        )
        .mappings()
        .all()
    )
    episodes = [_episode_summary(row) for row in rows]
    selected_id = (
        episode_id
        if episode_id is not None
        else episodes[0]["episode_id"]
        if episodes
        else None
    )
    selected: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = []
    transition_total = 0
    pending_transition = None
    transition_index: dict[str, Any] = {"items": [], "total": 0, "truncated": False}
    notices = [
        "Model summaries and raw responses are recorded output, not access to hidden reasoning."
    ]
    if selected_id is not None:
        row = (
            session.execute(
                select(*_EPISODE_COLUMNS, Episode.error)
                .where(Episode.run_id == run_id, Episode.episode_id == selected_id)
                .limit(1)
            )
            .mappings()
            .first()
        )
        if row is None:
            raise ReportNotFound(f"Episode not found in run {run_id}: {selected_id}")
        selected = _episode_summary(row)
        trace = _trace_metadata(session, selected_id)
        selected.update(
            game_spec=None,
            player_agents=None,
            returns=None,
            terminal=None,
            seed=None,
            decision_count=None,
            fallback_count=None,
            degraded=None,
            error=row["error"],
            provenance={},
            trace_available=trace is not None,
            mock=None,
            arena_type=None,
        )
        if trace is not None:
            pending_transition = _pending_transition(
                session, selected_id, pending_limit, pending_offset
            )
            for key in (
                "game",
                "agent",
                "player_agents",
                "arena_type",
                "status",
                "winner",
                "winner_players",
                "stop_reason",
                "stop_message",
                "game_progress",
                "returns",
                "terminal",
                "seed",
                "num_steps",
                "decision_count",
                "pending_decision_count",
                "pending_fallback_count",
                "failed_attempt_count",
                "fallback_count",
                "degraded",
                "error",
                "provenance",
            ):
                if trace[key] is not None:
                    selected[key] = trace[key]
            selected["game_spec"] = (
                trace["game_spec"]
                or trace["provenance"].get("game")
                or (configuration or {}).get("game")
            )
            flags = [
                entry["mock"]
                for entry in selected["provenance"].get("model_configs", [])
                if type(entry.get("mock")) is bool
            ]
            selected["mock"] = (
                True
                if any(flags)
                else False
                if flags
                and len(flags) == len(selected["provenance"].get("model_configs", []))
                else None
            )
            transition_total = session.execute(
                select(func.count(EpisodeTransition.step_index)).where(
                    EpisodeTransition.episode_id == selected_id
                )
            ).scalar_one()
            transitions = list(
                session.execute(
                    select(EpisodeTransition.payload)
                    .where(EpisodeTransition.episode_id == selected_id)
                    .order_by(EpisodeTransition.step_index)
                    .offset(step_offset)
                    .limit(step_limit)
                ).scalars()
            )
            transition_index = _transition_index(
                session, selected_id, transition_total, selected["game"]
            )
            if transition_total == 0:
                notices.append("No transitions were recorded for the selected episode.")
            elif not any(
                any(
                    key in item
                    for key in (
                        "state_before",
                        "state_after",
                        "public_state_before",
                        "public_state_after",
                        "review_state_before",
                        "review_state_after",
                    )
                )
                for item in transitions
            ):
                notices.append(
                    "Evaluator board snapshots were not recorded on this page; player observations are shown exactly as stored."
                )
            if pending_transition is not None:
                notices.append(
                    "Interrupted attempt: recorded choices and failed attempts below were not applied to the game. Their model output and usage are retained separately from completed transitions."
                )
        else:
            selected["game_spec"] = (configuration or {}).get("game")
            notices.append(
                "Legacy episode: no detailed trace is available; player seats, model responses, utilities, and mock/live mode cannot be reconstructed reliably."
            )
        if selected["mock"] is True:
            notices.append(
                "MOCK OUTPUT: at least one model used a mock policy. These results do not measure real model performance."
            )
        elif selected["mock"] is None:
            notices.append(
                "The selected episode's mock/live model mode was not recorded."
            )
        if selected["terminal"] is False or selected["status"] in {
            "TRUNCATED",
            "FAILED",
            "PENDING",
        }:
            notices.append(
                "The selected episode did not record a successful terminal outcome; intermediate returns are not a final score."
            )
        if selected["degraded"] or (selected["fallback_count"] or 0) > 0:
            notices.append(
                "Fallback-assisted episode: random legal actions replaced model failures. Its returns are excluded from clean model metrics and Elo."
            )
    else:
        notices.append(
            "No episodes are available on this page. The run may still be pending, or episode_offset is past the last saved episode."
        )
    return {
        "schema_version": 1,
        "run": {
            "run_id": run["run_id"],
            "status": run["status"],
            "run_type": run["run_type"],
            "num_episodes": run["num_episodes"],
            "episodes_saved": total,
            "created_at": _iso_utc(run["created_at"]),
            "configuration": configuration,
            "status_counts": status_counts,
        },
        "episodes": {
            "items": episodes,
            "total": total,
            "limit": episode_limit,
            "offset": episode_offset,
        },
        "selected_episode": selected,
        "transitions": {
            "items": transitions,
            "total": int(transition_total),
            "limit": step_limit,
            "offset": step_offset,
        },
        "transition_index": transition_index,
        "pending_transition": pending_transition,
        "notices": notices,
    }


# OSC includes clipboard operations and hyperlink targets. Drop the whole
# sequence, including unterminated sequences, before stripping other controls.
_STRING_CONTROL = re.compile(
    r"(?:\x1b\]|\x9d|\x1b[P^_X]|[\x90\x98\x9e\x9f]).*?(?:\x07|\x1b\\|\x9c|$)", re.DOTALL
)
_CSI_CONTROL = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_ESCAPE_CONTROL = re.compile(r"\x1b[ -/]*[0-~]")
_OTHER_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def terminal_safe(value: Any) -> str:
    """Return printable ASCII plus newlines; never emit terminal instructions."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = _STRING_CONTROL.sub("", text)
    text = _CSI_CONTROL.sub("", text)
    text = _ESCAPE_CONTROL.sub("", text)
    text = _OTHER_CONTROL.sub("", text).expandtabs(4)
    return text.encode("ascii", "backslashreplace").decode("ascii")


class _Text:
    def __init__(self, width: int):
        self.width = width
        self.lines: list[str] = []

    def line(self, value: Any = "", indent: str = "") -> None:
        text = terminal_safe(value)
        for line in text.split("\n"):
            self.lines.extend(
                textwrap.wrap(
                    line,
                    width=max(1, self.width - len(indent)),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
                or [indent]
            )

    def field(self, label: str, value: Any) -> None:
        lines = ("Not recorded" if value is None else terminal_safe(value)).split("\n")
        self.line(label + ": " + lines[0])
        for line in lines[1:]:
            self.line(line, "  ")

    def heading(self, value: str) -> None:
        self.line()
        self.line(value)
        self.line("-" * self.width)


def _page_label(page: dict[str, Any]) -> str:
    count = len(page["items"])
    start = page["offset"] + 1 if count else 0
    return f"{start}-{page['offset'] + count if count else 0} of {page['total']} (limit={page['limit']}, offset={page['offset']})"


def render_text_report(
    report: dict[str, Any], *, width: int = 100, include_prompt: bool = False
) -> str:
    """Render the selected report page, preserving reported text as inert ASCII."""
    if type(width) is not int or not 40 <= width <= 240:
        raise ReportRequestError("width must be an integer between 40 and 240")
    text = _Text(width)
    run = report["run"]
    text.line("OPEN SPIEL ARENA - RUN REPORT")
    text.line("=" * width)
    for key in (
        "run_id",
        "status",
        "run_type",
        "num_episodes",
        "episodes_saved",
        "created_at",
        "status_counts",
    ):
        text.field(key.replace("_", " ").title(), run.get(key))
    configuration = run.get("configuration") or {}
    for key in ("game", "seed", "num_steps", "failure_policy", "rotate_seats"):
        if key in configuration:
            text.field("Configured " + key.replace("_", " "), configuration[key])
    text.heading("NOTICES")
    for notice in report["notices"]:
        text.line("* " + terminal_safe(notice))
    text.heading("EPISODES: " + _page_label(report["episodes"]))
    selected = report.get("selected_episode")
    for episode in report["episodes"]["items"]:
        marker = (
            "*" if selected and selected["episode_id"] == episode["episode_id"] else " "
        )
        text.line(
            f"{marker} {episode['episode_id']} | {episode['game']} | {episode['status']} | {episode['num_steps']} transitions"
        )
        text.field("  Recorded agents", episode["agent"])
    next_episode = report["episodes"]["offset"] + len(report["episodes"]["items"])
    if next_episode < report["episodes"]["total"]:
        text.line(
            f"Next episodes: episode_offset={next_episode}&episode_limit={report['episodes']['limit']}"
        )
    if selected is None:
        return "\n".join(text.lines) + "\n"
    text.heading("SELECTED EPISODE")
    for key in (
        "episode_id",
        "game",
        "game_spec",
        "status",
        "winner",
        "winner_players",
        "stop_reason",
        "stop_message",
        "game_progress",
        "terminal",
        "returns",
        "seed",
        "player_agents",
        "num_steps",
        "decision_count",
        "fallback_count",
        "degraded",
        "trace_available",
        "mock",
        "error",
    ):
        label = (
            "Recorded winner"
            if key == "winner"
            else "Recorded utilities / returns"
            if key == "returns"
            else key.replace("_", " ").title()
        )
        text.field(label, selected.get(key))
    text.heading("PROVENANCE")
    for key, value in selected["provenance"].items():
        text.field(key, value)
    if not selected["provenance"]:
        text.line("No detailed provenance recorded.")
    text.heading("TRANSITIONS: " + _page_label(report["transitions"]))
    for transition in report["transitions"]["items"]:
        text.heading(
            f"Transition {transition.get('index', '?')} | {transition.get('kind', 'unknown')}"
        )
        if "state_before" in transition:
            text.line(
                "Evaluator state before (separate from the player's observation):"
            )
            text.line(transition["state_before"], "  ")
        if "public_state_before" in transition:
            text.line("Public board before:")
            text.line(transition["public_state_before"], "  ")
        if "review_state_before" in transition:
            review = transition["review_state_before"]
            text.field("Game state before", review["node"])
            text.field("Public state visibility", review["visibility"])
            if (
                "state_before" not in transition
                and review["board"].get("text") is not None
            ):
                text.line(review["board"]["text"], "  ")
        for key in (
            "players",
            "actions",
            "action_strings",
            "rewards",
            "returns",
            "terminal",
            "chance_outcomes",
            "distribution_support",
            "distribution",
            "state_hash_before",
            "state_hash",
        ):
            if key in transition:
                text.field(key.replace("_", " ").title(), transition[key])
        if transition.get("kind") == "mean_field":
            text.line(
                "Recorded exogenous population update; this does not establish an equilibrium."
            )
        for decision in transition.get("decisions", []):
            metadata = decision.get("response_metadata") or {}
            classifier = metadata.get("summary_source") == "adapter"
            text.line()
            text.field("Recorded player ID", decision.get("player"))
            text.field("Recorded actor", decision.get("agent"))
            if "selection" in decision:
                text.field("Selection mechanism", decision["selection"])
            text.field("Chosen action ID", decision.get("action"))
            text.field("Chosen action label", decision.get("action_string"))
            text.field("Legal action IDs", decision.get("legal_actions"))
            if "legal_action_strings" in decision:
                text.field("Legal action labels", decision["legal_action_strings"])
            text.field("Observation type", decision.get("observation_type"))
            text.line("Observation (recorded before action):")
            text.line(
                "Not recorded"
                if decision.get("observation") is None
                else decision["observation"],
                "  ",
            )
            label = (
                "Recorded fallback summary"
                if decision.get("fallback_reason")
                else "Adapter decision summary"
                if classifier
                else "Recorded mock policy summary"
                if decision.get("selection", {}).get("kind") == "mock"
                else "Reported model summary"
                if decision.get("response") is not None
                else "Recorded policy summary"
            )
            text.field(label, decision.get("summary"))
            raw_response = (
                metadata.get("provider_response")
                if classifier
                else decision.get("response")
            )
            text.line(
                "Raw classifier response:" if classifier else "Raw recorded response:"
            )
            text.line(
                "Not recorded"
                if raw_response is None
                else json.dumps(raw_response, ensure_ascii=False, indent=2)
                if isinstance(raw_response, dict)
                else raw_response,
                "  ",
            )
            if classifier:
                text.field(
                    "Decision source", metadata.get("decision_source", "classifier")
                )
                text.field("Classifier confidence", metadata.get("confidence"))
                text.field("Classifier probabilities", metadata.get("probabilities"))
            if decision.get("fallback_reason"):
                text.field("Fallback / error", decision["fallback_reason"])
            for key in ("usage", "latency_ms"):
                if key in decision:
                    text.field(key.replace("_", " ").title(), decision[key])
            if include_prompt:
                text.line("Raw recorded prompt:")
                text.line(
                    "Not recorded"
                    if decision.get("prompt") is None
                    else decision["prompt"],
                    "  ",
                )
        if "state_after" in transition:
            text.line("Evaluator state after:")
            text.line(transition["state_after"], "  ")
        if "public_state_after" in transition:
            text.line("Public board after:")
            text.line(transition["public_state_after"], "  ")
        if "review_state_after" in transition:
            review = transition["review_state_after"]
            text.field("Game state after", review["node"])
            text.field("Public state visibility", review["visibility"])
            if (
                "state_after" not in transition
                and review["board"].get("text") is not None
            ):
                text.line(review["board"]["text"], "  ")
    pending = report.get("pending_transition")
    if pending is not None:
        text.heading("INTERRUPTED ATTEMPTS - NOT APPLIED")
        text.line(
            "The game state did not advance. These attempts are separate from applied decisions and returns."
        )
        text.field("Attempted transition", pending.get("index"))
        text.field("Node kind", pending.get("kind"))
        text.field("Unapplied choices", _page_label(pending["decisions"]))
        attempts = [
            ("Selected but not applied", decision)
            for decision in pending["decisions"]["items"]
        ]
        if pending.get("failed_attempt") is not None:
            attempts.append(
                ("Failed attempt; no action applied", pending["failed_attempt"])
            )
        for label, decision in attempts:
            text.line()
            text.line(label)
            for key in (
                "player",
                "agent",
                "action",
                "action_string",
                "selection",
                "observation_type",
                "observation",
                "legal_actions",
                "legal_action_strings",
                "summary",
                "response",
                "response_metadata",
                "usage",
                "latency_ms",
                "failure",
            ):
                if key in decision:
                    text.field(key.replace("_", " ").title(), decision[key])
            if include_prompt:
                text.field("Raw recorded prompt", decision.get("prompt"))
        page = pending["decisions"]
        next_pending = page["offset"] + len(page["items"])
        if next_pending < page["total"]:
            text.line(
                f"More unapplied choices: pending_offset={next_pending}&pending_limit={page['limit']}"
            )
    if not include_prompt:
        text.line(
            "Prompts omitted from text view; the authenticated JSON page retains recorded prompts."
        )
    next_step = report["transitions"]["offset"] + len(report["transitions"]["items"])
    if next_step < report["transitions"]["total"]:
        text.line(
            f"Next transitions: episode_id={selected['episode_id']}&step_offset={next_step}&step_limit={report['transitions']['limit']}"
        )
    text.line(
        "Select an episode with episode_id; change episode_offset/step_offset to page through stored data."
    )
    return "\n".join(text.lines) + "\n"
