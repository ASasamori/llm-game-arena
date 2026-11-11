"""Seed the real run viewer with reproducible, explicitly scripted mock games.

This utility writes only to a new or empty local output directory. It creates no
provider client, does not connect to Redis, and never uses an ambient database.
Start the normal API separately using the SQLite URL in manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence, TYPE_CHECKING
from unittest.mock import patch
import uuid

if TYPE_CHECKING:
    from game_arena.openspiel.model_generation import GenerateResponse

PROJECT = Path(__file__).resolve().parents[1]
NAMESPACE = uuid.UUID("71175ed1-3eb5-58be-973c-326ac8938aa3")
DEMO_SEED = 123
PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "TYPESAFE_API_KEY",
    "JEV_API_KEY",
)
SPECS = (
    ("connect_four", "connect_four", 2, None),
    ("avalon", "avalon(players=6,discussion_rounds=1)", 6, 3),
    ("catan", "catan(players=4,max_turns=12,max_actions_per_turn=20)", 4, None),
)


class DemoPolicy:
    """A transparent fixed policy using only the supplied player's observation."""

    _mock_mode = True

    def __init__(self, game: str, seat: int):
        self.game, self.seat = game, seat
        self._model_name = f"demo-{game}-seat-{seat}"
        self.trade_turns: set[int] = set()

    def __str__(self) -> str:
        return "MOCK:" + self._model_name

    def generate_with_text_input(
        self, prompt: str, system_instruction: str | None = None, mock: bool = False
    ) -> GenerateResponse:
        from game_arena.openspiel.model_generation import GenerateResponse

        context = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])[0]
        actions = {item["id"]: item["label"] for item in context["legal_actions"]}
        if self.game == "connect_four":
            action = 3 if self.seat == 0 else 2
            description = (
                "repeat the preset center-column move"
                if self.seat == 0
                else "repeat the preset adjacent-column move"
            )
        else:
            view = json.loads(context["observation"])
            if self.game == "avalon":
                action, description = self._avalon(view, actions)
            else:
                action, description = self._catan(view, actions)
        if action not in actions:
            raise ValueError("Scripted demo policy selected an unavailable action")
        return GenerateResponse(
            main_response=(
                f"Summary: Mock script: {description}. This is a scripted policy, "
                f"not a model's reasoning.\nFinal Answer: {action}"
            ),
            usage={"input_tokens": 0, "output_tokens": 0},
            metadata={
                "mock": True,
                "provider": "none",
                "provider_requests": 0,
                "summary_source": "adapter",
                "decision_source": "scripted_demo",
            },
        )

    @staticmethod
    def _avalon(view: dict[str, Any], actions: dict[int, str]) -> tuple[int, str]:
        phase = view["phase"]
        if phase == "vote":
            return 1, "approve the proposed team so the demo advances"
        if phase == "discussion":
            return 1, "make the preset public claim of good alignment"
        if phase == "quest":
            sabotage = view["alignment"] == "evil" and view["quest_number"] != 2
            return int(sabotage), (
                "play the permitted failure card on this odd-numbered quest"
                if sabotage
                else "play a success card"
            )
        if phase == "assassinate":
            raise AssertionError("The demo must stop after its third quest")
        return next(iter(actions)), "propose the first listed legal team"

    def _catan(self, view: dict[str, Any], actions: dict[int, str]) -> tuple[int, str]:
        from game_arena.environments import catan

        phase = view["phase"]
        if phase == "setup_settlement":

            def score(action: int) -> int:
                vertex = action - catan.SETTLEMENT
                return sum(
                    6 - abs(7 - tile["number"])
                    for tile in view["board"]["tiles"]
                    if vertex in tile["vertices"] and tile["number"]
                )

            return max(
                actions, key=score
            ), "place at the legal site with most public number-token pips"
        if phase == "trade_partner":
            partners = [action for action in actions if action != catan.CANCEL]
            chosen = max(
                partners,
                key=lambda action: view["players"][action - catan.PARTNER][
                    "resource_count"
                ],
            )
            return (
                chosen,
                "offer a trade to a player with the largest public hand count",
            )
        if phase == "trade_draft":
            trade = view["trade"]
            if not sum(trade["give"].values()):
                candidates = [
                    a for a in actions if catan.GIVE_ADD <= a < catan.GIVE_ADD + 5
                ]
                chosen = max(
                    candidates,
                    key=lambda action: view["resources"][
                        catan.RESOURCES[action - catan.GIVE_ADD]
                    ],
                )
                return (
                    chosen,
                    "offer one resource from the largest stack in this player's own hand",
                )
            if not sum(trade["receive"].values()):
                candidates = [
                    a for a in actions if catan.RECEIVE_ADD <= a < catan.RECEIVE_ADD + 5
                ]
                chosen = min(
                    candidates,
                    key=lambda action: view["resources"][
                        catan.RESOURCES[action - catan.RECEIVE_ADD]
                    ],
                )
                return chosen, "request one different resource this player has less of"
            return (
                catan.SEND,
                "send the one-for-one offer for the recipient to accept or reject",
            )
        if phase == "trade_response":
            if catan.ACCEPT in actions:
                return (
                    catan.ACCEPT,
                    "accept the affordable offer under the preset demo policy",
                )
            return (
                catan.REJECT,
                "reject the offer because acceptance is not a legal action",
            )
        turn = view["turn_number"]
        if phase == "main" and catan.TRADE in actions and turn not in self.trade_turns:
            self.trade_turns.add(turn)
            return catan.TRADE, "start this turn's single scripted domestic trade offer"
        if phase == "main":
            for prefix in ("Upgrade", "Build settlement", "Build road"):
                available = [
                    action
                    for action, label in actions.items()
                    if label.startswith(prefix)
                ]
                if available:
                    return available[
                        0
                    ], "perform the first affordable build in the preset priority order"
            return catan.END, "end the turn after the demo's trade and build choices"
        return next(iter(actions)), "apply the first listed legal action for this phase"


def seed_demo(output: Path) -> dict[str, Any]:
    """Produce viewer-ready records using real rules and normal result persistence."""
    if output.is_symlink():
        raise ValueError("Demo output must not be a symlink")
    root = output.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(
            "Demo output must be a new or empty directory; existing data was left untouched"
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    database = root / "arena.sqlite"
    database_url = "sqlite+pysqlite:///" + str(database)
    environment = {
        "PYTHON_DOTENV_DISABLED": "1",
        "MODEL_API_MOCK_MODE": "true",
        "DATABASE_TARGET": "temporary_local",
        "DATABASE_URL": database_url,
        "WORKER_TRIGGER_MODE": "none",
        **{name: "" for name in PROVIDER_KEYS},
    }
    with patch.dict(os.environ, environment):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from game_arena.openspiel.harness import harness
        from game_arena.openspiel.model_generation import LLMModel
        from game_arena.openspiel.replay import replay_episode
        from game_arena.openspiel.utils import HarnessOptions
        from orchestrator import tasks
        from orchestrator.models import (
            Base,
            Run,
            RunConfiguration,
            RunDispatch,
            RunProgress,
        )

        engine = create_engine(database_url)
        try:
            Base.metadata.create_all(engine)
            sessions = sessionmaker(bind=engine)
            entries = []
            for name, specification, players, quest_limit in SPECS:
                run_id = str(uuid.uuid5(NAMESPACE, name + ":run"))
                episode_id = str(uuid.uuid5(NAMESPACE, name + ":episode"))
                policies: list[LLMModel] = [
                    DemoPolicy(name, seat) for seat in range(players)
                ]
                options = HarnessOptions(
                    type="PVP",
                    game=specification,
                    models=policies,
                    num_steps=1000,
                    input=None,
                    seed=DEMO_SEED,
                    episode_id=episode_id,
                    failure_policy="raise",
                    metadata={"demo": True, "mock_only": True, "provider_requests": 0},
                )
                if quest_limit is not None:
                    options["stop_after_quests"] = quest_limit
                result = harness(options)
                record: dict[str, Any] = dict(result)
                verified = replay_episode(record)
                if not verified["verified"]:
                    raise ValueError("Demo trajectory failed replay verification")
                with sessions() as session:
                    session.add(
                        Run(
                            run_id=run_id,
                            run_type="PVP",
                            num_episodes=1,
                            status="PENDING",
                        )
                    )
                    session.flush()
                    session.add(
                        RunConfiguration(
                            run_id=run_id,
                            payload={
                                "schema_version": 1,
                                "game": record["game_spec"],
                                "seed": DEMO_SEED,
                                "run_type": "PVP",
                                "model_names": [str(policy) for policy in policies],
                                "num_steps": 1000,
                                "stop_after_quests": quest_limit,
                                "failure_policy": "raise",
                                "runtime": record["manifest"],
                            },
                        )
                    )
                    session.add(RunDispatch(run_id=run_id, complete=True))
                    session.add(RunProgress(run_id=run_id, completed=0, failed=0))
                    session.commit()
                with patch.object(tasks, "get_session", sessions):
                    tasks._persist_result(result, run_id, None)
                if record.get("stop_reason") == "quest_limit":
                    with sessions() as session:
                        run = session.get(Run, run_id)
                        assert run is not None
                        run.status = "STOPPED"
                        session.commit()
                transitions = record["steps"]
                entries.append(
                    {
                        "name": name,
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "game": record["game_spec"],
                        "seed": DEMO_SEED,
                        "mock_only": True,
                        "provider_requests": 0,
                        "status": record["status"],
                        "terminal": record["terminal"],
                        "stop_reason": record.get("stop_reason"),
                        "transitions": record["num_steps"],
                        "decisions": record["decision_count"],
                        "report_path": f"/runs/{run_id}",
                        "report_data_path": f"/v1/runs/{run_id}/report",
                        "replay_verified": True,
                        "trade_offer_steps": [
                            event["index"]
                            for event in transitions
                            if "Send trade offer" in event["action_strings"]
                        ],
                        "accepted_trade_steps": [
                            event["index"]
                            for event in transitions
                            if "Accept trade offer" in event["action_strings"]
                        ],
                        "last_step": transitions[-1]["index"],
                    }
                )
        finally:
            engine.dispose()
    manifest = {
        "schema_version": 1,
        "description": "Scripted mock viewer demo. No actual model reasoning or provider requests.",
        "mock_only": True,
        "provider_requests": 0,
        "database_url": database_url,
        "runs": entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analytics/readme-demo"))
    args = parser.parse_args(argv)
    sys.path.insert(0, str(PROJECT))
    try:
        manifest = seed_demo(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
