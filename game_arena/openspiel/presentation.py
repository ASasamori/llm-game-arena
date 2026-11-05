"""Lossless spectator presentation for every OpenSpiel state contract.

The grid extractors follow the upstream ``ToString`` implementations under
https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/games.
They are conveniences layered over the unchanged engine display, never a new
game state or an observation supplied to a competitor. Unrecognized layouts
remain readable as exact text. Hidden-information states are never inspected.
"""

from __future__ import annotations

import re
from typing import Any

import pyspiel


_MAX_GRID_CELLS = 4096
_CROSS_LEGEND = {".": "Empty", "x": "Player 0", "o": "Player 1"}
_RANKED = {
    "checkers": ".o+8*",
    "breakthrough": ".bw",
    "clobber": ".ox",
    "lines_of_action": ".xo",
}
_RECTANGLES = {
    "tic_tac_toe": ".xo",
    "mnk": ".xo",
    "connect_four": ".xo",
    "catch": ".xo",
    "amazons": ".XO#",
    "pathfinding": ".*0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "cliff_walking": ".PXG",
    "markov_soccer": ".abABO",
    "stones_and_gems": ". @o*SHFCDEIXWQAB0123456789",
}


def _enum(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def _grid(
    rows: list[list[str]],
    *,
    row_labels: list[str] | None = None,
    column_labels: list[str] | None = None,
    layout: str = "square",
    empty_symbols: list[str] | None = None,
    legend: dict[str, str] | None = None,
    parser: str,
) -> dict[str, Any] | None:
    if (
        not rows
        or not rows[0]
        or any(len(row) != len(rows[0]) for row in rows)
        or len(rows) * len(rows[0]) > _MAX_GRID_CELLS
    ):
        return None
    return {
        "format": "grid",
        "rows": rows,
        "row_labels": row_labels or [str(index) for index in range(len(rows))],
        "column_labels": column_labels or [str(index) for index in range(len(rows[0]))],
        "layout": layout,
        "empty_symbols": empty_symbols if empty_symbols is not None else ["."],
        "legend": legend or {},
        "renderer": parser,
        "note": "Grid cells are extracted from the exact engine display retained below.",
    }


def _chess(text: str) -> dict[str, Any] | None:
    fields = text.split()
    if len(fields) != 6 or fields[1] not in {"w", "b"}:
        return None
    rows: list[list[str]] = []
    for rank in fields[0].split("/"):
        cells: list[str] = []
        for symbol in rank:
            if symbol in "12345678":
                cells.extend(["."] * int(symbol))
            elif symbol in "pnbrqkPNBRQK":
                cells.append(symbol)
            else:
                return None
        if len(cells) != 8:
            return None
        rows.append(cells)
    if len(rows) != 8:
        return None
    names = {
        "p": "Pawn",
        "n": "Knight",
        "b": "Bishop",
        "r": "Rook",
        "q": "Queen",
        "k": "King",
    }
    legend = {".": "Empty"}
    for symbol, name in names.items():
        # OpenSpiel's seat IDs differ from the conventional white-first order.
        legend[symbol] = "Black " + name + " (Player 0)"
        legend[symbol.upper()] = "White " + name + " (Player 1)"
    return _grid(
        rows,
        row_labels=list("87654321"),
        column_labels=list("abcdefgh"),
        layout="chess",
        legend=legend,
        parser="chess_fen",
    )


def _ranked(text: str, name: str) -> dict[str, Any] | None:
    rows: list[list[str]] = []
    labels: list[str] = []
    symbols = re.escape(_RANKED[name])
    lines = text.splitlines()
    columns = next(
        (
            list(line.strip())
            for line in reversed(lines)
            if re.fullmatch(r"\s+[a-z]+", line)
        ),
        [],
    )
    if not columns:
        return None
    for line in lines:
        match = re.fullmatch(
            r"\s*(\d+)([" + symbols + "]{" + str(len(columns)) + "})", line
        )
        if match:
            labels.append(match[1])
            rows.append(list(match[2]))
    if not rows or labels != [str(i) for i in range(len(rows), 0, -1)]:
        return None
    legends = {
        "checkers": {
            ".": "Empty",
            "o": "Player 0 man",
            "8": "Player 0 king",
            "+": "Player 1 man",
            "*": "Player 1 king",
        },
        "breakthrough": {".": "Empty", "b": "Player 0", "w": "Player 1"},
        "clobber": {".": "Empty", "o": "Player 0", "x": "Player 1"},
        "lines_of_action": _CROSS_LEGEND,
    }
    return _grid(
        rows,
        row_labels=labels,
        column_labels=columns,
        legend=legends[name],
        parser="ranked_" + name,
    )


def _go(text: str) -> dict[str, Any] | None:
    rows: list[list[str]] = []
    labels: list[str] = []
    columns: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+) ([+XO]+)", line)
        if match:
            labels.append(match[1])
            rows.append(list(match[2]))
        elif re.fullmatch(r"\s+[A-HJ-Z]+", line):
            columns = list(line.strip())
    if (
        not rows
        or len(columns) != len(rows[0])
        or labels != [str(i) for i in range(len(columns), 0, -1)]
    ):
        return None
    return _grid(
        rows,
        row_labels=labels,
        column_labels=columns,
        empty_symbols=["+"],
        legend={"+": "Empty", "X": "Black (Player 0)", "O": "White (Player 1)"},
        parser="go_coordinates",
    )


def _othello(text: str) -> dict[str, Any] | None:
    rows: list[list[str]] = []
    labels: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"(\d+) ((?:[-xo] )+)(\d+)", line)
        if match and match[1] == match[3]:
            labels.append(match[1])
            rows.append(match[2].split())
    if labels != list("12345678"):
        return None
    return _grid(
        rows,
        row_labels=labels,
        column_labels=list("abcdefgh"),
        empty_symbols=["-"],
        legend={"-": "Empty", "x": "Black (Player 0)", "o": "White (Player 1)"},
        parser="othello_coordinates",
    )


def _pentago(text: str) -> dict[str, Any] | None:
    rows: list[list[str]] = []
    labels: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r".{2}([1-6]) ((?:[.O@] ){5}[.O@])(?: .)?", line)
        if match:
            labels.append(match[1])
            rows.append(match[2].split())
    if labels != list("123456"):
        return None
    return _grid(
        rows,
        row_labels=labels,
        column_labels=list("abcdef"),
        legend={".": "Empty", "O": "Player 0", "@": "Player 1"},
        parser="pentago_quadrants",
    )


def _extract_grid(
    name: str, text: str, parameters: dict[str, Any]
) -> dict[str, Any] | None:
    if name in {"chess", "antichess"}:
        return _chess(text)
    if name in _RANKED:
        return _ranked(text, name)
    if name in {"go", "cursor_go"}:
        return _go(text)
    if name == "othello":
        return _othello(text)
    if name == "pentago":
        return _pentago(text)
    if name == "hex":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not all(
            re.fullmatch(r"[.xoXOpqyz](?: [.xoXOpqyz])*", line) for line in lines
        ):
            return None
        expected_rows = parameters.get("num_rows", parameters.get("board_size", 11))
        expected_columns = parameters.get("num_cols", parameters.get("board_size", 11))
        if len(lines) != expected_rows or any(
            len(line.split()) != expected_columns for line in lines
        ):
            return None
        return _grid(
            [line.split() for line in lines],
            column_labels=[chr(ord("a") + i) for i in range(len(lines[0].split()))]
            if lines
            else None,
            row_labels=[str(i + 1) for i in range(len(lines))],
            layout="hex",
            legend={
                ".": "Empty",
                "x": "Black (Player 0)",
                "o": "White (Player 1)",
                "X": "Black winning connection",
                "O": "White winning connection",
                "y": "Black connected north",
                "z": "Black connected south",
                "p": "White connected west",
                "q": "White connected east",
            },
            parser="hex_rows",
        )
    if name == "2048":
        lines = [line.split() for line in text.splitlines() if line.strip()]
        if not all(all(cell.isdigit() for cell in line) for line in lines):
            return None
        return _grid(lines, empty_symbols=["0"], parser="2048_tiles")
    if name == "ultimate_tic_tac_toe":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 9 or not all(
            re.fullmatch(r"[.xo]{3} [.xo]{3} [.xo]{3}", line) for line in lines
        ):
            return None
        result = _grid(
            [list(line.replace(" ", "")) for line in lines],
            legend=_CROSS_LEGEND,
            parser="ultimate_tic_tac_toe_subboards",
        )
        if result:
            result["group_size"] = 3
        return result
    if name in _RECTANGLES:
        symbols = re.escape(_RECTANGLES[name])
        # A known game's board rows are consecutive and have the same width.
        # Annotations (scores, chance-node labels, timers) stay in exact text.
        lines = text.splitlines()
        board_lines: list[str] = []
        for line in lines:
            if line and re.fullmatch("[" + symbols + "]+", line):
                board_lines.append(line)
            elif board_lines:
                if line and not (
                    (name == "stones_and_gems" and line.startswith("time left:"))
                    or (
                        name == "markov_soccer"
                        and (line.startswith("Chance Node") or line.startswith("Ball"))
                    )
                ):
                    return None
                break
        dimensions = {
            "tic_tac_toe": (3, 3),
            "mnk": (parameters.get("n", 3), parameters.get("m", 3)),
            "connect_four": (parameters.get("rows", 6), parameters.get("columns", 7)),
            "catch": (parameters.get("rows", 10), parameters.get("columns", 5)),
            "amazons": (
                parameters.get("board_size", 10),
                parameters.get("board_size", 10),
            ),
        }
        if name in dimensions:
            expected_rows, expected_columns = dimensions[name]
            if len(board_lines) != expected_rows or any(
                len(line) != expected_columns for line in board_lines
            ):
                return None
        legend = (
            _CROSS_LEGEND
            if name in {"tic_tac_toe", "mnk", "connect_four"}
            else {".": "Empty"}
        )
        if name == "amazons":
            legend = {
                ".": "Empty",
                "X": "Player 0 amazon",
                "O": "Player 1 amazon",
                "#": "Arrow / blocked",
            }
        if name == "catch":
            legend = {".": "Empty", "o": "Falling ball", "x": "Paddle"}
        return _grid(
            [list(line) for line in board_lines],
            legend=legend,
            parser="rectangular_" + name,
        )
    return None


def describe_state(state: Any) -> dict[str, Any] | None:
    """Return a public review model without changing state or player prompts.

    Unknown games and future public layouts retain their entire engine string.
    In particular, an imperfect-information game never calls ``str(state)`` or
    a player's observation here, including at terminal states.
    """
    game = state.get_game()
    if getattr(game, "environment_backend", "openspiel") == "native":
        return None
    game_type = game.get_type()
    current = int(state.current_player())
    terminal = state.is_terminal()
    kind = (
        "terminal"
        if terminal
        else "chance"
        if state.is_chance_node()
        else "simultaneous"
        if state.is_simultaneous_node()
        else "mean_field"
        if state.is_mean_field_node()
        else "decision"
    )
    active = [current] if current >= 0 and not terminal else []
    if kind == "simultaneous":
        active = [
            player
            for player in range(game.num_players())
            if state.legal_actions(player)
        ]
    try:
        move_number = state.move_number()
    except (AttributeError, RuntimeError, NotImplementedError):
        move_number = None
    public = game_type.information in {
        pyspiel.GameType.Information.PERFECT_INFORMATION,
        pyspiel.GameType.Information.ONE_SHOT,
    }
    result = {
        "schema_version": 1,
        "game": {
            "name": game_type.short_name,
            "long_name": game_type.long_name,
            "dynamics": _enum(game_type.dynamics),
            "information": _enum(game_type.information),
            "num_players": game.num_players(),
            "utility": _enum(game_type.utility),
            "reward_model": _enum(game_type.reward_model),
        },
        "node": {
            "kind": kind,
            "current_player": current,
            "active_players": active,
            "move_number": move_number,
            "terminal": terminal,
        },
        "visibility": {
            "public": public,
            "reason": "OpenSpiel declares this state public; the exact engine display is retained."
            if public
            else "OpenSpiel declares imperfect information. Full evaluator state is withheld; inspect each recorded actor's own observation separately.",
        },
        "board": {"format": "unavailable", "source": "unavailable", "text": None},
    }
    if public:
        text = str(state)
        result["board"] = {
            "format": "text",
            "source": "evaluator_state",
            "text": text,
            "renderer": "verbatim_text",
            "note": "This configuration uses the exact engine display; no dedicated grid interpretation is applied.",
        }
        if grid := _extract_grid(game_type.short_name, text, game.get_parameters()):
            result["board"].update(grid)
    return result
