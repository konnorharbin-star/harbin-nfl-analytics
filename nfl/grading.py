"""Independent grading for persisted NFL PAPER/SHADOW/PRODUCTION decisions."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .line_history import latest_pre_kickoff_snapshot, load_market_snapshots
from .market import american_implied_probability, american_to_decimal

DECISION_REQUIRED = {
    "decision_at",
    "game_id",
    "season",
    "week",
    "quant_market",
    "quant_side",
    "quant_odds",
    "portfolio_candidate_units",
    "portfolio_action",
}


def _parse_dt(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _edge_bucket(value: object) -> str:
    try:
        edge = float(value)
    except (TypeError, ValueError):
        return "missing"
    if edge < 0.02:
        return "<2%"
    if edge < 0.04:
        return "2-4%"
    if edge < 0.06:
        return "4-6%"
    return ">=6%"


def _grade_value(market: str, side: str, line: float | None, margin: float, total: float) -> float:
    if market == "moneyline":
        return margin if side == "home" else -margin
    if market == "spread":
        if line is None:
            raise DataContractError("spread decision is missing its entry line")
        return margin + line if side == "home" else -margin + line
    if market == "total":
        if line is None:
            raise DataContractError("total decision is missing its entry line")
        return total - line if side == "over" else line - total
    raise DataContractError(f"unsupported graded market: {market}")


def _closing_side(snapshot: dict[str, object], side: str) -> tuple[float | None, int | None]:
    for prefix in ("first", "second"):
        if str(snapshot.get(f"{prefix}_side")) == side:
            raw_line = snapshot.get(f"{prefix}_line")
            raw_odds = snapshot.get(f"{prefix}_american_odds")
            line = None if raw_line in {None, ""} else float(raw_line)
            odds = None if raw_odds in {None, ""} else int(float(raw_odds))
            return line, odds
    return None, None


def _execution_clv(
    market: str,
    side: str,
    entry_line: float | None,
    entry_odds: int,
    closing_line: float | None,
    closing_odds: int | None,
) -> float | None:
    if market == "spread" and entry_line is not None and closing_line is not None:
        return entry_line - closing_line
    if market == "total" and entry_line is not None and closing_line is not None:
        return closing_line - entry_line if side == "over" else entry_line - closing_line
    if market == "moneyline" and closing_odds is not None:
        return american_implied_probability(closing_odds) - american_implied_probability(entry_odds)
    return None


def _earliest_decisions(decisions: pl.DataFrame) -> pl.DataFrame:
    require_columns(decisions, DECISION_REQUIRED, "portfolio_decisions")
    if decisions.is_empty():
        return decisions
    rows = decisions.to_dicts()
    rows.sort(key=lambda row: _parse_dt(row.get("decision_at")) or datetime.max.replace(tzinfo=UTC))
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        units = float(row.get("portfolio_candidate_units") or 0.0)
        action = str(row.get("portfolio_action") or "PASS").upper()
        if units <= 0 or action not in {"PAPER", "SHADOW", "BET"}:
            continue
        key = (str(row["game_id"]), str(row["quant_market"]))
        selected.setdefault(key, row)
    return pl.DataFrame(list(selected.values())) if selected else pl.DataFrame()


def grade_portfolio_decisions(
    decisions: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    snapshots: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Grade the first persisted portfolio decision per game/market after final scores exist."""

    require_columns(
        schedules,
        {"game_id", "home_score", "away_score", "gameday", "home_team", "away_team"},
        "grading_schedules",
    )
    entries = _earliest_decisions(decisions)
    if entries.is_empty():
        return pl.DataFrame()
    final_map = {str(row["game_id"]): row for row in schedules.iter_rows(named=True)}
    history = snapshots if snapshots is not None else pl.DataFrame()
    graded: list[dict[str, object]] = []

    for entry in entries.to_dicts():
        game = final_map.get(str(entry["game_id"]))
        if game is None or game.get("home_score") is None or game.get("away_score") is None:
            continue
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        if not isfinite(home_score) or not isfinite(away_score):
            continue
        margin = home_score - away_score
        total = home_score + away_score
        market = str(entry["quant_market"])
        side = str(entry["quant_side"])
        line = entry.get("quant_price")
        entry_line = None if line in {None, ""} else float(line)
        odds = int(float(entry["quant_odds"]))
        value = _grade_value(market, side, entry_line, margin, total)
        decimal = american_to_decimal(odds)
        if value > 1e-12:
            result, units = "win", decimal - 1.0
        elif value < -1e-12:
            result, units = "loss", -1.0
        else:
            result, units = "push", 0.0

        close_line: float | None = None
        close_odds: int | None = None
        close_at: str | None = None
        if not history.is_empty() and entry.get("quant_book"):
            later = latest_pre_kickoff_snapshot(
                history,
                game_id=str(entry["game_id"]),
                market_type=market,
                book=str(entry["quant_book"]),
                decision_at=str(entry["decision_at"]),
            )
            if later is not None:
                close_line, close_odds = _closing_side(later, side)
                close_at = str(later.get("captured_at"))
        clv = _execution_clv(market, side, entry_line, odds, close_line, close_odds)

        row = dict(entry)
        row.update(
            {
                "home_score": home_score,
                "away_score": away_score,
                "actual_home_margin": margin,
                "actual_total": total,
                "result": result,
                "net_units": units,
                "profit": units,
                "edge_bucket": _edge_bucket(entry.get("quant_edge")),
                "closing_price": close_line,
                "closing_odds": close_odds,
                "closing_snapshot_at": close_at,
                "execution_clv": clv,
                "portfolio_verified": True,
                "entry_quote_verified": _parse_dt(entry.get("quant_quote_at")) is not None,
            }
        )
        graded.append(row)
    return pl.DataFrame(graded) if graded else pl.DataFrame()


def _max_drawdown(values: Iterable[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _roi_ci(values: list[float]) -> list[float | None]:
    if len(values) < 30:
        return [None, None]
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / sqrt(len(array)))
    return [mean - 1.96 * se, mean + 1.96 * se]


def _summary(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty():
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "max_drawdown": 0.0,
            "avg_clv": None,
            "roi_ci_95": [None, None],
        }
    profits = [float(value) for value in frame.get_column("net_units").to_list()]
    wins = frame.filter(pl.col("result") == "win").height
    losses = frame.filter(pl.col("result") == "loss").height
    pushes = frame.filter(pl.col("result") == "push").height
    decided = wins + losses
    clv = frame.get_column("execution_clv").drop_nulls().to_list()
    return {
        "bets": frame.height,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decided if decided else None,
        "units": sum(profits),
        "roi": sum(profits) / frame.height,
        "max_drawdown": _max_drawdown(profits),
        "avg_clv": None if not clv else float(sum(clv) / len(clv)),
        "clv_samples": len(clv),
        "roi_ci_95": _roi_ci(profits),
    }


def summarize_live_grading(graded: pl.DataFrame) -> dict[str, object]:
    if graded.is_empty():
        return {
            "evidence_source": "portfolio_decisions_v1",
            "portfolio_verified": True,
            "graded_bets": 0,
            "overall": _summary(graded),
            "by_market": {},
            "by_signal": {},
            "by_edge_bucket": {},
            "by_season": {},
        }

    def grouped(column: str) -> dict[str, object]:
        output: dict[str, object] = {}
        for value in graded.get_column(column).unique().sort().to_list():
            output[str(value)] = _summary(graded.filter(pl.col(column) == value))
        return output

    overall = _summary(graded)
    return {
        "evidence_source": "portfolio_decisions_v1",
        "portfolio_verified": True,
        "graded_bets": graded.height,
        "bets": graded.height,
        "roi": overall["roi"],
        "avg_clv": overall["avg_clv"],
        "overall": overall,
        "by_market": grouped("quant_market"),
        "by_signal": grouped("quant_signal"),
        "by_edge_bucket": grouped("edge_bucket"),
        "by_season": grouped("season"),
    }


def write_live_grading(
    graded: pl.DataFrame,
    *,
    graded_path: str | Path = "reports/live_graded_bets.csv",
    report_path: str | Path = "reports/live_performance.json",
    output_path: str | Path = "outputs/live_performance.json",
) -> dict[str, object]:
    summary = summarize_live_grading(graded)
    for path in (report_path, output_path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    target = Path(graded_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not graded.is_empty():
        graded.write_csv(target)
    elif not target.exists():
        with target.open("w", newline="") as handle:
            csv.writer(handle).writerow(["game_id", "quant_market", "result", "net_units"])
    return summary


def load_decision_ledger(path: str | Path = "history/portfolio_decisions_v1.csv") -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    return pl.read_csv(source, try_parse_dates=False)


def grade_live_from_files(
    schedules: pl.DataFrame,
    *,
    decision_path: str | Path = "history/portfolio_decisions_v1.csv",
    snapshot_path: str | Path = "history/market_snapshots.csv",
) -> tuple[pl.DataFrame, dict[str, object]]:
    decisions = load_decision_ledger(decision_path)
    snapshots = load_market_snapshots(snapshot_path)
    graded = grade_portfolio_decisions(decisions, schedules, snapshots=snapshots)
    return graded, write_live_grading(graded)
