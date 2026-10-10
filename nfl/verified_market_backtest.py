"""Promotion-quality historical NFL market evidence from timestamped provider quotes.

The football projections remain identical to the free archive backtest. This module
only replaces downstream market price provenance with point-in-time sportsbook
snapshots observed at explicit pre-kickoff decision timestamps.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .contracts import DataContractError, require_columns
from .free_market_backtest import PROJECTION_REQUIRED, summarize_archive_bets
from .market import remove_two_way_vig
from .market_backtest import (
    compare_selected_market_snapshots,
    grade_market_comparisons,
)
from .probability import GaussianScoreDistribution

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_ENTRY_MINUTES_BEFORE_KICKOFF = 60
DEFAULT_CLOSE_MINUTES_BEFORE_KICKOFF = 5
DEFAULT_MAX_QUOTE_AGE_MINUTES = 15


def eligible_projection_game_ids(
    projections: pl.DataFrame,
    *,
    min_probability_training_games: int = 64,
) -> list[str]:
    """Return games whose probability distribution can be fit chronologically."""

    require_columns(projections, PROJECTION_REQUIRED, "verified_market_projections")
    if min_probability_training_games < 64:
        raise ValueError("min_probability_training_games must be >= 64")

    eligible: list[str] = []
    groups = projections.select("season", "week").unique().sort(["season", "week"])
    for group in groups.iter_rows(named=True):
        season = int(group["season"])
        week = int(group["week"])
        history = projections.filter(
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        if history.height < min_probability_training_games:
            continue
        eligible.extend(
            str(value)
            for value in projections.filter(
                (pl.col("season") == season) & (pl.col("week") == week)
            )
            .get_column("game_id")
            .to_list()
        )
    return sorted(set(eligible))


def build_market_decisions(
    schedules: pl.DataFrame,
    game_ids: list[str],
    *,
    minutes_before_kickoff: int,
) -> pl.DataFrame:
    """Create one explicit UTC historical decision timestamp per target game."""

    if minutes_before_kickoff < 0:
        raise ValueError("minutes_before_kickoff must be >= 0")
    require_columns(
        schedules,
        {"game_id", "gameday", "gametime"},
        "verified_market_schedule",
    )
    targets = set(str(value) for value in game_ids)
    rows: list[dict[str, object]] = []
    for row in schedules.filter(
        pl.col("game_id").cast(pl.String).is_in(sorted(targets))
    ).iter_rows(named=True):
        day = row.get("gameday")
        time = row.get("gametime")
        if day in {None, ""} or time in {None, ""}:
            continue
        try:
            local = datetime.fromisoformat(f"{day}T{time}")
        except ValueError:
            continue
        if local.tzinfo is None:
            local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
        kickoff = local.astimezone(UTC)
        rows.append(
            {
                "game_id": str(row["game_id"]),
                "decision_time": kickoff - timedelta(
                    minutes=minutes_before_kickoff
                ),
            }
        )
    if not rows:
        raise DataContractError("verified historical market decisions are empty")
    decisions = pl.DataFrame(rows).sort("game_id")
    if decisions.get_column("game_id").n_unique() != len(targets):
        missing = sorted(
            targets.difference(
                str(value)
                for value in decisions.get_column("game_id").to_list()
            )
        )
        raise DataContractError(
            "missing kickoff timestamps for verified historical games: "
            + ", ".join(missing[:10])
        )
    return decisions


def build_chronological_market_comparisons(
    projections: pl.DataFrame,
    selected_entry_quotes: pl.DataFrame,
    *,
    min_probability_training_games: int = 64,
) -> pl.DataFrame:
    """Compare timestamped entries with a distribution fit only on earlier games."""

    require_columns(projections, PROJECTION_REQUIRED, "verified_market_projections")
    if selected_entry_quotes.is_empty():
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    groups = projections.select("season", "week").unique().sort(["season", "week"])
    for group in groups.iter_rows(named=True):
        season = int(group["season"])
        week = int(group["week"])
        history = projections.filter(
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        if history.height < min_probability_training_games:
            continue
        targets = projections.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        game_ids = [str(value) for value in targets.get_column("game_id").to_list()]
        quotes = selected_entry_quotes.filter(
            pl.col("game_id").cast(pl.String).is_in(game_ids)
        )
        if quotes.is_empty():
            continue
        distribution = GaussianScoreDistribution().fit(history)
        compared = compare_selected_market_snapshots(
            targets,
            quotes,
            distribution,
        )
        if not compared.is_empty():
            frames.append(compared)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(
        ["game_id", "market_type", "book", "side"]
    )


def _best_market_rows(comparisons: pl.DataFrame) -> pl.DataFrame:
    if comparisons.is_empty():
        return comparisons
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in comparisons.iter_rows(named=True):
        key = (str(row["game_id"]), str(row["market_type"]))
        grouped.setdefault(key, []).append(row)

    for _, candidates in sorted(grouped.items()):
        books = {str(row["book"]) for row in candidates}
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["expected_value_per_unit"]),
                float(row["probability_edge"]),
                int(row["american_odds"]),
                str(row["book"]),
                str(row["side"]),
            ),
        )
        output = dict(chosen)
        output["market_book_count"] = len(books)
        rows.append(output)
    return pl.DataFrame(rows).sort(["game_id", "market_type"])


def _pair_map(frame: pl.DataFrame) -> dict[tuple[str, str, str], dict[str, dict[str, object]]]:
    output: dict[
        tuple[str, str, str],
        dict[str, dict[str, object]],
    ] = {}
    if frame.is_empty():
        return output
    for row in frame.iter_rows(named=True):
        key = (
            str(row["game_id"]),
            str(row["book"]),
            str(row["market_type"]),
        )
        output.setdefault(key, {})[str(row["side"])] = row
    return output


def _historical_clv(
    entry: dict[str, object],
    *,
    entry_pair: dict[str, dict[str, object]],
    closing_pair: dict[str, dict[str, object]],
) -> tuple[float | None, dict[str, object] | None]:
    market = str(entry["market_type"])
    side = str(entry["side"])
    close = closing_pair.get(side)
    if close is None:
        return None, None

    if market == "spread":
        if entry.get("line") is None or close.get("line") is None:
            return None, close
        return float(entry["line"]) - float(close["line"]), close

    if market == "total":
        if entry.get("line") is None or close.get("line") is None:
            return None, close
        if side == "over":
            return float(close["line"]) - float(entry["line"]), close
        return float(entry["line"]) - float(close["line"]), close

    if market != "moneyline":
        return None, close

    required = {"home", "away"}
    if set(entry_pair) != required or set(closing_pair) != required:
        return None, close
    entry_home, entry_away = remove_two_way_vig(
        int(entry_pair["home"]["american_odds"]),
        int(entry_pair["away"]["american_odds"]),
    )
    close_home, close_away = remove_two_way_vig(
        int(closing_pair["home"]["american_odds"]),
        int(closing_pair["away"]["american_odds"]),
    )
    if side == "home":
        return close_home - entry_home, close
    return close_away - entry_away, close


def build_verified_market_bets(
    projections: pl.DataFrame,
    selected_entry_quotes: pl.DataFrame,
    selected_closing_quotes: pl.DataFrame,
    *,
    min_probability_training_games: int = 64,
) -> pl.DataFrame:
    """Line-shop, grade, and attach same-book CLV to verified historical entries."""

    comparisons = build_chronological_market_comparisons(
        projections,
        selected_entry_quotes,
        min_probability_training_games=min_probability_training_games,
    )
    chosen = _best_market_rows(comparisons)
    if chosen.is_empty():
        return chosen

    outcomes = projections.select(
        "game_id",
        "actual_home_margin",
        "actual_total",
    )
    graded = grade_market_comparisons(chosen, outcomes)
    metadata = projections.select(
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    ).unique(subset=["game_id"])
    graded = graded.join(metadata, on="game_id", how="left")

    entry_pairs = _pair_map(selected_entry_quotes)
    closing_pairs = _pair_map(selected_closing_quotes)
    rows: list[dict[str, object]] = []
    for source in graded.iter_rows(named=True):
        row = dict(source)
        key = (
            str(row["game_id"]),
            str(row["book"]),
            str(row["market_type"]),
        )
        clv, close = _historical_clv(
            row,
            entry_pair=entry_pairs.get(key, {}),
            closing_pair=closing_pairs.get(key, {}),
        )
        row["entry_line_observed"] = True
        row["entry_price_verified"] = True
        row["entry_quote_verified"] = True
        row["entry_timestamp_verified"] = True
        row["entry_price_stage"] = "timestamped_provider_entry"
        row["price_stage"] = "timestamped_provider_entry"
        row["clv_proxy"] = clv
        row["closing_quote_verified"] = close is not None
        row["closing_snapshot_at"] = (
            None if close is None else close.get("captured_at")
        )
        row["closing_line"] = None if close is None else close.get("line")
        row["closing_odds"] = (
            None if close is None else close.get("american_odds")
        )
        rows.append(row)

    return pl.DataFrame(rows).sort(
        ["season", "week", "game_id", "market_type"]
    )


def _segments(
    bets: pl.DataFrame,
    column: str,
) -> dict[str, dict[str, object]]:
    if bets.is_empty() or column not in bets.columns:
        return {}
    output: dict[str, dict[str, object]] = {}
    for value in bets.get_column(column).unique().sort().to_list():
        output[str(value)] = summarize_archive_bets(
            bets.filter(pl.col(column) == value)
        ).to_dict()
    return output


def build_verified_market_report(
    bets: pl.DataFrame,
    *,
    entry_minutes_before_kickoff: int,
    close_minutes_before_kickoff: int,
    max_quote_age_minutes: int,
) -> dict[str, object]:
    overall = (
        summarize_archive_bets(bets).to_dict()
        if not bets.is_empty()
        else {}
    )
    clv_samples = int(overall.get("clv_samples", 0) or 0)
    clv_coverage = clv_samples / bets.height if bets.height else 0.0
    books = (
        sorted(str(value) for value in bets.get_column("book").unique().to_list())
        if not bets.is_empty() and "book" in bets.columns
        else []
    )
    return {
        "version": 1,
        "status": "READY" if bets.height else "EMPTY",
        "provider": "the_odds_api",
        "timestamped_entry_prices": True,
        "entry_minutes_before_kickoff": entry_minutes_before_kickoff,
        "close_minutes_before_kickoff": close_minutes_before_kickoff,
        "max_quote_age_minutes": max_quote_age_minutes,
        "bets": bets.height,
        "books": books,
        "book_count": len(books),
        "clv_samples": clv_samples,
        "clv_coverage": clv_coverage,
        "overall": overall,
        "by_market": _segments(bets, "market_type"),
        "by_season": _segments(bets, "season"),
        "meaning": (
            "Promotion-quality historical sample using explicit timestamped provider "
            "entry prices and same-book later pre-kickoff closing observations. "
            "Sportsbook data remains downstream of the fair-score model."
        ),
    }


def write_verified_market_backtest(
    bets: pl.DataFrame,
    *,
    entry_minutes_before_kickoff: int,
    close_minutes_before_kickoff: int,
    max_quote_age_minutes: int,
    bets_path: str | Path = "reports/verified_market_bets.csv",
    report_path: str | Path = "reports/verified_market_backtest.json",
) -> dict[str, object]:
    report = build_verified_market_report(
        bets,
        entry_minutes_before_kickoff=entry_minutes_before_kickoff,
        close_minutes_before_kickoff=close_minutes_before_kickoff,
        max_quote_age_minutes=max_quote_age_minutes,
    )
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    bets_target = Path(bets_path)
    bets_target.parent.mkdir(parents=True, exist_ok=True)
    if not bets.is_empty():
        bets.write_csv(bets_target)
    return report
