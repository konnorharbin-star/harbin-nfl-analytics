"""Fixed football-only recency challenger against the NFL market archive.

Uses the existing opponent-adjusted Ridge fair-score model. The challenger
changes only current-season half-life (six completed weeks), never odds inputs.
Historical comparison is descriptive; market archive is not verified live close.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .recency import build_score_walkforward

HALF_LIFE = 6.0
TEST_SEASONS = (2024, 2025)
MIN_SEASON_GAMES = 150
KEYS = ("season", "week", "game_id")


def compare_core_challenger(
    schedules: pl.DataFrame, *, seasons: tuple[int, ...] = TEST_SEASONS,
) -> dict[str, object]:
    require_columns(
        schedules, {"season", "week", "game_id", "spread_line", "total_line"},
        "nfl_market_schedule",
    )
    if not seasons or tuple(sorted(set(seasons))) != seasons:
        raise ValueError("test seasons must be strictly increasing")
    if schedules.select(KEYS).unique().height != schedules.height:
        raise DataContractError("duplicate NFL schedule game keys")
    report: dict[str, object] = {}
    for season in seasons:
        original = build_score_walkforward(schedules, season, start_week=5)
        challenger = build_score_walkforward(
            schedules, season, start_week=5, half_life_weeks=HALF_LIFE
        )
        keys = list(KEYS)
        if original.height != challenger.height:
            raise DataContractError("challenger lost baseline games")
        comparison = original.join(
            challenger.select(
                *keys,
                pl.col("projected_home_margin").alias("challenger_margin"),
                pl.col("projected_total").alias("challenger_total"),
            ), on=keys, how="left", validate="1:1",
        ).join(
            schedules.select(*keys, "spread_line", "total_line"),
            on=keys, how="left", validate="1:1",
        )
        result = {}
        for target, actual, baseline, challenger_col, market in (
            ("margin", "actual_home_margin", "projected_home_margin",
             "challenger_margin", "spread_line"),
            ("total", "actual_total", "projected_total",
             "challenger_total", "total_line"),
        ):
            observed = comparison.filter(
                pl.col(actual).is_finite()
                & pl.col(baseline).is_finite()
                & pl.col(challenger_col).is_finite()
                & pl.col(market).is_finite()
            )
            n = observed.height
            def mae(column: str) -> float | None:
                return float(observed.select(
                    (pl.col(column) - pl.col(actual)).abs().mean()
                ).item()) if n else None
            base_mae, alt_mae, market_mae = (
                mae(baseline), mae(challenger_col), mae(market)
            )
            result[target] = {
                "games": n,
                "missing_market_games": comparison.height - n,
                "baseline_mae": base_mae,
                "challenger_mae": alt_mae,
                "archive_market_mae": market_mae,
                "beats_baseline": n >= MIN_SEASON_GAMES and alt_mae < base_mae,
                "beats_market_archive": (
                    n >= MIN_SEASON_GAMES and alt_mae < market_mae
                ),
                "status": "DESCRIPTIVE_ARCHIVE_RESEARCH" if n >= MIN_SEASON_GAMES
                else "INSUFFICIENT_MARKET_COVERAGE",
            }
        report[str(season)] = result
    return {
        "status": "HISTORICAL_CORE_CHALLENGER_RESEARCH_ONLY",
        "specification": "opponent-adjusted Ridge=8, six-week recency half-life",
        "archive_market_timestamp_verified": False,
        "training_uses_sportsbook": False,
        "model_selection_on_test_seasons": False,
        "historical_design_previously_inspected": True,
        "promotion_eligible": False,
        "staking_authorized": False,
        "by_season": report,
        "repeatable_margin_market_advantage": all(
            report[str(s)]["margin"]["beats_market_archive"] for s in seasons
        ),
        "repeatable_total_market_advantage": all(
            report[str(s)]["total"]["beats_market_archive"] for s in seasons
        ),
    }
