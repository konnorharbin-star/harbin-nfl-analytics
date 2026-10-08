"""NFL fair-score accuracy versus nflverse archive-final market benchmarks.

Archive-final spread/total fields are closing *proxies*, NOT independently
timestamp-verified sportsbook close or executable entry odds.
"""
from __future__ import annotations

import math

import polars as pl

from .contracts import DataContractError, require_columns

KEYS = ("season", "week", "game_id")
MIN_SEASON_GAMES = 100


def build_market_relative_games(predictions: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    require_columns(predictions, set(KEYS) | {
        "projected_home_margin", "projected_total",
        "actual_home_margin", "actual_total",
    }, "pregame_predictions")
    require_columns(schedules, set(KEYS) | {"spread_line", "total_line"}, "archive_market")
    for label, frame in (("predictions", predictions), ("schedules", schedules)):
        if frame.select(KEYS).unique().height != frame.height:
            raise DataContractError(f"duplicate {label} game keys")
    joined = predictions.join(
        schedules.select(*KEYS, "spread_line", "total_line"),
        on=list(KEYS), how="left", validate="1:1",
    ).with_columns(
        # nflverse spread_line is positive for home favorites: directly
        # comparable to predicted and realized home scoring margin.
        pl.col("spread_line").cast(pl.Float64, strict=False).alias("market_home_margin"),
        pl.col("total_line").cast(pl.Float64, strict=False).alias("market_total"),
    )
    return joined.with_columns(
        (pl.col("projected_home_margin") - pl.col("actual_home_margin"))
        .abs().alias("model_margin_error"),
        (pl.col("market_home_margin") - pl.col("actual_home_margin"))
        .abs().alias("market_margin_error"),
        (pl.col("projected_total") - pl.col("actual_total"))
        .abs().alias("model_total_error"),
        (pl.col("market_total") - pl.col("actual_total"))
        .abs().alias("market_total_error"),
        (pl.col("projected_home_margin") - pl.col("market_home_margin"))
        .alias("margin_model_market_disagreement"),
        (pl.col("projected_total") - pl.col("market_total"))
        .alias("total_model_market_disagreement"),
    ).sort(KEYS)


def evaluate_market_relative_games(games: pl.DataFrame) -> dict[str, object]:
    """Paired gamewise errors, with 2024 and 2025 fixed replication reporting."""
    require_columns(games, {"season", "week", "game_id", "model_margin_error",
        "market_margin_error", "model_total_error", "market_total_error"},
        "market_relative_games")
    results = {}
    for target in ("margin", "total"):
        model_col = f"model_{target}_error"
        market_col = f"market_{target}_error"
        valid = games.filter(
            pl.col(model_col).is_finite() & pl.col(market_col).is_finite()
        )
        seasons = {}
        for (season,), cohort in valid.group_by("season"):
            n = cohort.height
            model_mae = float(cohort[model_col].mean())
            market_mae = float(cohort[market_col].mean())
            seasons[str(season)] = {
                "games": n, "model_mae": model_mae,
                "market_proxy_mae": market_mae,
                "mae_advantage_points": market_mae - model_mae,
                "status": "DESCRIPTIVE_ARCHIVE_PROXY" if n >= MIN_SEASON_GAMES
                else "INSUFFICIENT_SAMPLE",
            }
        paired = all(
            seasons.get(str(year), {}).get("games", 0) >= MIN_SEASON_GAMES
            and seasons[str(year)]["mae_advantage_points"] > 0
            for year in (2024, 2025)
        )
        results[target] = {
            "covered_games": valid.height,
            "missing_market_games": games.height - valid.height,
            "by_season": {k: seasons[k] for k in sorted(seasons)},
            "model_beat_archive_proxy_both_years": paired,
            "production_promotion_eligible": False,
        }
    return {
        "status": "ARCHIVE_FINAL_MARKET_COMPARISON_RESEARCH_ONLY",
        "market_source": "NFLVERSE_SCHEDULE_ARCHIVE_FINAL",
        "closing_timestamp_verified": False,
        "executable_market_evidence": False,
        "opening_line_used_as_close": False,
        "model_projections_not_refit_to_market": True,
        "multiple_testing_adjusted": False,
        "independent_forward_validation_required": True,
        "staking_authorized": False,
        "targets": results,
    }
