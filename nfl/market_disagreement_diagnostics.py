"""Fixed model-vs-archive disagreement calibration, research only.

The archived final lines are not timestamp-verified closes or executable odds.
We measure whether model disagreement predicts the *direction* and relative
accuracy of outcomes, not a profitable wagering strategy.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns

BANDS = (0.0, 2.0, 5.0, 10.0, float("inf"))
MIN_GAMES = 30
TARGETS = {
    "margin": ("projected_home_margin", "market_home_margin", "actual_home_margin"),
    "total": ("projected_total", "market_total", "actual_total"),
}


def audit_disagreements(games: pl.DataFrame) -> dict[str, object]:
    keys = ("season", "week", "game_id")
    required = set(keys)
    for triple in TARGETS.values():
        required.update(triple)
    require_columns(games, required, "market_disagreement_games")
    if games.select(keys).unique().height != games.height:
        raise DataContractError("duplicate game keys in disagreement audit")
    results = {}
    for target, (model_col, market_col, actual_col) in TARGETS.items():
        valid = games.filter(pl.all_horizontal([
            pl.col(c).is_finite() for c in (model_col, market_col, actual_col)
        ])).with_columns(
            (pl.col(model_col) - pl.col(market_col)).alias("_gap"),
            (pl.col(actual_col) - pl.col(market_col)).alias("_outcome_vs_market"),
            ((pl.col(actual_col) - pl.col(model_col)).abs()
             < (pl.col(actual_col) - pl.col(market_col)).abs()).alias("_model_won"),
        )
        seasons = {}
        for (season,), season_games in valid.group_by("season"):
            bins = []
            for lower, upper in zip(BANDS[:-1], BANDS[1:], strict=True):
                for direction in ("model_higher", "model_lower"):
                    cohort = season_games.filter(
                        (pl.col("_gap").abs() >= lower)
                        & (pl.col("_gap").abs() < upper)
                        & ((pl.col("_gap") > 0) if direction == "model_higher"
                           else (pl.col("_gap") < 0))
                    )
                    count = cohort.height
                    bins.append({
                        "minimum_gap": lower, "maximum_gap": (
                            upper if upper != float("inf") else None
                        ),
                        "direction": direction, "games": count,
                        "model_more_accurate_rate": (
                            cohort["_model_won"].mean() if count >= MIN_GAMES else None
                        ),
                        "market_relative_realized_bias": (
                            cohort["_outcome_vs_market"].mean()
                            if count >= MIN_GAMES else None
                        ),
                        "status": "DESCRIPTIVE_ONLY" if count >= MIN_GAMES
                        else "INSUFFICIENT_SAMPLE",
                    })
            seasons[str(season)] = {
                "covered_games": season_games.height,
                "bins": bins,
            }
        results[target] = {
            "covered_games": valid.height,
            "missing_games": games.height - valid.height,
            "by_season": seasons,
        }
    return {
        "status": "MARKET_DISAGREEMENT_DIAGNOSTICS_RESEARCH_ONLY",
        "gap_bands_fixed_points": [0, 2, 5, 10],
        "market_data": "NFLVERSE_ARCHIVE_FINAL_UNVERIFIED_CLOSE_PROXY",
        "no_outcome_selected_thresholds": True,
        "no_timestamps_or_entry_prices_verified": True,
        "market_data_not_used_in_model_fitting": True,
        "not_staking_evidence": True,
        "production_model_changed": False,
        "targets": results,
    }
