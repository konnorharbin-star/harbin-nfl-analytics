"""Leakage-aware retrospective NFL game intelligence.

Pregame projections are immutable inputs. Postgame results only appear in
grading outputs, never as candidate pregame predictive features.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns

NEEDED = {
    "season", "week", "game_id", "home_team", "away_team",
    "projected_home_margin", "projected_total",
    "actual_home_margin", "actual_total",
}


def build_game_intelligence(projections: pl.DataFrame) -> pl.DataFrame:
    """One graded game per (season, week, game_id); no betting-market leakage."""
    require_columns(projections, NEEDED, "game_intelligence_predictions")
    keys = ["season", "week", "game_id"]
    if projections.select(keys).unique().height != projections.height:
        raise DataContractError("duplicate game intelligence prediction keys")
    frame = projections.filter(
        pl.col("actual_home_margin").is_not_null()
        & pl.col("actual_total").is_not_null()
    )
    if frame.is_empty():
        raise DataContractError("no completed NFL games for intelligence")
    return frame.select(
        *keys, "home_team", "away_team",
        *(
            [pl.col("gameday")] if "gameday" in frame.columns else []
        ),
        "projected_home_margin", "projected_total",
        "actual_home_margin", "actual_total",
        (pl.col("actual_home_margin") - pl.col("projected_home_margin"))
        .alias("margin_residual"),
        (pl.col("actual_total") - pl.col("projected_total"))
        .alias("total_residual"),
        (pl.col("actual_home_margin") - pl.col("projected_home_margin"))
        .abs().alias("margin_absolute_error"),
        (pl.col("actual_total") - pl.col("projected_total"))
        .abs().alias("total_absolute_error"),
        pl.when(pl.col("projected_home_margin") > 0).then(pl.lit("home"))
        .when(pl.col("projected_home_margin") < 0).then(pl.lit("away"))
        .otherwise(pl.lit("pickem")).alias("projected_winner"),
        pl.when(pl.col("actual_home_margin") > 0).then(pl.lit("home"))
        .when(pl.col("actual_home_margin") < 0).then(pl.lit("away"))
        .otherwise(pl.lit("tie")).alias("actual_winner"),
    ).sort(keys)


def summarize_game_intelligence(games: pl.DataFrame) -> dict[str, object]:
    """Season/week summary of errors; descriptive, not a new training policy."""
    def summary(subset: pl.DataFrame) -> dict[str, object]:
        return {
            "games": subset.height,
            "margin_mae": subset["margin_absolute_error"].mean(),
            "total_mae": subset["total_absolute_error"].mean(),
            "mean_margin_residual": subset["margin_residual"].mean(),
            "mean_total_residual": subset["total_residual"].mean(),
        }
    return {
        "status": "POSTGAME_DIAGNOSTICS_ONLY",
        "total": summary(games),
        "by_season": {
            str(season): summary(group)
            for season, group in games.partition_by("season", as_dict=True)
            for season in [season[0]]
        },
        "by_season_week": [
            {"season": int(season), "week": int(week), **summary(group)}
            for (season, week), group in games.partition_by(
                ["season", "week"], as_dict=True
            ).items()
        ],
        "pregame_model_modified": False,
        "staking_authorized": False,
        "warning": (
            "Outcomes grade frozen pregame estimates; do not use residuals "
            "as future-game features."
        ),
    }
