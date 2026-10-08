"""Readable per-game NFL betting board with strict executable-quote labeling."""
from __future__ import annotations

import polars as pl

from .contracts import require_columns

COLUMNS = (
    "season", "week", "game_id", "away_team", "home_team",
    "model_margin_home", "model_total", "quant_market", "quant_side",
    "quant_book", "quant_price", "quant_odds", "quant_probability",
    "quant_edge", "quant_ev", "quant_quote_at", "recommendation_status",
    "production_signal", "research_signal", "execution_ready",
    "portfolio_stake_units", "context_veto", "probability_reliability_veto",
)
EXTRA = (
    "predicted_home_score", "predicted_away_score", "projected_winner",
    "betting_action", "quote_quality",
)


def build_actionable_board(current: pl.DataFrame) -> pl.DataFrame:
    """Show every game, including games with no qualified sportsbook quote."""
    require_columns(current, {"season", "week", "game_id", "home_team",
                              "away_team", "model_margin_home", "model_total"},
                    "nfl_current_actionable_board")
    if current.select("season", "week", "game_id").unique().height != current.height:
        raise ValueError("duplicate game in current betting board")
    data = current
    for name in COLUMNS:
        if name not in data.columns:
            data = data.with_columns(pl.lit(None).alias(name))
    # A missing verified price is not a play even when a model edge is high.
    ready = (
        pl.col("execution_ready").cast(pl.Boolean, strict=False).fill_null(False)
        & ~pl.col("context_veto").cast(pl.Boolean, strict=False).fill_null(True)
        & ~pl.col("probability_reliability_veto").cast(
            pl.Boolean, strict=False
        ).fill_null(True)
        & (pl.col("portfolio_stake_units").cast(pl.Float64, strict=False).fill_null(0) > 0)
        & pl.col("quant_market").is_not_null()
        & pl.col("quant_odds").is_not_null()
        & pl.col("quant_quote_at").is_not_null()
    )
    return data.with_columns(
        ((pl.col("model_total") + pl.col("model_margin_home")) / 2)
        .alias("predicted_home_score"),
        ((pl.col("model_total") - pl.col("model_margin_home")) / 2)
        .alias("predicted_away_score"),
        pl.when(pl.col("model_margin_home") > 0)
        .then(pl.col("home_team"))
        .when(pl.col("model_margin_home") < 0)
        .then(pl.col("away_team"))
        .otherwise(pl.lit("TIE_PROJECTION")).alias("projected_winner"),
        pl.when(ready).then(pl.lit("BET"))
        .when(pl.col("quant_market").is_null()).then(pl.lit("NO_VERIFIED_MARKET"))
        .otherwise(pl.lit("PASS")).alias("betting_action"),
        pl.when(ready).then(pl.lit("EXECUTION_READY"))
        .when(pl.col("quant_quote_at").is_null()).then(pl.lit("NO_VERIFIED_QUOTE"))
        .otherwise(pl.lit("NOT_EXECUTABLE")).alias("quote_quality"),
    ).select(*COLUMNS, *EXTRA).sort("season", "week", "game_id")


def write_actionable_board(
    current: pl.DataFrame, path: str = "outputs/actionable_betting_board.csv"
) -> pl.DataFrame:
    from pathlib import Path

    board = build_actionable_board(current)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.write_csv(destination)
    return board
