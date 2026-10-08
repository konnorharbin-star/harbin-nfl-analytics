"""Pregame efficiency versus postgame NFL error attribution; research only.

Never derive prediction features from results or current-game play-by-play.
Only join externally built, chronologically frozen prior-game snapshots.
"""
from __future__ import annotations

import math

import polars as pl

from .contracts import DataContractError, require_columns
from .recent_form import RECENT_PBP_METRICS

KEYS = ("season", "week", "game_id")

PREGAME_FEATURES = tuple(
    f"recent_{metric}_{target}_signal"
    for metric in RECENT_PBP_METRICS
    for target in ("margin", "total")
)
# Snapshot field names are validated against the actual existing feature schema.
FORBIDDEN = (
    "actual_", "margin_residual", "total_residual",
    "margin_absolute_error", "total_absolute_error",
)


def attach_pregame_factors(
    graded: pl.DataFrame,
    snapshot: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Join strictly on game identifiers; prohibit current-game result features."""
    require_columns(graded, set(KEYS) | {"margin_residual", "total_residual"},
                    "game_intelligence_graded")
    require_columns(snapshot, set(KEYS) | set(feature_columns),
                    "game_intelligence_pregame_snapshot")
    if not feature_columns:
        raise DataContractError("no requested pregame factor columns")
    if any(
        name in KEYS or name in graded.columns or
        any(name.startswith(prefix) for prefix in FORBIDDEN)
        for name in feature_columns
    ):
        raise DataContractError("pregame features overlap outcomes or existing fields")
    if snapshot.select(KEYS).unique().height != snapshot.height:
        raise DataContractError("duplicate pregame factor snapshots")
    if graded.select(KEYS).unique().height != graded.height:
        raise DataContractError("duplicate graded game identifiers")
    if snapshot.filter(pl.col("season").is_null() | pl.col("week").is_null()
                       | pl.col("game_id").is_null()).height:
        raise DataContractError("snapshot contains missing game identifiers")
    return graded.join(
        snapshot.select(*KEYS, *feature_columns),
        on=list(KEYS), how="left", validate="1:1"
    ).sort(KEYS)


def evaluate_factor_diagnostics(
    merged: pl.DataFrame, feature_columns: tuple[str, ...],
    *, min_games: int = 30,
) -> dict[str, object]:
    """Measure descriptive association only, not causal effects or betting edges."""
    output: dict[str, object] = {}
    for feature in feature_columns:
        rows = []
        for row in merged.select(feature, "margin_residual", "total_residual").iter_rows():
            try:
                values = tuple(float(x) for x in row)
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(x) for x in values):
                rows.append(values)
        if len(rows) < min_games:
            output[feature] = {"status": "INSUFFICIENT_SAMPLE", "games": len(rows)}
            continue
        values = pl.DataFrame(
            rows, schema=[feature, "margin_residual", "total_residual"], orient="row"
        )
        output[feature] = {
            "status": "DESCRIPTIVE_ONLY",
            "games": len(rows),
            "margin_residual_correlation": values.select(
                pl.corr(feature, "margin_residual")
            ).item(),
            "total_residual_correlation": values.select(
                pl.corr(feature, "total_residual")
            ).item(),
        }
    return {
        "status": "PREGAME_FACTOR_ERROR_DIAGNOSTICS_ONLY",
        "features": output,
        "pregame_model_modified": False,
        "staking_authorized": False,
        "validation_required": "chronological out-of-sample tests before model inclusion",
    }
