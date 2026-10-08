"""NFL extreme-forecast-error diagnostics from exclusively pregame PBP factors.

Postgame residuals define labels for retrospective diagnostics, never features.
No thresholds are optimized on outcomes, and no betting decisions are changed.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .game_intelligence_factors import PREGAME_FEATURES

ERROR_THRESHOLDS = {"margin": 14.0, "total": 17.0}
MIN_COHORT = 30


def analyze_error_tails(frame: pl.DataFrame) -> dict[str, object]:
    required = {"season", "week", "game_id", "margin_residual",
                "total_residual", *PREGAME_FEATURES}
    require_columns(frame, required, "intelligence_error_tail")
    if frame.select("season", "week", "game_id").unique().height != frame.height:
        raise DataContractError("duplicate game in error-tail analysis")
    results: dict[str, object] = {}
    for target, threshold in ERROR_THRESHOLDS.items():
        residual = f"{target}_residual"
        valid = frame.filter(pl.col(residual).is_finite())
        tail = valid.filter(pl.col(residual).abs() >= threshold)
        feature_rows: dict[str, object] = {}
        for feature in PREGAME_FEATURES:
            if not feature.endswith(f"_{target}_signal"):
                continue
            cohort = valid.filter(pl.col(feature).is_finite())
            high = cohort.filter(pl.col(feature) >= 0)
            low = cohort.filter(pl.col(feature) < 0)
            segments = {}
            for name, subset in (("nonnegative", high), ("negative", low)):
                count = subset.height
                extreme = subset.filter(pl.col(residual).abs() >= threshold).height
                segments[name] = {
                    "games": count,
                    "extreme_errors": extreme,
                    "extreme_error_rate": extreme / count if count >= MIN_COHORT else None,
                    "status": "DESCRIPTIVE_ONLY" if count >= MIN_COHORT
                    else "INSUFFICIENT_SAMPLE",
                }
            feature_rows[feature] = segments
        by_season = {}
        for (season,), cohort in valid.group_by("season"):
            extremes = cohort.filter(pl.col(residual).abs() >= threshold).height
            by_season[str(season)] = {
                "games": cohort.height, "extreme_errors": extremes,
                "extreme_error_rate": extremes / cohort.height,
            }
        results[target] = {
            "threshold_points": threshold,
            "games": valid.height,
            "extreme_errors": tail.height,
            "extreme_error_rate": (
                tail.height / valid.height if valid.height else None
            ),
            "by_season": by_season,
            "pregame_factor_sign_splits": feature_rows,
        }
    return {
        "status": "RESEARCH_ONLY",
        "thresholds_frozen_in_code": True,
        "selection_warning": (
            "Overlapping exploratory factor splits; descriptive correlations "
            "do not establish causal advantage or betting profitability."
        ),
        "targets": results,
        "production_model_changed": False,
        "staking_authorized": False,
    }
