"""Fixed NFL efficiency challenger against the independent fair-score baseline.

Research only: each evaluation season is predicted using strictly earlier
seasons; game outcomes never become same-game inputs. No selection on test folds.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .game_intelligence_factors import PREGAME_FEATURES
from .oa_residuals import FeatureRidgeModel

FEATURES_BY_TARGET = {
    "margin": tuple(x for x in PREGAME_FEATURES if x.endswith("_margin_signal")),
    "total": tuple(x for x in PREGAME_FEATURES if x.endswith("_total_signal")),
}
RIDGE_ALPHA = 100.0
BLEND = 0.25
MIN_TRAIN = 100
MIN_TEST = 30


def evaluate_intelligence_challenger(
    frame: pl.DataFrame, *, test_seasons: tuple[int, ...] = (2024, 2025),
) -> dict[str, object]:
    """Measure fixed Ridge residual challenger; fail closed on partial samples."""
    required = {"season", "week", "game_id", "margin_residual", "total_residual",
                "projected_home_margin", "projected_total",
                "actual_home_margin", "actual_total", *PREGAME_FEATURES}
    require_columns(frame, required, "game_intelligence_challenger")
    keys = ["season", "week", "game_id"]
    if frame.select(keys).unique().height != frame.height:
        raise DataContractError("duplicate game keys in intelligence challenger")
    if not test_seasons or tuple(sorted(set(test_seasons))) != test_seasons:
        raise ValueError("test seasons must be unique and chronological")
    output: dict[str, object] = {}
    for target in ("margin", "total"):
        features = FEATURES_BY_TARGET[target]
        baseline_col = "projected_home_margin" if target == "margin" else "projected_total"
        actual_col = "actual_home_margin" if target == "margin" else "actual_total"
        residual_col = f"{target}_residual"
        folds: list[dict[str, object]] = []
        for season in test_seasons:
            train = frame.filter(pl.col("season") < season)
            test = frame.filter(pl.col("season") == season)
            valid_expr = pl.all_horizontal([
                pl.col(name).is_finite() for name in (*features, residual_col)
            ])
            train = train.filter(valid_expr)
            test = test.filter(
                pl.all_horizontal([pl.col(name).is_finite()
                                   for name in (*features, baseline_col, actual_col)])
            )
            if train.height < MIN_TRAIN or test.height < MIN_TEST:
                folds.append({"season": season, "status": "INSUFFICIENT_SAMPLE",
                              "train_games": train.height, "test_games": test.height})
                continue
            model = FeatureRidgeModel(features, RIDGE_ALPHA).fit(train, residual_col)
            baseline = np.asarray(test[baseline_col], dtype=float)
            actual = np.asarray(test[actual_col], dtype=float)
            adjusted = baseline + BLEND * np.asarray(model.predict(test), dtype=float)
            mae_base = float(np.mean(np.abs(actual - baseline)))
            mae_new = float(np.mean(np.abs(actual - adjusted)))
            rmse_base = math.sqrt(float(np.mean((actual - baseline) ** 2)))
            rmse_new = math.sqrt(float(np.mean((actual - adjusted) ** 2)))
            folds.append({
                "season": season, "status": "DESCRIPTIVE_HISTORICAL",
                "train_games": train.height, "test_games": test.height,
                "baseline_mae": mae_base, "challenger_mae": mae_new,
                "baseline_rmse": rmse_base, "challenger_rmse": rmse_new,
                "mae_improvement": mae_base - mae_new,
                "rmse_improvement": rmse_base - rmse_new,
            })
        output[target] = {
            "folds": folds,
            "positive_mae_folds": sum(x.get("mae_improvement", 0) > 0 for x in folds),
            "consistent_improvement": (
                len(folds) == len(test_seasons) and
                all(x.get("mae_improvement", 0) > 0 and
                    x.get("rmse_improvement", 0) > 0 for x in folds)
            ),
        }
    return {
        "status": "RESEARCH_ONLY",
        "specification": "fixed pregame PBP features, ridge alpha 100, blend 0.25",
        "historical_folds_already_inspected": True,
        "independent_2026_forward_proof_required": True,
        "targets": output,
        "canonical_score_changed": False,
        "betting_policy_changed": False,
    }
