"""Pregame EPA/success/explosive challenger versus baseline and archived market.

This is a fixed, retrospective, football-only Ridge correction experiment.
Scores for each season are generated using only its preceding seasons to fit
corrections; market archive is a comparator, never a training feature.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .oa_residuals import FeatureRidgeModel

METRICS = ("epa_per_play", "success_rate", "explosive_rate")
TRAIN_SEASONS = (2022, 2023, 2024, 2025)
TEST_SEASONS = (2024, 2025)
ALPHA = 100.0
BLEND = 0.25
MIN_TRAIN = 100
MIN_TEST = 100


def evaluate_efficiency_market(frame: pl.DataFrame) -> dict[str, object]:
    features = {
        t: tuple(f"recent_{metric}_{t}_signal" for metric in METRICS)
        for t in ("margin", "total")
    }
    needed = {"season", "week", "game_id", "actual_home_margin", "actual_total",
              "baseline_home_margin", "baseline_total", "spread_line",
              "total_line", "margin_residual", "total_residual",
              *(f for names in features.values() for f in names)}
    require_columns(frame, needed, "efficiency_market_frame")
    if frame.select("season", "week", "game_id").unique().height != frame.height:
        raise DataContractError("duplicate efficiency-market forecast keys")
    results = {}
    for target in ("margin", "total"):
        truth = "actual_home_margin" if target == "margin" else "actual_total"
        base = "baseline_home_margin" if target == "margin" else "baseline_total"
        market = "spread_line" if target == "margin" else "total_line"
        residual = f"{target}_residual"
        columns = (*features[target], truth, base, residual, market)
        folds = []
        for season in TEST_SEASONS:
            train = frame.filter(pl.col("season") < season).filter(
                pl.all_horizontal([pl.col(c).is_finite() for c in (*features[target], residual)])
            )
            test = frame.filter(pl.col("season") == season).filter(
                pl.all_horizontal([pl.col(c).is_finite() for c in columns])
            )
            if train.height < MIN_TRAIN or test.height < MIN_TEST:
                folds.append({"season": season, "status": "INSUFFICIENT_SAMPLE",
                              "train_games": train.height, "test_games": test.height})
                continue
            model = FeatureRidgeModel(features[target], ALPHA).fit(train, residual)
            actual = test[truth].to_numpy().astype(float)
            baseline = test[base].to_numpy().astype(float)
            market_est = test[market].to_numpy().astype(float)
            candidate = baseline + BLEND * model.predict(test)
            baseline_mae, challenger_mae, market_mae = [
                float(np.mean(np.abs(prediction - actual)))
                for prediction in (baseline, candidate, market_est)
            ]
            baseline_rmse = float(np.sqrt(np.mean((baseline - actual) ** 2)))
            challenger_rmse = float(np.sqrt(np.mean((candidate - actual) ** 2)))
            folds.append({
                "season": season, "status": "HISTORICAL_ARCHIVE_RESEARCH",
                "train_games": train.height, "test_games": test.height,
                "baseline_mae": baseline_mae, "challenger_mae": challenger_mae,
                "archive_market_mae": market_mae,
                "baseline_rmse": baseline_rmse, "challenger_rmse": challenger_rmse,
                "beats_baseline": challenger_mae < baseline_mae,
                "beats_archive_market": challenger_mae < market_mae,
            })
        results[target] = {
            "folds": folds,
            "repeatable_market_win": len(folds) == len(TEST_SEASONS) and all(
                row.get("beats_archive_market", False) for row in folds
            ),
        }
    return {
        "status": "RESEARCH_ONLY",
        "specification": "pregame EPA + success + explosives, Ridge100, blend0.25",
        "features_frozen": True,
        "market": "nflverse archive-final, not timestamp-verified close",
        "train_uses_market": False,
        "independent_forward_proof_required": True,
        "production_changed": False,
        "staking_authorized": False,
        "targets": results,
    }
