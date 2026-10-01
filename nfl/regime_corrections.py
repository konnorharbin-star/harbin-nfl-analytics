"""Chronological tests of two Stage 21 residual-regime hypotheses.

The candidate corrections are deliberately simple empirical means. They use only
completed earlier development seasons, apply only inside the predeclared regime, and
contain no tunable coefficient or sportsbook input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Callable

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns


@dataclass(frozen=True)
class RegimeCorrectionFold:
    season: int
    train_regime_games: int
    test_regime_games: int
    correction_points: float
    regime_baseline_mae: float
    regime_adjusted_mae: float
    regime_baseline_rmse: float
    regime_adjusted_rmse: float
    regime_mae_improvement: float
    regime_rmse_improvement: float
    overall_baseline_mae: float
    overall_adjusted_mae: float
    overall_baseline_rmse: float
    overall_adjusted_rmse: float
    overall_mae_improvement: float
    overall_rmse_improvement: float
    passed: bool


@dataclass(frozen=True)
class RegimeCorrectionCandidate:
    target: str
    regime_name: str
    games: int
    regime_games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    regime_baseline_mae: float
    regime_adjusted_mae: float
    regime_baseline_rmse: float
    regime_adjusted_rmse: float
    regime_mae_improvement: float
    regime_rmse_improvement: float
    positive_folds: int
    total_folds: int
    shadow_candidate: bool
    folds: tuple[RegimeCorrectionFold, ...]


@dataclass(frozen=True)
class RegimeCorrectionEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin_away_rest: RegimeCorrectionCandidate
    total_low_projection: RegimeCorrectionCandidate
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _margin_away_rest(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("ctx_rest_diff_days") <= -2.0


def _total_low_projection(frame: pl.DataFrame) -> pl.Expr:
    return pl.col("baseline_total") < 42.0


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return (
        float(np.mean(np.abs(error))),
        float(sqrt(float(np.mean(np.square(error))))),
    )


def _target_columns(target: str) -> tuple[str, str, str, Callable[[pl.DataFrame], pl.Expr], str]:
    if target == "margin":
        return (
            "actual_home_margin",
            "baseline_home_margin",
            "margin_residual",
            _margin_away_rest,
            "away_rest_edge",
        )
    if target == "total":
        return (
            "actual_total",
            "baseline_total",
            "total_residual",
            _total_low_projection,
            "projected_total_under_42",
        )
    raise ValueError(f"unsupported correction target: {target}")


def _evaluate_target(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
) -> RegimeCorrectionCandidate:
    actual_col, baseline_col, residual_col, regime_filter, regime_name = _target_columns(target)
    folds: list[RegimeCorrectionFold] = []
    all_actual: list[np.ndarray] = []
    all_baseline: list[np.ndarray] = []
    all_adjusted: list[np.ndarray] = []
    all_regime_actual: list[np.ndarray] = []
    all_regime_baseline: list[np.ndarray] = []
    all_regime_adjusted: list[np.ndarray] = []

    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.is_empty() or test.is_empty():
            raise DataContractError(f"season {season} lacks correction train/test rows")

        train_regime = train.filter(regime_filter(train))
        test_regime = test.filter(regime_filter(test))
        if train_regime.is_empty():
            raise DataContractError(
                f"no earlier-season training rows for {target} regime {regime_name}"
            )
        if test_regime.is_empty():
            raise DataContractError(
                f"no test rows for {target} regime {regime_name} in {season}"
            )

        correction = float(train_regime.get_column(residual_col).mean())
        if not np.isfinite(correction):
            raise DataContractError("regime correction estimate is non-finite")

        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        mask = np.asarray(test.select(regime_filter(test).alias("_match")).get_column("_match"))
        adjusted = baseline.copy()
        adjusted[mask] += correction

        regime_actual = actual[mask]
        regime_baseline = baseline[mask]
        regime_adjusted = adjusted[mask]
        regime_baseline_mae, regime_baseline_rmse = _metrics(
            regime_actual,
            regime_baseline,
        )
        regime_adjusted_mae, regime_adjusted_rmse = _metrics(
            regime_actual,
            regime_adjusted,
        )
        overall_baseline_mae, overall_baseline_rmse = _metrics(actual, baseline)
        overall_adjusted_mae, overall_adjusted_rmse = _metrics(actual, adjusted)

        regime_mae_improvement = regime_baseline_mae - regime_adjusted_mae
        regime_rmse_improvement = regime_baseline_rmse - regime_adjusted_rmse
        overall_mae_improvement = overall_baseline_mae - overall_adjusted_mae
        overall_rmse_improvement = overall_baseline_rmse - overall_adjusted_rmse
        passed = (
            regime_mae_improvement > 0.0
            and regime_rmse_improvement > 0.0
            and overall_mae_improvement > 0.0
            and overall_rmse_improvement > 0.0
        )

        folds.append(
            RegimeCorrectionFold(
                season=season,
                train_regime_games=train_regime.height,
                test_regime_games=test_regime.height,
                correction_points=correction,
                regime_baseline_mae=regime_baseline_mae,
                regime_adjusted_mae=regime_adjusted_mae,
                regime_baseline_rmse=regime_baseline_rmse,
                regime_adjusted_rmse=regime_adjusted_rmse,
                regime_mae_improvement=regime_mae_improvement,
                regime_rmse_improvement=regime_rmse_improvement,
                overall_baseline_mae=overall_baseline_mae,
                overall_adjusted_mae=overall_adjusted_mae,
                overall_baseline_rmse=overall_baseline_rmse,
                overall_adjusted_rmse=overall_adjusted_rmse,
                overall_mae_improvement=overall_mae_improvement,
                overall_rmse_improvement=overall_rmse_improvement,
                passed=passed,
            )
        )
        all_actual.append(actual)
        all_baseline.append(baseline)
        all_adjusted.append(adjusted)
        all_regime_actual.append(regime_actual)
        all_regime_baseline.append(regime_baseline)
        all_regime_adjusted.append(regime_adjusted)

    actual = np.concatenate(all_actual)
    baseline = np.concatenate(all_baseline)
    adjusted = np.concatenate(all_adjusted)
    regime_actual = np.concatenate(all_regime_actual)
    regime_baseline = np.concatenate(all_regime_baseline)
    regime_adjusted = np.concatenate(all_regime_adjusted)

    baseline_mae, baseline_rmse = _metrics(actual, baseline)
    adjusted_mae, adjusted_rmse = _metrics(actual, adjusted)
    regime_baseline_mae, regime_baseline_rmse = _metrics(regime_actual, regime_baseline)
    regime_adjusted_mae, regime_adjusted_rmse = _metrics(regime_actual, regime_adjusted)
    mae_improvement = baseline_mae - adjusted_mae
    rmse_improvement = baseline_rmse - adjusted_rmse
    regime_mae_improvement = regime_baseline_mae - regime_adjusted_mae
    regime_rmse_improvement = regime_baseline_rmse - regime_adjusted_rmse
    positive_folds = sum(int(fold.passed) for fold in folds)
    shadow_candidate = (
        positive_folds == len(folds)
        and mae_improvement > 0.0
        and rmse_improvement > 0.0
        and regime_mae_improvement > 0.0
        and regime_rmse_improvement > 0.0
    )

    return RegimeCorrectionCandidate(
        target=target,
        regime_name=regime_name,
        games=len(actual),
        regime_games=len(regime_actual),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        regime_baseline_mae=regime_baseline_mae,
        regime_adjusted_mae=regime_adjusted_mae,
        regime_baseline_rmse=regime_baseline_rmse,
        regime_adjusted_rmse=regime_adjusted_rmse,
        regime_mae_improvement=regime_mae_improvement,
        regime_rmse_improvement=regime_rmse_improvement,
        positive_folds=positive_folds,
        total_folds=len(folds),
        shadow_candidate=shadow_candidate,
        folds=tuple(folds),
    )


def evaluate_regime_corrections(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
) -> RegimeCorrectionEvaluation:
    """Replay fixed Stage 21 hypotheses using earlier-season empirical corrections."""

    require_columns(
        dataset,
        {
            "season",
            "actual_home_margin",
            "baseline_home_margin",
            "margin_residual",
            "actual_total",
            "baseline_total",
            "total_residual",
            "ctx_rest_diff_days",
        },
        "regime_correction_dataset",
    )
    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    seasons = tuple(sorted(int(value) for value in dataset.get_column("season").unique()))
    if min(test_seasons) <= min(seasons):
        raise ValueError("each test season requires an earlier development season")

    margin = _evaluate_target(dataset, target="margin", test_seasons=test_seasons)
    total = _evaluate_target(dataset, target="total", test_seasons=test_seasons)
    return RegimeCorrectionEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin_away_rest=margin,
        total_low_projection=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
    )
