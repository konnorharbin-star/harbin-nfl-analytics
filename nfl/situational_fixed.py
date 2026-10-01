"""Fixed-spec rolling evaluation for situational NFL PBP residual candidates.

All candidate structure is development-only. The same feature set and ridge penalty
must be used in every chronological fold. Baseline/zero adjustment is an explicit
selection and 2026 is not consumed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .oa_residuals import FeatureRidgeModel
from .situational import SITUATIONAL_METRICS, situational_feature_columns

SITUATIONAL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "scoring": (
        "red_zone_epa",
        "third_down_success",
        "short_yardage_success",
    ),
    "redzone_pressure": ("red_zone_epa", "sack_rate"),
    "core": ("red_zone_epa", "third_down_success", "sack_rate"),
    "tempo": ("early_down_pass_rate", "plays_per_game"),
    "all": SITUATIONAL_METRICS,
}


@dataclass(frozen=True)
class SituationalFoldMetrics:
    season: int
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    passed: bool


@dataclass(frozen=True)
class SituationalTargetSelection:
    target: str
    feature_set: str
    ridge_alpha: float | None
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    positive_folds: int
    total_folds: int
    shadow_candidate: bool
    folds: tuple[SituationalFoldMetrics, ...]


@dataclass(frozen=True)
class SituationalRollingEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin: SituationalTargetSelection
    total: SituationalTargetSelection
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["margin"]["folds"] = [asdict(value) for value in self.margin.folds]
        result["total"]["folds"] = [asdict(value) for value in self.total.folds]
        return result


def _target_spec(target: str) -> tuple[str, str]:
    if target == "margin_residual":
        return "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return "actual_total", "baseline_total"
    raise ValueError(f"unsupported residual target: {target}")


def _features(feature_set: str, target: str) -> tuple[str, ...]:
    try:
        metrics = SITUATIONAL_FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(f"unknown situational feature set: {feature_set}") from exc
    return situational_feature_columns(metrics, target=target)


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _candidate_metrics(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_set: str,
    ridge_alpha: float,
    min_training_games: int,
) -> tuple[float, float, float, float, tuple[SituationalFoldMetrics, ...]]:
    actual_col, baseline_col = _target_spec(target)
    features = _features(feature_set, target)
    require_columns(dataset, {actual_col, baseline_col, target, *features}, "situational_dataset")

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[SituationalFoldMetrics] = []
    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"situational fold {season} requires at least {min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(f"situational fold {season} has no evaluation games")

        model = FeatureRidgeModel(features, ridge_alpha).fit(train, target)
        correction = model.predict(test)
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + correction
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            SituationalFoldMetrics(
                season=season,
                games=test.height,
                baseline_mae=baseline_mae,
                adjusted_mae=adjusted_mae,
                baseline_rmse=baseline_rmse,
                adjusted_rmse=adjusted_rmse,
                mae_improvement=mae_improvement,
                rmse_improvement=rmse_improvement,
                passed=mae_improvement > 0 and rmse_improvement > 0,
            )
        )
        actual_parts.append(actual)
        baseline_parts.append(baseline)
        adjusted_parts.append(adjusted)

    actual_all = np.concatenate(actual_parts)
    baseline_all = np.concatenate(baseline_parts)
    adjusted_all = np.concatenate(adjusted_parts)
    baseline_mae, baseline_rmse = _errors(actual_all, baseline_all)
    adjusted_mae, adjusted_rmse = _errors(actual_all, adjusted_all)
    return baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, tuple(folds)


def _disabled_selection(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
) -> SituationalTargetSelection:
    actual_col, baseline_col = _target_spec(target)
    test = dataset.filter(pl.col("season").is_in(test_seasons))
    if test.is_empty():
        raise DataContractError("situational disabled selection has no evaluation games")
    actual = np.asarray(test.get_column(actual_col), dtype=float)
    baseline = np.asarray(test.get_column(baseline_col), dtype=float)
    baseline_mae, baseline_rmse = _errors(actual, baseline)
    folds: list[SituationalFoldMetrics] = []
    for season in test_seasons:
        fold = test.filter(pl.col("season") == season)
        fold_actual = np.asarray(fold.get_column(actual_col), dtype=float)
        fold_baseline = np.asarray(fold.get_column(baseline_col), dtype=float)
        mae, rmse = _errors(fold_actual, fold_baseline)
        folds.append(
            SituationalFoldMetrics(
                season=season,
                games=fold.height,
                baseline_mae=mae,
                adjusted_mae=mae,
                baseline_rmse=rmse,
                adjusted_rmse=rmse,
                mae_improvement=0.0,
                rmse_improvement=0.0,
                passed=False,
            )
        )
    return SituationalTargetSelection(
        target=target,
        feature_set="disabled",
        ridge_alpha=None,
        games=test.height,
        baseline_mae=baseline_mae,
        adjusted_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=baseline_rmse,
        mae_improvement=0.0,
        rmse_improvement=0.0,
        positive_folds=0,
        total_folds=len(test_seasons),
        shadow_candidate=False,
        folds=tuple(folds),
    )


def _select_target(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_sets: tuple[str, ...],
    ridge_grid: tuple[float, ...],
    min_training_games: int,
) -> SituationalTargetSelection:
    disabled = _disabled_selection(dataset, target=target, test_seasons=test_seasons)
    candidates: list[tuple[float, int, float, str, tuple[object, ...]]] = []
    for feature_set in feature_sets:
        feature_count = len(_features(feature_set, target))
        for ridge_alpha in ridge_grid:
            metrics = _candidate_metrics(
                dataset,
                target=target,
                test_seasons=test_seasons,
                feature_set=feature_set,
                ridge_alpha=ridge_alpha,
                min_training_games=min_training_games,
            )
            baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = metrics
            if not all(fold.passed for fold in folds):
                continue
            if adjusted_mae >= baseline_mae or adjusted_rmse >= baseline_rmse:
                continue
            objective = adjusted_rmse + (0.10 * adjusted_mae)
            candidates.append(
                (objective, feature_count, float(ridge_alpha), feature_set, metrics)
            )
    if not candidates:
        return disabled

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, ridge_alpha, feature_set, metrics = candidates[0]
    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = metrics
    assert isinstance(folds, tuple)
    return SituationalTargetSelection(
        target=target,
        feature_set=feature_set,
        ridge_alpha=ridge_alpha,
        games=sum(fold.games for fold in folds),
        baseline_mae=float(baseline_mae),
        adjusted_mae=float(adjusted_mae),
        baseline_rmse=float(baseline_rmse),
        adjusted_rmse=float(adjusted_rmse),
        mae_improvement=float(baseline_mae - adjusted_mae),
        rmse_improvement=float(baseline_rmse - adjusted_rmse),
        positive_folds=sum(fold.passed for fold in folds),
        total_folds=len(folds),
        shadow_candidate=True,
        folds=folds,
    )


def evaluate_situational_rolling(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    feature_sets: tuple[str, ...] = ("scoring", "redzone_pressure", "core", "tempo", "all"),
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 150,
) -> SituationalRollingEvaluation:
    """Select only fixed candidates that improve every chronological development fold."""

    require_columns(
        dataset,
        {
            "season",
            "margin_residual",
            "total_residual",
            "actual_home_margin",
            "baseline_home_margin",
            "actual_total",
            "baseline_total",
        },
        "situational_rolling_dataset",
    )
    seasons = tuple(sorted(int(value) for value in dataset.get_column("season").unique()))
    if not test_seasons or min(test_seasons) <= min(seasons):
        raise ValueError("test seasons must have at least one earlier training season")
    if any(season not in seasons for season in test_seasons):
        raise ValueError("every test season must be present in the dataset")
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")

    margin = _select_target(
        dataset,
        target="margin_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        min_training_games=min_training_games,
    )
    total = _select_target(
        dataset,
        target="total_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        min_training_games=min_training_games,
    )
    return SituationalRollingEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        meaning=(
            "2022-2025 are development evidence only. A fixed situational feature/ridge must "
            "improve MAE and RMSE in every 2023-2025 fold. Baseline remains canonical and 2026 "
            "is reserved for prospective evidence."
        ),
    )
