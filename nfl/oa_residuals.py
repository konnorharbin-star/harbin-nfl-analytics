"""Nested holdout validation for opponent-adjusted PBP residual candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .opponent_adjusted import PBP_METRICS, opponent_adjusted_feature_columns

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "epa": ("epa_per_play",),
    "epa_success": ("epa_per_play", "success_rate"),
    "core": (
        "epa_per_play",
        "success_rate",
        "pass_epa_per_dropback",
        "rush_epa_per_attempt",
    ),
    "all": PBP_METRICS,
}


class FeatureRidgeModel:
    """Standardized ridge regression over an explicit feature tuple."""

    def __init__(self, features: tuple[str, ...], alpha: float) -> None:
        if not features:
            raise ValueError("features must not be empty")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.features = features
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        require_columns(frame, set(self.features), "oa_residual_features")
        matrix = frame.select(list(self.features)).to_numpy().astype(float)
        if not np.isfinite(matrix).all():
            raise DataContractError("opponent-adjusted residual features are non-finite")
        return matrix

    def fit(self, frame: pl.DataFrame, target: str) -> FeatureRidgeModel:
        require_columns(frame, {target}, "oa_residual_training")
        if frame.height < max(24, len(self.features) * 6):
            raise DataContractError("insufficient rows for opponent-adjusted residual model")
        x = self._matrix(frame)
        y = np.asarray(frame.get_column(target), dtype=float)
        if not np.isfinite(y).all():
            raise DataContractError(f"target {target} contains non-finite values")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = (x - mean) / scale
        design = np.column_stack([np.ones(z.shape[0]), z])
        penalty = np.eye(design.shape[1], dtype=float) * self.alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)

        self.mean_ = mean
        self.scale_ = scale
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].astype(float)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.coef_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("opponent-adjusted residual model is not fitted")
        x = self._matrix(frame)
        z = (x - self.mean_) / self.scale_
        return self.intercept_ + z @ self.coef_


@dataclass(frozen=True)
class OATargetMetrics:
    feature_set: str
    alpha: float
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    candidate_pass: bool


@dataclass(frozen=True)
class OANestedEvaluation:
    validation_season: int
    holdout_season: int
    training_games: int
    validation_games: int
    holdout_games: int
    margin: OATargetMetrics
    total: OATargetMetrics

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["margin"] = asdict(self.margin)
        result["total"] = asdict(self.total)
        return result


def _target_spec(target: str) -> tuple[str, str]:
    if target == "margin_residual":
        return "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return "actual_total", "baseline_total"
    raise ValueError(f"unsupported residual target: {target}")


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _features(feature_set: str, target: str) -> tuple[str, ...]:
    try:
        metrics = FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(f"unknown feature set: {feature_set}") from exc
    return opponent_adjusted_feature_columns(metrics, target=target)


def _adjusted_metrics(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    target: str,
    feature_set: str,
    alpha: float,
) -> tuple[float, float, float, float]:
    actual_col, baseline_col = _target_spec(target)
    features = _features(feature_set, target)
    require_columns(test, {actual_col, baseline_col}, "oa_residual_test")

    model = FeatureRidgeModel(features, alpha).fit(train, target)
    correction = model.predict(test)
    actual = np.asarray(test.get_column(actual_col), dtype=float)
    baseline = np.asarray(test.get_column(baseline_col), dtype=float)
    adjusted = baseline + correction

    baseline_mae, baseline_rmse = _errors(actual, baseline)
    adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
    return baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse


def _select_candidate(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    target: str,
    feature_sets: tuple[str, ...],
    alpha_grid: tuple[float, ...],
) -> tuple[str, float]:
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not alpha_grid or any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain positive values")

    scored: list[tuple[float, int, float, str]] = []
    for feature_set in feature_sets:
        feature_count = len(_features(feature_set, target))
        for alpha in alpha_grid:
            _, _, adjusted_mae, adjusted_rmse = _adjusted_metrics(
                train,
                validation,
                target=target,
                feature_set=feature_set,
                alpha=alpha,
            )
            objective = adjusted_rmse + (0.10 * adjusted_mae)
            scored.append((objective, feature_count, float(alpha), feature_set))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, alpha, feature_set = scored[0]
    return feature_set, alpha


def _evaluate_target(
    initial_train: pl.DataFrame,
    validation: pl.DataFrame,
    final_train: pl.DataFrame,
    holdout: pl.DataFrame,
    *,
    target: str,
    feature_sets: tuple[str, ...],
    alpha_grid: tuple[float, ...],
) -> OATargetMetrics:
    feature_set, alpha = _select_candidate(
        initial_train,
        validation,
        target=target,
        feature_sets=feature_sets,
        alpha_grid=alpha_grid,
    )
    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse = _adjusted_metrics(
        final_train,
        holdout,
        target=target,
        feature_set=feature_set,
        alpha=alpha,
    )
    mae_improvement = baseline_mae - adjusted_mae
    rmse_improvement = baseline_rmse - adjusted_rmse
    return OATargetMetrics(
        feature_set=feature_set,
        alpha=alpha,
        games=holdout.height,
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        candidate_pass=mae_improvement > 0 and rmse_improvement > 0,
    )


def evaluate_oa_nested_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    feature_sets: tuple[str, ...] = ("epa", "epa_success", "core", "all"),
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 150,
) -> OANestedEvaluation:
    """Select structure/hyperparameters before scoring the untouched holdout."""

    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")

    required = {
        "season",
        "margin_residual",
        "total_residual",
        "actual_home_margin",
        "baseline_home_margin",
        "actual_total",
        "baseline_total",
    }
    for feature_set in feature_sets:
        required.update(_features(feature_set, "margin_residual"))
        required.update(_features(feature_set, "total_residual"))
    require_columns(dataset, required, "oa_nested_dataset")

    initial_train = dataset.filter(pl.col("season") < validation_season)
    validation = dataset.filter(pl.col("season") == validation_season)
    final_train = dataset.filter(pl.col("season") < holdout_season)
    holdout = dataset.filter(pl.col("season") == holdout_season)

    if initial_train.height < min_training_games:
        raise DataContractError(
            f"opponent-adjusted tuning requires at least {min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain games")

    margin = _evaluate_target(
        initial_train,
        validation,
        final_train,
        holdout,
        target="margin_residual",
        feature_sets=feature_sets,
        alpha_grid=alpha_grid,
    )
    total = _evaluate_target(
        initial_train,
        validation,
        final_train,
        holdout,
        target="total_residual",
        feature_sets=feature_sets,
        alpha_grid=alpha_grid,
    )

    return OANestedEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        training_games=initial_train.height,
        validation_games=validation.height,
        holdout_games=holdout.height,
        margin=margin,
        total=total,
    )
