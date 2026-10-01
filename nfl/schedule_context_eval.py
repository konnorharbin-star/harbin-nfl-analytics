"""Fixed rolling evaluation for schedule/venue residual candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .schedule_context import MARGIN_CONTEXT_FEATURES, TOTAL_CONTEXT_FEATURES


class ContextRidgeModel:
    """Small standardized ridge model for fixed schedule-context features."""

    def __init__(self, feature_names: tuple[str, ...], alpha: float) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.feature_names = feature_names
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        require_columns(frame, set(self.feature_names), "schedule_context_features")
        matrix = frame.select(list(self.feature_names)).to_numpy().astype(float)
        if not np.isfinite(matrix).all():
            raise DataContractError("schedule-context features contain null/non-finite values")
        return matrix

    def fit(self, frame: pl.DataFrame, target: str) -> ContextRidgeModel:
        require_columns(frame, {target}, "schedule_context_training")
        minimum_rows = max(32, len(self.feature_names) * 4)
        if frame.height < minimum_rows:
            raise DataContractError("insufficient rows to fit schedule-context model")
        x = self._matrix(frame)
        y = np.asarray(frame.get_column(target), dtype=float)
        if not np.isfinite(y).all():
            raise DataContractError(f"target {target} contains null/non-finite values")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = (x - mean) / scale
        design = np.column_stack([np.ones(z.shape[0]), z])
        penalty = np.eye(design.shape[1]) * self.alpha
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
            raise RuntimeError("schedule-context model has not been fitted")
        x = self._matrix(frame)
        z = (x - self.mean_) / self.scale_
        return self.intercept_ + z @ self.coef_


@dataclass(frozen=True)
class ContextFoldMetrics:
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
class ContextCandidateMetrics:
    alpha: float
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    positive_folds: int
    total_folds: int
    eligible: bool
    relative_score: float
    folds: tuple[ContextFoldMetrics, ...]


@dataclass(frozen=True)
class ContextTargetEvaluation:
    target: str
    feature_names: tuple[str, ...]
    selected_alpha: float | None
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
    best_tested_alpha: float
    best_tested_positive_folds: int
    best_tested_adjusted_mae: float
    best_tested_adjusted_rmse: float
    best_tested_folds: tuple[ContextFoldMetrics, ...]


@dataclass(frozen=True)
class ScheduleContextEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin: ContextTargetEvaluation
    total: ContextTargetEvaluation
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(sqrt(float(np.mean(np.square(error)))))
    return mae, rmse


def _target_spec(target: str) -> tuple[tuple[str, ...], str, str]:
    if target == "margin_residual":
        return MARGIN_CONTEXT_FEATURES, "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return TOTAL_CONTEXT_FEATURES, "actual_total", "baseline_total"
    raise ValueError(f"unsupported context residual target: {target}")


def _evaluate_candidate(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...],
    target: str,
    alpha: float,
) -> ContextCandidateMetrics:
    feature_names, actual_col, baseline_col = _target_spec(target)
    folds: list[ContextFoldMetrics] = []
    all_actual: list[np.ndarray] = []
    all_baseline: list[np.ndarray] = []
    all_adjusted: list[np.ndarray] = []

    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if test.is_empty():
            raise DataContractError(f"no schedule-context rows for test season {season}")
        model = ContextRidgeModel(feature_names, alpha).fit(train, target)
        correction = model.predict(test)
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + correction
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            ContextFoldMetrics(
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
        all_actual.append(actual)
        all_baseline.append(baseline)
        all_adjusted.append(adjusted)

    actual = np.concatenate(all_actual)
    baseline = np.concatenate(all_baseline)
    adjusted = np.concatenate(all_adjusted)
    baseline_mae, baseline_rmse = _errors(actual, baseline)
    adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
    mae_improvement = baseline_mae - adjusted_mae
    rmse_improvement = baseline_rmse - adjusted_rmse
    positive_folds = sum(int(fold.passed) for fold in folds)
    eligible = (
        positive_folds == len(folds)
        and mae_improvement > 0
        and rmse_improvement > 0
    )
    relative_score = adjusted_mae / baseline_mae + adjusted_rmse / baseline_rmse
    return ContextCandidateMetrics(
        alpha=float(alpha),
        games=len(actual),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        positive_folds=positive_folds,
        total_folds=len(folds),
        eligible=eligible,
        relative_score=relative_score,
        folds=tuple(folds),
    )


def _select_target(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...],
    target: str,
    alpha_grid: tuple[float, ...],
) -> ContextTargetEvaluation:
    if not alpha_grid or any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain positive values")
    feature_names, _, _ = _target_spec(target)
    candidates = [
        _evaluate_candidate(
            dataset,
            test_seasons=test_seasons,
            target=target,
            alpha=alpha,
        )
        for alpha in alpha_grid
    ]
    candidates.sort(key=lambda item: (item.relative_score, item.alpha))
    best = candidates[0]
    eligible = [candidate for candidate in candidates if candidate.eligible]
    selected = min(eligible, key=lambda item: (item.relative_score, item.alpha)) if eligible else None

    if selected is None:
        adjusted_mae = best.baseline_mae
        adjusted_rmse = best.baseline_rmse
        mae_improvement = 0.0
        rmse_improvement = 0.0
        positive_folds = 0
        selected_alpha = None
    else:
        adjusted_mae = selected.adjusted_mae
        adjusted_rmse = selected.adjusted_rmse
        mae_improvement = selected.mae_improvement
        rmse_improvement = selected.rmse_improvement
        positive_folds = selected.positive_folds
        selected_alpha = selected.alpha

    return ContextTargetEvaluation(
        target=target,
        feature_names=feature_names,
        selected_alpha=selected_alpha,
        games=best.games,
        baseline_mae=best.baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=best.baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        positive_folds=positive_folds,
        total_folds=best.total_folds,
        shadow_candidate=selected is not None,
        best_tested_alpha=best.alpha,
        best_tested_positive_folds=best.positive_folds,
        best_tested_adjusted_mae=best.adjusted_mae,
        best_tested_adjusted_rmse=best.adjusted_rmse,
        best_tested_folds=best.folds,
    )


def evaluate_schedule_context(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 100.0),
) -> ScheduleContextEvaluation:
    """Apply one fixed schedule-context feature set across expanding season folds."""

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
            *MARGIN_CONTEXT_FEATURES,
            *TOTAL_CONTEXT_FEATURES,
        },
        "schedule_context_dataset",
    )
    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    seasons = tuple(sorted(int(value) for value in dataset.get_column("season").unique()))
    if min(test_seasons) <= min(seasons):
        raise ValueError("each test season requires at least one earlier training season")

    margin = _select_target(
        dataset,
        test_seasons=test_seasons,
        target="margin_residual",
        alpha_grid=alpha_grid,
    )
    total = _select_target(
        dataset,
        test_seasons=test_seasons,
        target="total_residual",
        alpha_grid=alpha_grid,
    )
    return ScheduleContextEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
    )
