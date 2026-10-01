"""Chronologically validated residual adjustments for the NFL fair-score engine.

The residual layer may correct the independent score baseline using football-only PBP
matchup features. Hyperparameters are selected on a validation season and then
measured once on a later untouched holdout season. Sportsbook prices are not inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns

RESIDUAL_FEATURES = (
    "epa_per_play_matchup_advantage",
    "success_rate_matchup_advantage",
    "pass_epa_per_dropback_matchup_advantage",
    "rush_epa_per_attempt_matchup_advantage",
    "explosive_rate_matchup_advantage",
    "early_down_epa_matchup_advantage",
)


class ResidualRidgeModel:
    """Small standardized ridge model fit only on historical residual targets."""

    def __init__(self, alpha: float = 10.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    @property
    def fitted(self) -> bool:
        return self.coef_ is not None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        require_columns(frame, set(RESIDUAL_FEATURES), "residual_features")
        matrix = frame.select(list(RESIDUAL_FEATURES)).to_numpy().astype(float)
        if not np.isfinite(matrix).all():
            raise DataContractError("residual features contain null/non-finite values")
        return matrix

    def fit(self, frame: pl.DataFrame, target: str) -> ResidualRidgeModel:
        require_columns(frame, {target}, "residual_training")
        if frame.height < len(RESIDUAL_FEATURES) * 4:
            raise DataContractError("insufficient rows to fit residual model")

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
        if not self.fitted or self.mean_ is None or self.scale_ is None or self.intercept_ is None:
            raise RuntimeError("residual model has not been fitted")
        x = self._matrix(frame)
        z = (x - self.mean_) / self.scale_
        assert self.coef_ is not None
        return self.intercept_ + z @ self.coef_


@dataclass(frozen=True)
class TargetHoldoutMetrics:
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
class NestedResidualEvaluation:
    validation_season: int
    holdout_season: int
    training_games: int
    validation_games: int
    holdout_games: int
    margin: TargetHoldoutMetrics
    total: TargetHoldoutMetrics

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["margin"] = asdict(self.margin)
        result["total"] = asdict(self.total)
        return result


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _target_spec(target: str) -> tuple[str, str]:
    if target == "margin_residual":
        return "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return "actual_total", "baseline_total"
    raise ValueError(f"unsupported residual target: {target}")


def _adjusted_metrics(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    target: str,
    alpha: float,
) -> tuple[float, float, float, float]:
    actual_col, baseline_col = _target_spec(target)
    require_columns(test, {actual_col, baseline_col}, "residual_test")

    model = ResidualRidgeModel(alpha=alpha).fit(train, target)
    correction = model.predict(test)
    actual = np.asarray(test.get_column(actual_col), dtype=float)
    baseline = np.asarray(test.get_column(baseline_col), dtype=float)
    adjusted = baseline + correction

    baseline_mae, baseline_rmse = _errors(actual, baseline)
    adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
    return baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse


def _select_alpha(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    target: str,
    alpha_grid: tuple[float, ...],
) -> float:
    if not alpha_grid or any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain positive values")

    scored: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        _, _, _, adjusted_rmse = _adjusted_metrics(
            train,
            validation,
            target=target,
            alpha=alpha,
        )
        scored.append((adjusted_rmse, alpha))
    scored.sort(key=lambda item: (item[0], item[1]))
    return float(scored[0][1])


def _evaluate_target(
    initial_train: pl.DataFrame,
    validation: pl.DataFrame,
    final_train: pl.DataFrame,
    holdout: pl.DataFrame,
    *,
    target: str,
    alpha_grid: tuple[float, ...],
) -> TargetHoldoutMetrics:
    alpha = _select_alpha(initial_train, validation, target=target, alpha_grid=alpha_grid)
    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse = _adjusted_metrics(
        final_train,
        holdout,
        target=target,
        alpha=alpha,
    )
    mae_improvement = baseline_mae - adjusted_mae
    rmse_improvement = baseline_rmse - adjusted_rmse
    return TargetHoldoutMetrics(
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


def evaluate_nested_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int,
    holdout_season: int,
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 150,
) -> NestedResidualEvaluation:
    """Tune before the holdout, refit on all prior seasons, and score the holdout once."""

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
        *RESIDUAL_FEATURES,
    }
    require_columns(dataset, required, "residual_dataset")

    initial_train = dataset.filter(pl.col("season") < validation_season)
    validation = dataset.filter(pl.col("season") == validation_season)
    final_train = dataset.filter(pl.col("season") < holdout_season)
    holdout = dataset.filter(pl.col("season") == holdout_season)

    if initial_train.height < min_training_games:
        raise DataContractError(
            f"nested residual tuning requires at least {min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain games")

    margin = _evaluate_target(
        initial_train,
        validation,
        final_train,
        holdout,
        target="margin_residual",
        alpha_grid=alpha_grid,
    )
    total = _evaluate_target(
        initial_train,
        validation,
        final_train,
        holdout,
        target="total_residual",
        alpha_grid=alpha_grid,
    )

    return NestedResidualEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        training_games=initial_train.height,
        validation_games=validation.height,
        holdout_games=holdout.height,
        margin=margin,
        total=total,
    )
