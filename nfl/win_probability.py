"""Chronological calibration for NFL home-win probabilities.

The fair-score home margin is the only predictor. A small logistic calibration model
is fit on historical pregame projections, tuned on a later validation season, and
scored once on an untouched holdout. Sportsbook prices are not inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .probability import GaussianScoreDistribution


@dataclass(frozen=True)
class WinProbabilityMetrics:
    games: int
    brier: float
    log_loss: float
    ece: float
    mid_confidence_games: int
    mid_confidence_mean_probability: float | None
    mid_confidence_actual_rate: float | None
    mid_confidence_gap: float | None


@dataclass(frozen=True)
class WinProbabilityHoldoutEvaluation:
    validation_season: int
    holdout_season: int
    alpha: float
    training_games: int
    validation_games: int
    holdout_games: int
    gaussian: WinProbabilityMetrics
    logistic: WinProbabilityMetrics
    brier_improvement: float
    log_loss_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["gaussian"] = asdict(self.gaussian)
        result["logistic"] = asdict(self.logistic)
        return result


class LogisticWinModel:
    """One-feature regularized logistic calibration over projected home margin."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = float(alpha)
        self.margin_mean: float | None = None
        self.margin_scale: float | None = None
        self.intercept: float | None = None
        self.slope: float | None = None
        self.training_games: int = 0

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _training_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        require_columns(
            frame,
            {"projected_home_margin", "actual_home_margin"},
            "win_probability_training",
        )
        non_ties = frame.filter(pl.col("actual_home_margin") != 0)
        if non_ties.height < 64:
            raise DataContractError("win calibration requires at least 64 non-tied games")
        x = np.asarray(non_ties.get_column("projected_home_margin"), dtype=float)
        y = np.asarray(non_ties.get_column("actual_home_margin") > 0, dtype=float)
        if not np.isfinite(x).all():
            raise DataContractError("projected margins contain non-finite values")
        return x, y

    def fit(self, frame: pl.DataFrame) -> LogisticWinModel:
        x, y = self._training_arrays(frame)
        mean = float(np.mean(x))
        scale = float(np.std(x, ddof=1))
        if scale <= 1e-9:
            raise DataContractError("projected margin variance is degenerate")
        z = (x - mean) / scale
        design = np.column_stack([np.ones(z.size), z])
        beta = np.zeros(2, dtype=float)
        penalty = np.diag([0.0, self.alpha])

        for _ in range(50):
            probability = self._sigmoid(design @ beta)
            weights = np.clip(probability * (1.0 - probability), 1e-6, None)
            gradient = design.T @ (probability - y) + penalty @ beta
            hessian = design.T @ (design * weights[:, None]) + penalty
            step = np.linalg.solve(hessian, gradient)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-10:
                break

        self.margin_mean = mean
        self.margin_scale = scale
        self.intercept = float(beta[0])
        self.slope = float(beta[1])
        self.training_games = x.size
        return self

    def predict_probability(self, projected_home_margin: float) -> float:
        if (
            self.margin_mean is None
            or self.margin_scale is None
            or self.intercept is None
            or self.slope is None
        ):
            raise RuntimeError("win calibration model has not been fitted")
        z = (float(projected_home_margin) - self.margin_mean) / self.margin_scale
        value = np.asarray([self.intercept + self.slope * z], dtype=float)
        return float(self._sigmoid(value)[0])

    def add_probability_column(self, frame: pl.DataFrame) -> pl.DataFrame:
        require_columns(frame, {"projected_home_margin"}, "win_probability_projection")
        return frame.with_columns(
            pl.col("projected_home_margin")
            .map_elements(
                self.predict_probability,
                return_dtype=pl.Float64,
            )
            .alias("home_win_probability")
        )


def _metrics(actual_margin: np.ndarray, probability: np.ndarray) -> WinProbabilityMetrics:
    non_ties = actual_margin != 0
    actual = (actual_margin[non_ties] > 0).astype(float)
    predicted = np.clip(probability[non_ties], 1e-9, 1.0 - 1e-9)
    if actual.size == 0:
        raise DataContractError("win probability evaluation contains no non-tied games")
    brier = float(np.mean(np.square(predicted - actual)))
    log_loss = float(
        -np.mean(actual * np.log(predicted) + (1.0 - actual) * np.log(1.0 - predicted))
    )

    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        upper = edges[index + 1] + (1e-12 if index == 9 else 0.0)
        mask = (predicted >= edges[index]) & (predicted < upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(actual[mask])) - float(np.mean(predicted[mask]))
            )

    confidence = np.maximum(predicted, 1.0 - predicted)
    selected_side_won = np.where(predicted >= 0.5, actual, 1.0 - actual)
    mid = (confidence >= 0.58) & (confidence <= 0.62)
    mid_games = int(np.sum(mid))
    mid_mean = float(np.mean(confidence[mid])) if mid_games else None
    mid_actual = float(np.mean(selected_side_won[mid])) if mid_games else None
    mid_gap = (
        abs(mid_mean - mid_actual)
        if mid_mean is not None and mid_actual is not None
        else None
    )
    return WinProbabilityMetrics(
        games=int(actual.size),
        brier=brier,
        log_loss=log_loss,
        ece=float(ece),
        mid_confidence_games=mid_games,
        mid_confidence_mean_probability=mid_mean,
        mid_confidence_actual_rate=mid_actual,
        mid_confidence_gap=mid_gap,
    )


def score_logistic(frame: pl.DataFrame, model: LogisticWinModel) -> WinProbabilityMetrics:
    require_columns(
        frame,
        {"projected_home_margin", "actual_home_margin"},
        "logistic_win_evaluation",
    )
    margins = np.asarray(frame.get_column("projected_home_margin"), dtype=float)
    actual = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    probability = np.asarray([model.predict_probability(value) for value in margins])
    return _metrics(actual, probability)


def score_gaussian(
    frame: pl.DataFrame,
    model: GaussianScoreDistribution,
) -> WinProbabilityMetrics:
    require_columns(
        frame,
        {"projected_home_margin", "actual_home_margin"},
        "gaussian_win_evaluation",
    )
    margins = np.asarray(frame.get_column("projected_home_margin"), dtype=float)
    actual = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    probability = np.asarray([model.home_win_probability(value) for value in margins])
    return _metrics(actual, probability)


def _select_alpha(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    alpha_grid: tuple[float, ...],
) -> float:
    if not alpha_grid or any(alpha < 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain non-negative values")
    scored: list[tuple[float, float, float]] = []
    for alpha in alpha_grid:
        model = LogisticWinModel(alpha=alpha).fit(train)
        metrics = score_logistic(validation, model)
        scored.append((metrics.log_loss, metrics.brier, float(alpha)))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][2]


def evaluate_win_probability_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 300,
) -> WinProbabilityHoldoutEvaluation:
    """Tune logistic regularization on validation, then score one later holdout."""

    require_columns(dataset, {"season"}, "win_probability_dataset")
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")

    initial_train = dataset.filter(pl.col("season") < validation_season)
    validation = dataset.filter(pl.col("season") == validation_season)
    final_train = dataset.filter(pl.col("season") < holdout_season)
    holdout = dataset.filter(pl.col("season") == holdout_season)
    if initial_train.height < min_training_games:
        raise DataContractError(
            f"win calibration requires at least {min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain games")

    alpha = _select_alpha(initial_train, validation, alpha_grid)
    logistic = LogisticWinModel(alpha=alpha).fit(final_train)
    gaussian = GaussianScoreDistribution().fit(final_train)
    logistic_metrics = score_logistic(holdout, logistic)
    gaussian_metrics = score_gaussian(holdout, gaussian)
    brier_improvement = gaussian_metrics.brier - logistic_metrics.brier
    log_loss_improvement = gaussian_metrics.log_loss - logistic_metrics.log_loss
    return WinProbabilityHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        alpha=alpha,
        training_games=logistic.training_games,
        validation_games=validation.height,
        holdout_games=holdout.height,
        gaussian=gaussian_metrics,
        logistic=logistic_metrics,
        brier_improvement=brier_improvement,
        log_loss_improvement=log_loss_improvement,
        candidate_pass=brier_improvement > 0 and log_loss_improvement > 0,
    )
