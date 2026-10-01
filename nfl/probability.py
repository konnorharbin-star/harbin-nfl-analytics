"""Calibrated probability distributions around the independent NFL fair score.

This module converts a football-only point projection into probabilistic margin and
total distributions. Distribution parameters are estimated exclusively from earlier
chronological projection errors. Sportsbook lines are not model inputs; arbitrary
spread/total thresholds may be evaluated only after the distribution is fitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, log, pi, sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns

SQRT_TWO = sqrt(2.0)
Z_50 = 0.6744897501960817
Z_80 = 1.2815515655446004


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / SQRT_TWO))


def _normal_nll(errors: np.ndarray, mean: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    centered = errors - mean
    return float(
        np.mean(0.5 * np.square(centered / sigma) + log(sigma) + 0.5 * log(2.0 * pi))
    )


@dataclass(frozen=True)
class ProbabilityCalibrationMetrics:
    games: int
    margin_nll: float
    total_nll: float
    home_win_brier: float
    margin_50_coverage: float
    margin_80_coverage: float
    total_50_coverage: float
    total_80_coverage: float


@dataclass(frozen=True)
class ProbabilityHoldoutEvaluation:
    validation_season: int
    holdout_season: int
    margin_scale: float
    total_scale: float
    training_games: int
    validation_games: int
    holdout_games: int
    holdout_uncalibrated: ProbabilityCalibrationMetrics
    holdout_calibrated: ProbabilityCalibrationMetrics
    margin_nll_improvement: float
    total_nll_improvement: float
    home_win_brier_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["holdout_uncalibrated"] = asdict(self.holdout_uncalibrated)
        result["holdout_calibrated"] = asdict(self.holdout_calibrated)
        return result


class GaussianScoreDistribution:
    """Gaussian residual distribution for projected margin and game total."""

    def __init__(self, *, margin_scale: float = 1.0, total_scale: float = 1.0) -> None:
        if margin_scale <= 0 or total_scale <= 0:
            raise ValueError("distribution scale multipliers must be > 0")
        self.margin_scale = float(margin_scale)
        self.total_scale = float(total_scale)
        self.margin_mean: float | None = None
        self.margin_sigma: float | None = None
        self.total_mean: float | None = None
        self.total_sigma: float | None = None
        self.training_games: int = 0

    @property
    def fitted(self) -> bool:
        return self.margin_sigma is not None and self.total_sigma is not None

    def fit(self, frame: pl.DataFrame) -> GaussianScoreDistribution:
        required = {
            "projected_home_margin",
            "projected_total",
            "actual_home_margin",
            "actual_total",
        }
        require_columns(frame, required, "probability_training")
        if frame.height < 64:
            raise DataContractError("probability calibration requires at least 64 historical games")

        margin_error = np.asarray(
            frame.get_column("actual_home_margin")
            - frame.get_column("projected_home_margin"),
            dtype=float,
        )
        total_error = np.asarray(
            frame.get_column("actual_total") - frame.get_column("projected_total"),
            dtype=float,
        )
        if not np.isfinite(margin_error).all() or not np.isfinite(total_error).all():
            raise DataContractError("probability training residuals contain non-finite values")

        margin_sigma = float(np.std(margin_error, ddof=1))
        total_sigma = float(np.std(total_error, ddof=1))
        if margin_sigma <= 1e-6 or total_sigma <= 1e-6:
            raise DataContractError("probability residual variance is degenerate")

        self.margin_mean = float(np.mean(margin_error))
        self.margin_sigma = margin_sigma * self.margin_scale
        self.total_mean = float(np.mean(total_error))
        self.total_sigma = total_sigma * self.total_scale
        self.training_games = frame.height
        return self

    def _require_fit(self) -> tuple[float, float, float, float]:
        if (
            self.margin_mean is None
            or self.margin_sigma is None
            or self.total_mean is None
            or self.total_sigma is None
        ):
            raise RuntimeError("score distribution has not been fitted")
        return self.margin_mean, self.margin_sigma, self.total_mean, self.total_sigma

    def home_win_probability(self, projected_home_margin: float) -> float:
        margin_mean, margin_sigma, _, _ = self._require_fit()
        center = float(projected_home_margin) + margin_mean
        return float(1.0 - _normal_cdf((0.0 - center) / margin_sigma))

    def home_cover_probability(self, projected_home_margin: float, home_spread: float) -> float:
        """Return P(home margin + home_spread > 0) for an arbitrary spread."""

        margin_mean, margin_sigma, _, _ = self._require_fit()
        center = float(projected_home_margin) + margin_mean
        threshold = -float(home_spread)
        return float(1.0 - _normal_cdf((threshold - center) / margin_sigma))

    def over_probability(self, projected_total: float, total_line: float) -> float:
        """Return P(game total > total_line) for an arbitrary total threshold."""

        _, _, total_mean, total_sigma = self._require_fit()
        center = float(projected_total) + total_mean
        return float(1.0 - _normal_cdf((float(total_line) - center) / total_sigma))

    def add_distribution_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Attach fair centers, uncertainty, and moneyline win probabilities."""

        require_columns(
            frame,
            {"projected_home_margin", "projected_total"},
            "probability_projection",
        )
        margin_mean, margin_sigma, total_mean, total_sigma = self._require_fit()
        return frame.with_columns(
            (pl.col("projected_home_margin") + margin_mean).alias("fair_home_margin_mean"),
            (pl.col("projected_total") + total_mean).alias("fair_total_mean"),
            pl.lit(margin_sigma).alias("margin_sigma"),
            pl.lit(total_sigma).alias("total_sigma"),
        ).with_columns(
            pl.col("fair_home_margin_mean")
            .map_elements(
                lambda value: 1.0 - _normal_cdf(-float(value) / margin_sigma),
                return_dtype=pl.Float64,
            )
            .alias("home_win_probability")
        )


def _metrics(
    frame: pl.DataFrame,
    model: GaussianScoreDistribution,
) -> ProbabilityCalibrationMetrics:
    required = {
        "projected_home_margin",
        "projected_total",
        "actual_home_margin",
        "actual_total",
    }
    require_columns(frame, required, "probability_evaluation")
    if frame.is_empty():
        raise DataContractError("probability evaluation frame is empty")

    margin_mean, margin_sigma, total_mean, total_sigma = model._require_fit()
    projected_margin = np.asarray(frame.get_column("projected_home_margin"), dtype=float)
    projected_total = np.asarray(frame.get_column("projected_total"), dtype=float)
    actual_margin = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    actual_total = np.asarray(frame.get_column("actual_total"), dtype=float)
    margin_error = actual_margin - projected_margin
    total_error = actual_total - projected_total

    win_probability = np.asarray(
        [model.home_win_probability(value) for value in projected_margin],
        dtype=float,
    )
    home_win = (actual_margin > 0).astype(float)
    non_ties = actual_margin != 0
    if not np.any(non_ties):
        raise DataContractError("probability evaluation contains no non-tied games")
    brier = float(np.mean(np.square(win_probability[non_ties] - home_win[non_ties])))

    margin_centered = np.abs(margin_error - margin_mean)
    total_centered = np.abs(total_error - total_mean)
    return ProbabilityCalibrationMetrics(
        games=frame.height,
        margin_nll=_normal_nll(margin_error, margin_mean, margin_sigma),
        total_nll=_normal_nll(total_error, total_mean, total_sigma),
        home_win_brier=brier,
        margin_50_coverage=float(np.mean(margin_centered <= Z_50 * margin_sigma)),
        margin_80_coverage=float(np.mean(margin_centered <= Z_80 * margin_sigma)),
        total_50_coverage=float(np.mean(total_centered <= Z_50 * total_sigma)),
        total_80_coverage=float(np.mean(total_centered <= Z_80 * total_sigma)),
    )


def _select_scale(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    target: str,
    scale_grid: tuple[float, ...],
) -> float:
    if not scale_grid or any(scale <= 0 for scale in scale_grid):
        raise ValueError("scale_grid must contain positive values")

    scored: list[tuple[float, float]] = []
    for scale in scale_grid:
        if target == "margin":
            model = GaussianScoreDistribution(margin_scale=scale).fit(train)
            objective = _metrics(validation, model).margin_nll
        elif target == "total":
            model = GaussianScoreDistribution(total_scale=scale).fit(train)
            objective = _metrics(validation, model).total_nll
        else:
            raise ValueError(f"unsupported probability target: {target}")
        scored.append((objective, float(scale)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def evaluate_probability_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    scale_grid: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5),
    min_training_games: int = 300,
) -> ProbabilityHoldoutEvaluation:
    """Tune dispersion on validation data, then score a later untouched holdout."""

    require_columns(dataset, {"season"}, "probability_dataset")
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")

    initial_train = dataset.filter(pl.col("season") < validation_season)
    validation = dataset.filter(pl.col("season") == validation_season)
    final_train = dataset.filter(pl.col("season") < holdout_season)
    holdout = dataset.filter(pl.col("season") == holdout_season)
    if initial_train.height < min_training_games:
        raise DataContractError(
            f"probability calibration requires at least {min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain games")

    margin_scale = _select_scale(
        initial_train,
        validation,
        target="margin",
        scale_grid=scale_grid,
    )
    total_scale = _select_scale(
        initial_train,
        validation,
        target="total",
        scale_grid=scale_grid,
    )

    uncalibrated = GaussianScoreDistribution().fit(final_train)
    calibrated = GaussianScoreDistribution(
        margin_scale=margin_scale,
        total_scale=total_scale,
    ).fit(final_train)
    baseline_metrics = _metrics(holdout, uncalibrated)
    calibrated_metrics = _metrics(holdout, calibrated)

    margin_improvement = baseline_metrics.margin_nll - calibrated_metrics.margin_nll
    total_improvement = baseline_metrics.total_nll - calibrated_metrics.total_nll
    brier_improvement = baseline_metrics.home_win_brier - calibrated_metrics.home_win_brier
    candidate_pass = (
        margin_improvement >= 0
        and total_improvement >= 0
        and brier_improvement >= -1e-6
        and (margin_improvement > 0 or total_improvement > 0)
    )
    return ProbabilityHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        margin_scale=margin_scale,
        total_scale=total_scale,
        training_games=final_train.height,
        validation_games=validation.height,
        holdout_games=holdout.height,
        holdout_uncalibrated=baseline_metrics,
        holdout_calibrated=calibrated_metrics,
        margin_nll_improvement=margin_improvement,
        total_nll_improvement=total_improvement,
        home_win_brier_improvement=brier_improvement,
        candidate_pass=candidate_pass,
    )
