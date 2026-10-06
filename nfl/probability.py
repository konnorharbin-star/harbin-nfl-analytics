"""Chronological NFL score-distribution calibration.

The football fair score remains independent of sportsbook prices. This module models
uncertainty around that score using only prior game outcomes. The legacy Gaussian
model remains available, while the conditional Student-t candidate allows residual
variance and tail weight to depend on the football-only projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, lgamma, log, pi, sqrt

import numpy as np
import polars as pl
from scipy.special import stdtr

from .contracts import DataContractError, require_columns

SQRT_TWO = sqrt(2.0)
Z_50 = 0.6744897501960817
Z_80 = 1.2815515655446004
NORMAL_DF = 1_000_000.0


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / SQRT_TWO))


def _normal_nll(
    errors: np.ndarray,
    mean: float,
    sigma: float | np.ndarray,
) -> float:
    scale = np.asarray(sigma, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("sigma must be > 0")
    centered = errors - mean
    return float(
        np.mean(
            0.5 * np.square(centered / scale)
            + np.log(scale)
            + 0.5 * log(2.0 * pi)
        )
    )


def _student_scale_from_sigma(
    sigma: float | np.ndarray,
    df: float,
) -> np.ndarray:
    values = np.asarray(sigma, dtype=float)
    if df >= NORMAL_DF:
        return values
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be > 2")
    return values * sqrt((df - 2.0) / df)


def _student_t_nll(
    errors: np.ndarray,
    mean: float,
    sigma: float | np.ndarray,
    df: float,
) -> float:
    if df >= NORMAL_DF:
        return _normal_nll(errors, mean, sigma)
    scale = _student_scale_from_sigma(sigma, df)
    centered = (errors - mean) / scale
    constant = (
        lgamma((df + 1.0) / 2.0)
        - lgamma(df / 2.0)
        - 0.5 * log(df * pi)
    )
    log_pdf = (
        constant
        - np.log(scale)
        - ((df + 1.0) / 2.0) * np.log1p(np.square(centered) / df)
    )
    return float(-np.mean(log_pdf))


def _student_t_cdf(value: float, df: float) -> float:
    if df >= NORMAL_DF:
        return _normal_cdf(value)
    return float(stdtr(df, value))


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


@dataclass(frozen=True)
class ConditionalProbabilityHoldoutEvaluation:
    validation_season: int
    holdout_season: int
    training_games: int
    validation_games: int
    holdout_games: int
    baseline_margin_scale: float
    baseline_total_scale: float
    margin_scale: float
    total_scale: float
    margin_df: float
    total_df: float
    margin_strength: float
    total_strength: float
    holdout_baseline: ProbabilityCalibrationMetrics
    holdout_candidate: ProbabilityCalibrationMetrics
    margin_nll_improvement: float
    total_nll_improvement: float
    margin_candidate_pass: bool
    total_candidate_pass: bool
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["holdout_baseline"] = asdict(self.holdout_baseline)
        result["holdout_candidate"] = asdict(self.holdout_candidate)
        return result


class GaussianScoreDistribution:
    """Gaussian residual distribution for projected margin and game total."""

    def __init__(
        self,
        *,
        margin_scale: float = 1.0,
        total_scale: float = 1.0,
    ) -> None:
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
            raise DataContractError(
                "probability calibration requires at least 64 historical games"
            )

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
            raise DataContractError(
                "probability training residuals contain non-finite values"
            )

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
        return (
            self.margin_mean,
            self.margin_sigma,
            self.total_mean,
            self.total_sigma,
        )

    def margin_sigma_for(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        del projected_home_margin, projected_total
        _, sigma, _, _ = self._require_fit()
        return sigma

    def total_sigma_for(
        self,
        projected_total: float,
        projected_home_margin: float | None = None,
    ) -> float:
        del projected_total, projected_home_margin
        _, _, _, sigma = self._require_fit()
        return sigma

    def home_win_probability(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        margin_mean, _, _, _ = self._require_fit()
        sigma = self.margin_sigma_for(
            projected_home_margin,
            projected_total,
        )
        center = float(projected_home_margin) + margin_mean
        return float(1.0 - _normal_cdf((0.0 - center) / sigma))

    def home_cover_probability(
        self,
        projected_home_margin: float,
        home_spread: float,
        projected_total: float | None = None,
    ) -> float:
        """Return P(home margin + home_spread > 0) for an arbitrary spread."""

        margin_mean, _, _, _ = self._require_fit()
        sigma = self.margin_sigma_for(
            projected_home_margin,
            projected_total,
        )
        center = float(projected_home_margin) + margin_mean
        threshold = -float(home_spread)
        return float(1.0 - _normal_cdf((threshold - center) / sigma))

    def over_probability(
        self,
        projected_total: float,
        total_line: float,
        projected_home_margin: float | None = None,
    ) -> float:
        """Return P(game total > total_line) for an arbitrary total threshold."""

        _, _, total_mean, _ = self._require_fit()
        sigma = self.total_sigma_for(
            projected_total,
            projected_home_margin,
        )
        center = float(projected_total) + total_mean
        return float(1.0 - _normal_cdf((float(total_line) - center) / sigma))

    def add_distribution_columns(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Attach fair centers, row-level uncertainty, and win probabilities."""

        require_columns(
            frame,
            {"projected_home_margin", "projected_total"},
            "probability_projection",
        )
        margin_mean, _, total_mean, _ = self._require_fit()
        rows: list[dict[str, float]] = []
        for row in frame.iter_rows(named=True):
            margin = float(row["projected_home_margin"])
            total = float(row["projected_total"])
            rows.append(
                {
                    "fair_home_margin_mean": margin + margin_mean,
                    "fair_total_mean": total + total_mean,
                    "margin_sigma": self.margin_sigma_for(margin, total),
                    "total_sigma": self.total_sigma_for(total, margin),
                    "home_win_probability": self.home_win_probability(
                        margin,
                        total,
                    ),
                }
            )
        return frame.hstack(pl.DataFrame(rows))


class ConditionalStudentTScoreDistribution(GaussianScoreDistribution):
    """Heavy-tailed residual model with shrunk projection-conditional variance.

    Margin uncertainty is conditioned on projected favorite magnitude. Total
    uncertainty is conditioned on the projected scoring environment. Bucket variance
    is strongly shrunk toward the global variance and clipped to prevent sparse
    historical regimes from creating extreme confidence.
    """

    def __init__(
        self,
        *,
        margin_scale: float = 1.0,
        total_scale: float = 1.0,
        margin_df: float = 8.0,
        total_df: float = 8.0,
        margin_strength: float = 0.5,
        total_strength: float = 0.5,
        shrinkage_games: float = 96.0,
    ) -> None:
        super().__init__(
            margin_scale=margin_scale,
            total_scale=total_scale,
        )
        if margin_df <= 2 or total_df <= 2:
            raise ValueError("Student-t degrees of freedom must be > 2")
        if not 0.0 <= margin_strength <= 1.0:
            raise ValueError("margin_strength must be in [0, 1]")
        if not 0.0 <= total_strength <= 1.0:
            raise ValueError("total_strength must be in [0, 1]")
        if shrinkage_games <= 0:
            raise ValueError("shrinkage_games must be > 0")
        self.margin_df = float(margin_df)
        self.total_df = float(total_df)
        self.margin_strength = float(margin_strength)
        self.total_strength = float(total_strength)
        self.shrinkage_games = float(shrinkage_games)
        self.margin_bucket_edges: tuple[float, float] | None = None
        self.total_bucket_edges: tuple[float, float] | None = None
        self.margin_bucket_factors: tuple[float, float, float] | None = None
        self.total_bucket_factors: tuple[float, float, float] | None = None

    @staticmethod
    def _fit_bucket_factors(
        values: np.ndarray,
        errors: np.ndarray,
        mean: float,
        *,
        shrinkage_games: float,
    ) -> tuple[tuple[float, float], tuple[float, float, float]]:
        lower, upper = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        if upper <= lower + 1e-9:
            lower = float(np.quantile(values, 0.25))
            upper = float(np.quantile(values, 0.75))
        global_var = float(np.mean(np.square(errors - mean)))
        if global_var <= 1e-9:
            raise DataContractError("conditional residual variance is degenerate")

        masks = (
            values <= lower,
            (values > lower) & (values <= upper),
            values > upper,
        )
        factors: list[float] = []
        for mask in masks:
            n = int(np.sum(mask))
            bucket_var = (
                float(np.mean(np.square(errors[mask] - mean)))
                if n
                else global_var
            )
            shrunk = (
                n * bucket_var + shrinkage_games * global_var
            ) / (n + shrinkage_games)
            factor = sqrt(max(shrunk, 1e-9) / global_var)
            factors.append(float(np.clip(factor, 0.70, 1.40)))
        return (float(lower), float(upper)), tuple(factors)  # type: ignore[return-value]

    @staticmethod
    def _bucket_factor(
        value: float,
        edges: tuple[float, float],
        factors: tuple[float, float, float],
    ) -> float:
        if value <= edges[0]:
            return factors[0]
        if value <= edges[1]:
            return factors[1]
        return factors[2]

    def fit(
        self,
        frame: pl.DataFrame,
    ) -> ConditionalStudentTScoreDistribution:
        super().fit(frame)
        margin_mean, _, total_mean, _ = self._require_fit()
        projected_margin = np.asarray(
            frame.get_column("projected_home_margin"),
            dtype=float,
        )
        projected_total = np.asarray(
            frame.get_column("projected_total"),
            dtype=float,
        )
        margin_error = np.asarray(
            frame.get_column("actual_home_margin")
            - frame.get_column("projected_home_margin"),
            dtype=float,
        )
        total_error = np.asarray(
            frame.get_column("actual_total") - frame.get_column("projected_total"),
            dtype=float,
        )
        self.margin_bucket_edges, self.margin_bucket_factors = (
            self._fit_bucket_factors(
                np.abs(projected_margin),
                margin_error,
                margin_mean,
                shrinkage_games=self.shrinkage_games,
            )
        )
        self.total_bucket_edges, self.total_bucket_factors = (
            self._fit_bucket_factors(
                projected_total,
                total_error,
                total_mean,
                shrinkage_games=self.shrinkage_games,
            )
        )
        return self

    def margin_sigma_for(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        del projected_total
        _, sigma, _, _ = self._require_fit()
        if self.margin_bucket_edges is None or self.margin_bucket_factors is None:
            raise RuntimeError("conditional margin distribution has not been fitted")
        factor = self._bucket_factor(
            abs(float(projected_home_margin)),
            self.margin_bucket_edges,
            self.margin_bucket_factors,
        )
        return sigma * factor**self.margin_strength

    def total_sigma_for(
        self,
        projected_total: float,
        projected_home_margin: float | None = None,
    ) -> float:
        del projected_home_margin
        _, _, _, sigma = self._require_fit()
        if self.total_bucket_edges is None or self.total_bucket_factors is None:
            raise RuntimeError("conditional total distribution has not been fitted")
        factor = self._bucket_factor(
            float(projected_total),
            self.total_bucket_edges,
            self.total_bucket_factors,
        )
        return sigma * factor**self.total_strength

    def home_win_probability(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        margin_mean, _, _, _ = self._require_fit()
        sigma = self.margin_sigma_for(
            projected_home_margin,
            projected_total,
        )
        scale = float(_student_scale_from_sigma(sigma, self.margin_df))
        center = float(projected_home_margin) + margin_mean
        z = (0.0 - center) / scale
        return float(1.0 - _student_t_cdf(z, self.margin_df))

    def home_cover_probability(
        self,
        projected_home_margin: float,
        home_spread: float,
        projected_total: float | None = None,
    ) -> float:
        margin_mean, _, _, _ = self._require_fit()
        sigma = self.margin_sigma_for(
            projected_home_margin,
            projected_total,
        )
        scale = float(_student_scale_from_sigma(sigma, self.margin_df))
        center = float(projected_home_margin) + margin_mean
        threshold = -float(home_spread)
        return float(
            1.0
            - _student_t_cdf(
                (threshold - center) / scale,
                self.margin_df,
            )
        )

    def over_probability(
        self,
        projected_total: float,
        total_line: float,
        projected_home_margin: float | None = None,
    ) -> float:
        _, _, total_mean, _ = self._require_fit()
        sigma = self.total_sigma_for(
            projected_total,
            projected_home_margin,
        )
        scale = float(_student_scale_from_sigma(sigma, self.total_df))
        center = float(projected_total) + total_mean
        return float(
            1.0
            - _student_t_cdf(
                (float(total_line) - center) / scale,
                self.total_df,
            )
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

    margin_mean, _, total_mean, _ = model._require_fit()
    projected_margin = np.asarray(
        frame.get_column("projected_home_margin"),
        dtype=float,
    )
    projected_total = np.asarray(
        frame.get_column("projected_total"),
        dtype=float,
    )
    actual_margin = np.asarray(
        frame.get_column("actual_home_margin"),
        dtype=float,
    )
    actual_total = np.asarray(
        frame.get_column("actual_total"),
        dtype=float,
    )
    margin_error = actual_margin - projected_margin
    total_error = actual_total - projected_total
    margin_sigmas = np.asarray(
        [
            model.margin_sigma_for(margin, total)
            for margin, total in zip(
                projected_margin,
                projected_total,
                strict=True,
            )
        ],
        dtype=float,
    )
    total_sigmas = np.asarray(
        [
            model.total_sigma_for(total, margin)
            for margin, total in zip(
                projected_margin,
                projected_total,
                strict=True,
            )
        ],
        dtype=float,
    )

    win_probability = np.asarray(
        [
            model.home_win_probability(margin, total)
            for margin, total in zip(
                projected_margin,
                projected_total,
                strict=True,
            )
        ],
        dtype=float,
    )
    home_win = (actual_margin > 0).astype(float)
    non_ties = actual_margin != 0
    if not np.any(non_ties):
        raise DataContractError(
            "probability evaluation contains no non-tied games"
        )
    brier = float(
        np.mean(
            np.square(
                win_probability[non_ties] - home_win[non_ties]
            )
        )
    )

    if isinstance(model, ConditionalStudentTScoreDistribution):
        margin_nll = _student_t_nll(
            margin_error,
            margin_mean,
            margin_sigmas,
            model.margin_df,
        )
        total_nll = _student_t_nll(
            total_error,
            total_mean,
            total_sigmas,
            model.total_df,
        )
        margin_scale = _student_scale_from_sigma(
            margin_sigmas,
            model.margin_df,
        )
        total_scale = _student_scale_from_sigma(
            total_sigmas,
            model.total_df,
        )
        margin_standardized = np.abs(
            (margin_error - margin_mean) / margin_scale
        )
        total_standardized = np.abs(
            (total_error - total_mean) / total_scale
        )
        margin_50_threshold = float(stdtr(model.margin_df, 0.0))
        del margin_50_threshold
        margin_50 = np.asarray(
            [
                _student_t_cdf(value, model.margin_df)
                - _student_t_cdf(-value, model.margin_df)
                for value in margin_standardized
            ]
        )
        margin_80 = margin_50.copy()
        total_50 = np.asarray(
            [
                _student_t_cdf(value, model.total_df)
                - _student_t_cdf(-value, model.total_df)
                for value in total_standardized
            ]
        )
        total_80 = total_50.copy()
        margin_50_coverage = float(np.mean(margin_50 <= 0.50))
        margin_80_coverage = float(np.mean(margin_80 <= 0.80))
        total_50_coverage = float(np.mean(total_50 <= 0.50))
        total_80_coverage = float(np.mean(total_80 <= 0.80))
    else:
        margin_nll = _normal_nll(
            margin_error,
            margin_mean,
            margin_sigmas,
        )
        total_nll = _normal_nll(
            total_error,
            total_mean,
            total_sigmas,
        )
        margin_centered = np.abs(margin_error - margin_mean)
        total_centered = np.abs(total_error - total_mean)
        margin_50_coverage = float(
            np.mean(margin_centered <= Z_50 * margin_sigmas)
        )
        margin_80_coverage = float(
            np.mean(margin_centered <= Z_80 * margin_sigmas)
        )
        total_50_coverage = float(
            np.mean(total_centered <= Z_50 * total_sigmas)
        )
        total_80_coverage = float(
            np.mean(total_centered <= Z_80 * total_sigmas)
        )

    return ProbabilityCalibrationMetrics(
        games=frame.height,
        margin_nll=margin_nll,
        total_nll=total_nll,
        home_win_brier=brier,
        margin_50_coverage=margin_50_coverage,
        margin_80_coverage=margin_80_coverage,
        total_50_coverage=total_50_coverage,
        total_80_coverage=total_80_coverage,
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
            model = GaussianScoreDistribution(
                margin_scale=scale
            ).fit(train)
            objective = _metrics(validation, model).margin_nll
        elif target == "total":
            model = GaussianScoreDistribution(
                total_scale=scale
            ).fit(train)
            objective = _metrics(validation, model).total_nll
        else:
            raise ValueError(
                f"unsupported probability target: {target}"
            )
        scored.append((objective, float(scale)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def _select_conditional_parameters(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    target: str,
    scale_grid: tuple[float, ...],
    df_grid: tuple[float, ...],
    strength_grid: tuple[float, ...],
) -> tuple[float, float, float]:
    scored: list[tuple[float, float, float, float]] = []
    for scale in scale_grid:
        for df in df_grid:
            for strength in strength_grid:
                kwargs: dict[str, float] = {}
                if target == "margin":
                    kwargs.update(
                        {
                            "margin_scale": scale,
                            "margin_df": df,
                            "margin_strength": strength,
                            "total_df": NORMAL_DF,
                            "total_strength": 0.0,
                        }
                    )
                elif target == "total":
                    kwargs.update(
                        {
                            "total_scale": scale,
                            "total_df": df,
                            "total_strength": strength,
                            "margin_df": NORMAL_DF,
                            "margin_strength": 0.0,
                        }
                    )
                else:
                    raise ValueError(
                        f"unsupported conditional target: {target}"
                    )
                model = ConditionalStudentTScoreDistribution(
                    **kwargs
                ).fit(train)
                metrics = _metrics(validation, model)
                objective = (
                    metrics.margin_nll
                    if target == "margin"
                    else metrics.total_nll
                )
                scored.append(
                    (
                        objective,
                        float(scale),
                        float(df),
                        float(strength),
                    )
                )
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
        )
    )
    _, scale, df, strength = scored[0]
    return scale, df, strength


def evaluate_probability_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    scale_grid: tuple[float, ...] = (
        0.75,
        0.9,
        1.0,
        1.1,
        1.25,
        1.5,
    ),
    min_training_games: int = 300,
) -> ProbabilityHoldoutEvaluation:
    """Tune Gaussian dispersion on validation, then score a later holdout."""

    require_columns(dataset, {"season"}, "probability_dataset")
    if validation_season >= holdout_season:
        raise ValueError(
            "validation_season must be earlier than holdout_season"
        )

    initial_train = dataset.filter(
        pl.col("season") < validation_season
    )
    validation = dataset.filter(
        pl.col("season") == validation_season
    )
    final_train = dataset.filter(
        pl.col("season") < holdout_season
    )
    holdout = dataset.filter(
        pl.col("season") == holdout_season
    )
    if initial_train.height < min_training_games:
        raise DataContractError(
            "probability calibration requires at least "
            f"{min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError(
            "validation and holdout seasons must both contain games"
        )

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

    margin_improvement = (
        baseline_metrics.margin_nll
        - calibrated_metrics.margin_nll
    )
    total_improvement = (
        baseline_metrics.total_nll
        - calibrated_metrics.total_nll
    )
    brier_improvement = (
        baseline_metrics.home_win_brier
        - calibrated_metrics.home_win_brier
    )
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


def evaluate_conditional_probability_holdout(
    dataset: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    scale_grid: tuple[float, ...] = (0.9, 1.0, 1.1, 1.25),
    df_grid: tuple[float, ...] = (
        4.0,
        6.0,
        10.0,
        20.0,
        NORMAL_DF,
    ),
    strength_grid: tuple[float, ...] = (0.0, 0.5, 1.0),
    min_training_games: int = 300,
) -> ConditionalProbabilityHoldoutEvaluation:
    """Validate heavy tails and conditional variance on an untouched season."""

    require_columns(dataset, {"season"}, "conditional_probability_dataset")
    if validation_season >= holdout_season:
        raise ValueError(
            "validation_season must be earlier than holdout_season"
        )
    initial_train = dataset.filter(
        pl.col("season") < validation_season
    )
    validation = dataset.filter(
        pl.col("season") == validation_season
    )
    final_train = dataset.filter(
        pl.col("season") < holdout_season
    )
    holdout = dataset.filter(
        pl.col("season") == holdout_season
    )
    if initial_train.height < min_training_games:
        raise DataContractError(
            "conditional probability calibration requires at least "
            f"{min_training_games} pre-validation games"
        )
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError(
            "validation and holdout seasons must both contain games"
        )

    baseline_margin_scale = _select_scale(
        initial_train,
        validation,
        target="margin",
        scale_grid=scale_grid,
    )
    baseline_total_scale = _select_scale(
        initial_train,
        validation,
        target="total",
        scale_grid=scale_grid,
    )
    margin_scale, margin_df, margin_strength = (
        _select_conditional_parameters(
            initial_train,
            validation,
            target="margin",
            scale_grid=scale_grid,
            df_grid=df_grid,
            strength_grid=strength_grid,
        )
    )
    total_scale, total_df, total_strength = (
        _select_conditional_parameters(
            initial_train,
            validation,
            target="total",
            scale_grid=scale_grid,
            df_grid=df_grid,
            strength_grid=strength_grid,
        )
    )

    baseline = GaussianScoreDistribution(
        margin_scale=baseline_margin_scale,
        total_scale=baseline_total_scale,
    ).fit(final_train)
    candidate = ConditionalStudentTScoreDistribution(
        margin_scale=margin_scale,
        total_scale=total_scale,
        margin_df=margin_df,
        total_df=total_df,
        margin_strength=margin_strength,
        total_strength=total_strength,
    ).fit(final_train)
    baseline_metrics = _metrics(holdout, baseline)
    candidate_metrics = _metrics(holdout, candidate)

    margin_improvement = (
        baseline_metrics.margin_nll
        - candidate_metrics.margin_nll
    )
    total_improvement = (
        baseline_metrics.total_nll
        - candidate_metrics.total_nll
    )
    margin_coverage_not_worse = (
        abs(candidate_metrics.margin_80_coverage - 0.80)
        <= abs(baseline_metrics.margin_80_coverage - 0.80) + 0.01
    )
    total_coverage_not_worse = (
        abs(candidate_metrics.total_80_coverage - 0.80)
        <= abs(baseline_metrics.total_80_coverage - 0.80) + 0.01
    )
    margin_pass = margin_improvement > 0 and margin_coverage_not_worse
    total_pass = total_improvement > 0 and total_coverage_not_worse

    return ConditionalProbabilityHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        training_games=final_train.height,
        validation_games=validation.height,
        holdout_games=holdout.height,
        baseline_margin_scale=baseline_margin_scale,
        baseline_total_scale=baseline_total_scale,
        margin_scale=margin_scale,
        total_scale=total_scale,
        margin_df=margin_df,
        total_df=total_df,
        margin_strength=margin_strength,
        total_strength=total_strength,
        holdout_baseline=baseline_metrics,
        holdout_candidate=candidate_metrics,
        margin_nll_improvement=margin_improvement,
        total_nll_improvement=total_improvement,
        margin_candidate_pass=margin_pass,
        total_candidate_pass=total_pass,
        candidate_pass=margin_pass or total_pass,
    )
