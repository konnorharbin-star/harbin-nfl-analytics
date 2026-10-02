"""Whole-week isotonic calibration research over NFL logistic win probabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .contracts import DataContractError, require_columns
from .temporal_validation import WholeWeekPartitionMetadata, whole_week_partition
from .win_probability import LogisticWinModel, score_logistic


@dataclass(frozen=True)
class CalibrationMetrics:
    games: int
    brier: float
    log_loss: float
    ece: float


@dataclass(frozen=True)
class IsotonicProbabilityEvaluation:
    logistic_alpha: float
    partition: WholeWeekPartitionMetadata
    calibration_rows: int
    evaluation_rows: int
    raw_logistic: CalibrationMetrics
    isotonic: CalibrationMetrics
    brier_improvement: float
    log_loss_improvement: float
    ece_improvement: float
    candidate_pass: bool
    selection_uses_evaluation: bool
    canonical_probability_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["partition"] = self.partition.to_dict()
        out["raw_logistic"] = asdict(self.raw_logistic)
        out["isotonic"] = asdict(self.isotonic)
        return out


def _arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    require_columns(
        frame,
        {"projected_home_margin", "actual_home_margin"},
        "isotonic_probability",
    )
    non_ties = frame.filter(pl.col("actual_home_margin") != 0)
    if non_ties.height < 32:
        raise DataContractError("isotonic calibration requires at least 32 non-tied games")
    margin = np.asarray(non_ties.get_column("projected_home_margin"), dtype=float)
    outcome = np.asarray(non_ties.get_column("actual_home_margin") > 0, dtype=float)
    return margin, outcome


def _probabilities(model: LogisticWinModel, margin: np.ndarray) -> np.ndarray:
    return np.asarray([model.predict_probability(value) for value in margin], dtype=float)


def _metrics(actual: np.ndarray, probability: np.ndarray) -> CalibrationMetrics:
    predicted = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
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
    return CalibrationMetrics(
        games=int(actual.size),
        brier=brier,
        log_loss=log_loss,
        ece=float(ece),
    )


def _select_alpha(
    core: pl.DataFrame,
    tune: pl.DataFrame,
    alpha_grid: tuple[float, ...],
) -> float:
    if not alpha_grid or any(alpha < 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain non-negative values")
    scored: list[tuple[float, float, float]] = []
    for alpha in alpha_grid:
        model = LogisticWinModel(alpha=alpha).fit(core)
        metrics = score_logistic(tune, model)
        scored.append((metrics.log_loss, metrics.brier, float(alpha)))
    scored.sort()
    return scored[0][2]


def evaluate_isotonic_probability(
    dataset: pl.DataFrame,
    *,
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0),
    min_core_rows: int = 500,
    min_section_rows: int = 90,
) -> IsotonicProbabilityEvaluation:
    """Fit isotonic mapping on calibration only and score untouched evaluation."""

    core, tune, calibration, evaluation, partition = whole_week_partition(
        dataset,
        min_core_rows=min_core_rows,
        min_section_rows=min_section_rows,
    )
    alpha = _select_alpha(core, tune, alpha_grid)

    logistic = LogisticWinModel(alpha=alpha).fit(calibration)
    calibration_margin, calibration_outcome = _arrays(calibration)
    calibration_probability = _probabilities(logistic, calibration_margin)
    if np.unique(np.round(calibration_probability, 10)).size < 8:
        raise DataContractError("logistic calibration probabilities are too degenerate")

    isotonic = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.01,
        y_max=0.99,
    ).fit(calibration_probability, calibration_outcome)

    evaluation_margin, evaluation_outcome = _arrays(evaluation)
    raw_probability = _probabilities(logistic, evaluation_margin)
    isotonic_probability = np.asarray(isotonic.predict(raw_probability), dtype=float)

    raw = _metrics(evaluation_outcome, raw_probability)
    calibrated = _metrics(evaluation_outcome, isotonic_probability)
    brier_improvement = raw.brier - calibrated.brier
    log_loss_improvement = raw.log_loss - calibrated.log_loss
    ece_improvement = raw.ece - calibrated.ece
    candidate_pass = (
        brier_improvement > 0.0
        and log_loss_improvement > 0.0
        and ece_improvement > 0.0
    )
    return IsotonicProbabilityEvaluation(
        logistic_alpha=alpha,
        partition=partition,
        calibration_rows=calibration.height,
        evaluation_rows=evaluation.height,
        raw_logistic=raw,
        isotonic=calibrated,
        brier_improvement=brier_improvement,
        log_loss_improvement=log_loss_improvement,
        ece_improvement=ece_improvement,
        candidate_pass=candidate_pass,
        selection_uses_evaluation=False,
        canonical_probability_change_enabled=False,
        promotion_eligible=False,
    )
