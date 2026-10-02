"""Whole-week nested probability validation for the canonical NFL fair score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .probability import GaussianScoreDistribution, ProbabilityCalibrationMetrics, _metrics
from .temporal_validation import WholeWeekPartitionMetadata, whole_week_partition
from .win_probability import LogisticWinModel, score_logistic


@dataclass(frozen=True)
class BinaryProbabilityMetrics:
    games: int
    brier: float
    log_loss: float
    ece: float


@dataclass(frozen=True)
class NestedProbabilityEvaluation:
    margin_scale: float
    total_scale: float
    logistic_alpha: float
    partition: WholeWeekPartitionMetadata
    tuning_rows: int
    calibration_rows: int
    evaluation_rows: int
    evaluation_unscaled_gaussian: ProbabilityCalibrationMetrics
    evaluation_selected_gaussian: ProbabilityCalibrationMetrics
    evaluation_gaussian_win: BinaryProbabilityMetrics
    evaluation_logistic_win: BinaryProbabilityMetrics
    margin_nll_improvement: float
    total_nll_improvement: float
    logistic_brier_improvement_vs_gaussian: float
    logistic_log_loss_improvement_vs_gaussian: float
    selection_uses_evaluation: bool
    canonical_probability_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["partition"] = self.partition.to_dict()
        result["evaluation_unscaled_gaussian"] = asdict(self.evaluation_unscaled_gaussian)
        result["evaluation_selected_gaussian"] = asdict(self.evaluation_selected_gaussian)
        result["evaluation_gaussian_win"] = asdict(self.evaluation_gaussian_win)
        result["evaluation_logistic_win"] = asdict(self.evaluation_logistic_win)
        return result


def _binary_metrics(actual_margin: np.ndarray, probability: np.ndarray) -> BinaryProbabilityMetrics:
    non_ties = actual_margin != 0
    actual = (actual_margin[non_ties] > 0).astype(float)
    predicted = np.clip(np.asarray(probability, dtype=float)[non_ties], 1e-9, 1.0 - 1e-9)
    if actual.size == 0:
        raise ValueError("probability evaluation requires at least one non-tied game")
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
    return BinaryProbabilityMetrics(
        games=int(actual.size),
        brier=brier,
        log_loss=log_loss,
        ece=float(ece),
    )


def _gaussian_win_metrics(
    frame: pl.DataFrame,
    model: GaussianScoreDistribution,
) -> BinaryProbabilityMetrics:
    actual = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    projected = np.asarray(frame.get_column("projected_home_margin"), dtype=float)
    probability = np.asarray(
        [model.home_win_probability(value) for value in projected],
        dtype=float,
    )
    return _binary_metrics(actual, probability)


def _logistic_win_metrics(
    frame: pl.DataFrame,
    model: LogisticWinModel,
) -> BinaryProbabilityMetrics:
    actual = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    projected = np.asarray(frame.get_column("projected_home_margin"), dtype=float)
    probability = np.asarray(
        [model.predict_probability(value) for value in projected],
        dtype=float,
    )
    return _binary_metrics(actual, probability)


def _select_scale(
    core: pl.DataFrame,
    tune: pl.DataFrame,
    *,
    target: str,
    scale_grid: tuple[float, ...],
) -> float:
    if not scale_grid or any(value <= 0 for value in scale_grid):
        raise ValueError("scale_grid must contain positive values")

    scored: list[tuple[float, float]] = []
    for scale in scale_grid:
        if target == "margin":
            model = GaussianScoreDistribution(margin_scale=scale).fit(core)
            objective = _metrics(tune, model).margin_nll
        elif target == "total":
            model = GaussianScoreDistribution(total_scale=scale).fit(core)
            objective = _metrics(tune, model).total_nll
        else:
            raise ValueError(f"unsupported nested probability target: {target}")
        scored.append((float(objective), float(scale)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][1]


def _select_alpha(
    core: pl.DataFrame,
    tune: pl.DataFrame,
    alpha_grid: tuple[float, ...],
) -> float:
    if not alpha_grid or any(value < 0 for value in alpha_grid):
        raise ValueError("alpha_grid must contain non-negative values")

    scored: list[tuple[float, float, float]] = []
    for alpha in alpha_grid:
        model = LogisticWinModel(alpha=alpha).fit(core)
        metrics = score_logistic(tune, model)
        scored.append((metrics.log_loss, metrics.brier, float(alpha)))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][2]


def evaluate_nested_probability(
    dataset: pl.DataFrame,
    *,
    scale_grid: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5),
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0),
    min_core_rows: int = 500,
    min_section_rows: int = 90,
) -> NestedProbabilityEvaluation:
    """Select before calibration and score once on untouched whole-week evaluation."""

    core, tune, calibration, evaluation, partition = whole_week_partition(
        dataset,
        min_core_rows=min_core_rows,
        min_section_rows=min_section_rows,
    )

    margin_scale = _select_scale(core, tune, target="margin", scale_grid=scale_grid)
    total_scale = _select_scale(core, tune, target="total", scale_grid=scale_grid)
    logistic_alpha = _select_alpha(core, tune, alpha_grid)

    # Calibration models consume the dedicated calibration block only. Evaluation
    # outcomes cannot affect selected hyperparameters or fitted calibration state.
    unscaled_gaussian = GaussianScoreDistribution().fit(calibration)
    selected_gaussian = GaussianScoreDistribution(
        margin_scale=margin_scale,
        total_scale=total_scale,
    ).fit(calibration)
    logistic = LogisticWinModel(alpha=logistic_alpha).fit(calibration)

    unscaled_metrics = _metrics(evaluation, unscaled_gaussian)
    selected_metrics = _metrics(evaluation, selected_gaussian)
    gaussian_win = _gaussian_win_metrics(evaluation, selected_gaussian)
    logistic_win = _logistic_win_metrics(evaluation, logistic)

    return NestedProbabilityEvaluation(
        margin_scale=margin_scale,
        total_scale=total_scale,
        logistic_alpha=logistic_alpha,
        partition=partition,
        tuning_rows=tune.height,
        calibration_rows=calibration.height,
        evaluation_rows=evaluation.height,
        evaluation_unscaled_gaussian=unscaled_metrics,
        evaluation_selected_gaussian=selected_metrics,
        evaluation_gaussian_win=gaussian_win,
        evaluation_logistic_win=logistic_win,
        margin_nll_improvement=unscaled_metrics.margin_nll - selected_metrics.margin_nll,
        total_nll_improvement=unscaled_metrics.total_nll - selected_metrics.total_nll,
        logistic_brier_improvement_vs_gaussian=gaussian_win.brier - logistic_win.brier,
        logistic_log_loss_improvement_vs_gaussian=(
            gaussian_win.log_loss - logistic_win.log_loss
        ),
        selection_uses_evaluation=False,
        canonical_probability_change_enabled=False,
        promotion_eligible=False,
    )
