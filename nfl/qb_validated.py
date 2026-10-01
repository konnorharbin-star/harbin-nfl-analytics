"""Frozen quarterback-adjustment structure selected before the 2026 shadow period.

The structure here is not retuned on 2026 data. Margin uses the EPA-only QB signal;
total uses QB EPA + CPOE. Both ridge penalties were selected on 2024 and cleared the
untouched 2025 holdout before being frozen for forward evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .qb_residuals import QBRidgeModel, qb_feature_columns

VALIDATED_QB_MARGIN_FEATURE_SET = "epa"
VALIDATED_QB_MARGIN_ALPHA = 0.1
VALIDATED_QB_TOTAL_FEATURE_SET = "quality"
VALIDATED_QB_TOTAL_ALPHA = 0.1


@dataclass(frozen=True)
class QBAdjustedMetrics:
    games: int
    baseline_margin_mae: float
    adjusted_margin_mae: float
    baseline_margin_rmse: float
    adjusted_margin_rmse: float
    margin_mae_improvement: float
    margin_rmse_improvement: float
    baseline_total_mae: float
    adjusted_total_mae: float
    baseline_total_rmse: float
    adjusted_total_rmse: float
    total_mae_improvement: float
    total_rmse_improvement: float
    margin_pass: bool
    total_pass: bool


class ValidatedQBAdjustment:
    """Target-specific QB residual adjustment with frozen structure/hyperparameters."""

    def __init__(self) -> None:
        self.margin_model = QBRidgeModel(
            qb_feature_columns(VALIDATED_QB_MARGIN_FEATURE_SET, "margin_residual"),
            VALIDATED_QB_MARGIN_ALPHA,
        )
        self.total_model = QBRidgeModel(
            qb_feature_columns(VALIDATED_QB_TOTAL_FEATURE_SET, "total_residual"),
            VALIDATED_QB_TOTAL_ALPHA,
        )
        self.training_rows = 0

    def fit(self, training_dataset: pl.DataFrame) -> ValidatedQBAdjustment:
        """Fit frozen QB structures on all football data available before deployment."""

        if training_dataset.height < 100:
            raise DataContractError("validated QB adjustment requires at least 100 training games")
        self.margin_model.fit(training_dataset, "margin_residual")
        self.total_model.fit(training_dataset, "total_residual")
        self.training_rows = training_dataset.height
        return self

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append QB-adjusted margin, total, and implied team points."""

        if self.training_rows <= 0:
            raise RuntimeError("validated QB adjustment has not been fitted")
        require_columns(
            frame,
            {"baseline_home_margin", "baseline_total"},
            "qb_adjustment_frame",
        )
        margin_correction = self.margin_model.predict(frame)
        total_correction = self.total_model.predict(frame)
        adjusted_margin = np.asarray(frame.get_column("baseline_home_margin"), dtype=float) + (
            margin_correction
        )
        adjusted_total = (
            np.asarray(frame.get_column("baseline_total"), dtype=float) + total_correction
        )
        adjusted_home = (adjusted_total + adjusted_margin) / 2.0
        adjusted_away = (adjusted_total - adjusted_margin) / 2.0

        return frame.with_columns(
            pl.Series("qb_margin_correction", margin_correction),
            pl.Series("qb_total_correction", total_correction),
            pl.Series("qb_adjusted_home_margin", adjusted_margin),
            pl.Series("qb_adjusted_total", adjusted_total),
            pl.Series("qb_adjusted_home_points", np.maximum(0.0, adjusted_home)),
            pl.Series("qb_adjusted_away_points", np.maximum(0.0, adjusted_away)),
        )


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def score_qb_adjustment(frame: pl.DataFrame) -> QBAdjustedMetrics:
    """Compare a fitted QB-adjusted frame with its independent baseline."""

    required = {
        "actual_home_margin",
        "actual_total",
        "baseline_home_margin",
        "baseline_total",
        "qb_adjusted_home_margin",
        "qb_adjusted_total",
    }
    require_columns(frame, required, "qb_adjusted_scoring")
    if frame.is_empty():
        raise DataContractError("cannot score an empty QB-adjusted frame")

    actual_margin = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    actual_total = np.asarray(frame.get_column("actual_total"), dtype=float)
    baseline_margin = np.asarray(frame.get_column("baseline_home_margin"), dtype=float)
    baseline_total = np.asarray(frame.get_column("baseline_total"), dtype=float)
    adjusted_margin = np.asarray(frame.get_column("qb_adjusted_home_margin"), dtype=float)
    adjusted_total = np.asarray(frame.get_column("qb_adjusted_total"), dtype=float)

    baseline_margin_mae, baseline_margin_rmse = _errors(actual_margin, baseline_margin)
    adjusted_margin_mae, adjusted_margin_rmse = _errors(actual_margin, adjusted_margin)
    baseline_total_mae, baseline_total_rmse = _errors(actual_total, baseline_total)
    adjusted_total_mae, adjusted_total_rmse = _errors(actual_total, adjusted_total)

    margin_mae_improvement = baseline_margin_mae - adjusted_margin_mae
    margin_rmse_improvement = baseline_margin_rmse - adjusted_margin_rmse
    total_mae_improvement = baseline_total_mae - adjusted_total_mae
    total_rmse_improvement = baseline_total_rmse - adjusted_total_rmse

    return QBAdjustedMetrics(
        games=frame.height,
        baseline_margin_mae=baseline_margin_mae,
        adjusted_margin_mae=adjusted_margin_mae,
        baseline_margin_rmse=baseline_margin_rmse,
        adjusted_margin_rmse=adjusted_margin_rmse,
        margin_mae_improvement=margin_mae_improvement,
        margin_rmse_improvement=margin_rmse_improvement,
        baseline_total_mae=baseline_total_mae,
        adjusted_total_mae=adjusted_total_mae,
        baseline_total_rmse=baseline_total_rmse,
        adjusted_total_rmse=adjusted_total_rmse,
        total_mae_improvement=total_mae_improvement,
        total_rmse_improvement=total_rmse_improvement,
        margin_pass=margin_mae_improvement > 0 and margin_rmse_improvement > 0,
        total_pass=total_mae_improvement > 0 and total_rmse_improvement > 0,
    )
