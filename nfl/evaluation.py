"""Evaluation utilities for the independent NFL fair-score baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns


@dataclass(frozen=True)
class BaselineMetrics:
    games: int
    margin_mae: float
    margin_rmse: float
    margin_bias: float
    total_mae: float
    total_rmse: float
    total_bias: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)



def _metric_triplet(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    bias = float(np.mean(errors))
    return mae, rmse, bias



def summarize_baseline(dataset: pl.DataFrame) -> BaselineMetrics:
    """Summarize score-model error without referencing sportsbook lines."""

    required = {
        "actual_home_margin",
        "baseline_home_margin",
        "actual_total",
        "baseline_total",
    }
    require_columns(dataset, required, "walkforward_dataset")
    if dataset.is_empty():
        raise DataContractError("cannot evaluate an empty walk-forward dataset")

    actual_margin = np.asarray(dataset.get_column("actual_home_margin"), dtype=float)
    predicted_margin = np.asarray(dataset.get_column("baseline_home_margin"), dtype=float)
    actual_total = np.asarray(dataset.get_column("actual_total"), dtype=float)
    predicted_total = np.asarray(dataset.get_column("baseline_total"), dtype=float)

    margin_mae, margin_rmse, margin_bias = _metric_triplet(actual_margin, predicted_margin)
    total_mae, total_rmse, total_bias = _metric_triplet(actual_total, predicted_total)

    return BaselineMetrics(
        games=dataset.height,
        margin_mae=margin_mae,
        margin_rmse=margin_rmse,
        margin_bias=margin_bias,
        total_mae=total_mae,
        total_rmse=total_rmse,
        total_bias=total_bias,
    )
