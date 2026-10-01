"""Statistical evidence gates for forward shadow model comparisons.

A tiny forward sample can move MAE/RMSE by chance. This module keeps shadow-model
promotion conservative by bootstrapping paired game errors: baseline and candidate
are always resampled on the same games. A target cannot earn promotion evidence until
it has enough forward games and the lower confidence bounds for both MAE and RMSE
improvement are positive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ShadowStatus = Literal[
    "SHADOW_INSUFFICIENT_SAMPLE",
    "SHADOW_FAILING",
    "SHADOW_INCONCLUSIVE",
    "PROMOTION_EVIDENCE",
]


@dataclass(frozen=True)
class BootstrapInterval:
    point_improvement: float
    lower: float
    upper: float
    probability_improvement: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TargetShadowEvidence:
    games: int
    minimum_games: int
    confidence: float
    mae: BootstrapInterval
    rmse: BootstrapInterval
    status: ShadowStatus

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_arrays(
    actual: np.ndarray,
    baseline: np.ndarray,
    adjusted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(np.asarray(array, dtype=float).reshape(-1) for array in (actual, baseline, adjusted))
    lengths = {array.shape[0] for array in values}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("actual, baseline, and adjusted must be non-empty equal-length arrays")
    if not all(np.isfinite(array).all() for array in values):
        raise ValueError("shadow scoring arrays must contain only finite values")
    return values


def _losses(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = predicted - actual
    return np.abs(error), np.square(error)


def _interval(
    samples: np.ndarray,
    *,
    point: float,
    confidence: float,
) -> BootstrapInterval:
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        point_improvement=float(point),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
        probability_improvement=float(np.mean(samples > 0.0)),
    )


def paired_bootstrap_evidence(
    actual: np.ndarray,
    baseline: np.ndarray,
    adjusted: np.ndarray,
    *,
    iterations: int = 5000,
    confidence: float = 0.95,
    minimum_games: int = 128,
    seed: int = 20260930,
) -> TargetShadowEvidence:
    """Return paired bootstrap intervals and a conservative forward-shadow status."""

    if iterations < 500:
        raise ValueError("iterations must be >= 500")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    if minimum_games < 1:
        raise ValueError("minimum_games must be >= 1")

    actual, baseline, adjusted = _validate_arrays(actual, baseline, adjusted)
    baseline_abs, baseline_sq = _losses(actual, baseline)
    adjusted_abs, adjusted_sq = _losses(actual, adjusted)

    point_mae = float(np.mean(baseline_abs) - np.mean(adjusted_abs))
    point_rmse = float(
        np.sqrt(np.mean(baseline_sq)) - np.sqrt(np.mean(adjusted_sq))
    )

    rng = np.random.default_rng(seed)
    games = actual.shape[0]
    mae_samples = np.empty(iterations, dtype=float)
    rmse_samples = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        indexes = rng.integers(0, games, size=games)
        mae_samples[iteration] = float(
            np.mean(baseline_abs[indexes]) - np.mean(adjusted_abs[indexes])
        )
        rmse_samples[iteration] = float(
            np.sqrt(np.mean(baseline_sq[indexes]))
            - np.sqrt(np.mean(adjusted_sq[indexes]))
        )

    mae = _interval(mae_samples, point=point_mae, confidence=confidence)
    rmse = _interval(rmse_samples, point=point_rmse, confidence=confidence)

    if games < minimum_games:
        status: ShadowStatus = "SHADOW_INSUFFICIENT_SAMPLE"
    elif point_mae <= 0.0 or point_rmse <= 0.0:
        status = "SHADOW_FAILING"
    elif mae.lower > 0.0 and rmse.lower > 0.0:
        status = "PROMOTION_EVIDENCE"
    else:
        status = "SHADOW_INCONCLUSIVE"

    return TargetShadowEvidence(
        games=games,
        minimum_games=minimum_games,
        confidence=confidence,
        mae=mae,
        rmse=rmse,
        status=status,
    )
