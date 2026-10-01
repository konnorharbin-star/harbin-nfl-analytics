"""Fixed multi-season evaluation for the factorized fair-score architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .factorized_score import build_factorized_walkforward


@dataclass(frozen=True)
class FactorizedFoldMetrics:
    season: int
    games: int
    baseline_margin_mae: float
    candidate_margin_mae: float
    baseline_margin_rmse: float
    candidate_margin_rmse: float
    baseline_total_mae: float
    candidate_total_mae: float
    baseline_total_rmse: float
    candidate_total_rmse: float
    passed: bool


@dataclass(frozen=True)
class FactorizedCandidateMetrics:
    ridge: float
    games: int
    baseline_margin_mae: float
    candidate_margin_mae: float
    baseline_margin_rmse: float
    candidate_margin_rmse: float
    baseline_total_mae: float
    candidate_total_mae: float
    baseline_total_rmse: float
    candidate_total_rmse: float
    positive_folds: int
    total_folds: int
    passed: bool
    folds: tuple[FactorizedFoldMetrics, ...]


@dataclass(frozen=True)
class FactorizedArchitectureEvaluation:
    test_seasons: tuple[int, ...]
    ridge_grid: tuple[float, ...]
    selected_ridge: float | None
    shadow_candidate: bool
    games: int
    baseline_margin_mae: float
    selected_margin_mae: float
    baseline_margin_rmse: float
    selected_margin_rmse: float
    baseline_total_mae: float
    selected_total_mae: float
    baseline_total_rmse: float
    selected_total_rmse: float
    best_tested_ridge: float
    best_tested_positive_folds: int
    best_tested_passed: bool
    candidates: tuple[FactorizedCandidateMetrics, ...]
    canonical_score_change_enabled: bool
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["candidates"] = []
        for candidate in self.candidates:
            item = asdict(candidate)
            item["folds"] = [asdict(fold) for fold in candidate.folds]
            result["candidates"].append(item)
        return result


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _frame_metrics(frame: pl.DataFrame) -> tuple[float, ...]:
    required = {
        "actual_home_margin",
        "actual_total",
        "baseline_home_margin",
        "baseline_total",
        "factorized_home_margin",
        "factorized_total",
    }
    require_columns(frame, required, "factorized_predictions")
    if frame.is_empty():
        raise DataContractError("cannot score an empty factorized prediction frame")

    actual_margin = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    actual_total = np.asarray(frame.get_column("actual_total"), dtype=float)
    baseline_margin = np.asarray(frame.get_column("baseline_home_margin"), dtype=float)
    baseline_total = np.asarray(frame.get_column("baseline_total"), dtype=float)
    candidate_margin = np.asarray(frame.get_column("factorized_home_margin"), dtype=float)
    candidate_total = np.asarray(frame.get_column("factorized_total"), dtype=float)

    baseline_margin_mae, baseline_margin_rmse = _errors(actual_margin, baseline_margin)
    candidate_margin_mae, candidate_margin_rmse = _errors(actual_margin, candidate_margin)
    baseline_total_mae, baseline_total_rmse = _errors(actual_total, baseline_total)
    candidate_total_mae, candidate_total_rmse = _errors(actual_total, candidate_total)
    return (
        baseline_margin_mae,
        candidate_margin_mae,
        baseline_margin_rmse,
        candidate_margin_rmse,
        baseline_total_mae,
        candidate_total_mae,
        baseline_total_rmse,
        candidate_total_rmse,
    )


def _passes(metrics: tuple[float, ...]) -> bool:
    (
        baseline_margin_mae,
        candidate_margin_mae,
        baseline_margin_rmse,
        candidate_margin_rmse,
        baseline_total_mae,
        candidate_total_mae,
        baseline_total_rmse,
        candidate_total_rmse,
    ) = metrics
    return (
        candidate_margin_mae < baseline_margin_mae
        and candidate_margin_rmse < baseline_margin_rmse
        and candidate_total_mae < baseline_total_mae
        and candidate_total_rmse < baseline_total_rmse
    )


def _objective(candidate: FactorizedCandidateMetrics) -> float:
    return (
        candidate.candidate_margin_mae / candidate.baseline_margin_mae
        + candidate.candidate_margin_rmse / candidate.baseline_margin_rmse
        + candidate.candidate_total_mae / candidate.baseline_total_mae
        + candidate.candidate_total_rmse / candidate.baseline_total_rmse
    )


def _evaluate_ridge(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...],
    ridge: float,
    start_week: int,
    end_week: int,
) -> FactorizedCandidateMetrics:
    frames: list[pl.DataFrame] = []
    folds: list[FactorizedFoldMetrics] = []
    for season in test_seasons:
        frame = build_factorized_walkforward(
            schedules,
            pbp,
            season,
            start_week=start_week,
            end_week=end_week,
            factorized_ridge=ridge,
        )
        metrics = _frame_metrics(frame)
        folds.append(
            FactorizedFoldMetrics(
                season=season,
                games=frame.height,
                baseline_margin_mae=metrics[0],
                candidate_margin_mae=metrics[1],
                baseline_margin_rmse=metrics[2],
                candidate_margin_rmse=metrics[3],
                baseline_total_mae=metrics[4],
                candidate_total_mae=metrics[5],
                baseline_total_rmse=metrics[6],
                candidate_total_rmse=metrics[7],
                passed=_passes(metrics),
            )
        )
        frames.append(frame)

    combined = pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
    metrics = _frame_metrics(combined)
    positive_folds = sum(fold.passed for fold in folds)
    return FactorizedCandidateMetrics(
        ridge=float(ridge),
        games=combined.height,
        baseline_margin_mae=metrics[0],
        candidate_margin_mae=metrics[1],
        baseline_margin_rmse=metrics[2],
        candidate_margin_rmse=metrics[3],
        baseline_total_mae=metrics[4],
        candidate_total_mae=metrics[5],
        baseline_total_rmse=metrics[6],
        candidate_total_rmse=metrics[7],
        positive_folds=positive_folds,
        total_folds=len(folds),
        passed=positive_folds == len(folds) and _passes(metrics),
        folds=tuple(folds),
    )


def evaluate_factorized_architecture(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    ridge_grid: tuple[float, ...] = (2.0, 8.0, 32.0),
    start_week: int = 5,
    end_week: int = 18,
) -> FactorizedArchitectureEvaluation:
    """Test fixed factorized architectures against the canonical direct-score baseline."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")

    candidates = tuple(
        _evaluate_ridge(
            schedules,
            pbp,
            test_seasons=test_seasons,
            ridge=float(ridge),
            start_week=start_week,
            end_week=end_week,
        )
        for ridge in ridge_grid
    )
    best_tested = min(candidates, key=lambda candidate: (_objective(candidate), candidate.ridge))
    eligible = [candidate for candidate in candidates if candidate.passed]
    selected = (
        min(eligible, key=lambda candidate: (_objective(candidate), candidate.ridge))
        if eligible
        else None
    )
    reference = candidates[0]
    if selected is None:
        selected_margin_mae = reference.baseline_margin_mae
        selected_margin_rmse = reference.baseline_margin_rmse
        selected_total_mae = reference.baseline_total_mae
        selected_total_rmse = reference.baseline_total_rmse
        selected_ridge = None
    else:
        selected_margin_mae = selected.candidate_margin_mae
        selected_margin_rmse = selected.candidate_margin_rmse
        selected_total_mae = selected.candidate_total_mae
        selected_total_rmse = selected.candidate_total_rmse
        selected_ridge = selected.ridge

    return FactorizedArchitectureEvaluation(
        test_seasons=test_seasons,
        ridge_grid=tuple(float(value) for value in ridge_grid),
        selected_ridge=selected_ridge,
        shadow_candidate=selected is not None,
        games=reference.games,
        baseline_margin_mae=reference.baseline_margin_mae,
        selected_margin_mae=selected_margin_mae,
        baseline_margin_rmse=reference.baseline_margin_rmse,
        selected_margin_rmse=selected_margin_rmse,
        baseline_total_mae=reference.baseline_total_mae,
        selected_total_mae=selected_total_mae,
        baseline_total_rmse=reference.baseline_total_rmse,
        selected_total_rmse=selected_total_rmse,
        best_tested_ridge=best_tested.ridge,
        best_tested_positive_folds=best_tested.positive_folds,
        best_tested_passed=best_tested.passed,
        candidates=candidates,
        canonical_score_change_enabled=False,
        promotion_eligible=False,
        meaning=(
            "2023-2025 are development folds only. A single factorized ridge must improve margin "
            "and total MAE/RMSE in every fold and in aggregate. The direct-score baseline remains "
            "canonical; any survivor may only justify a frozen 2026 shadow experiment."
        ),
    )
