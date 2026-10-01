"""Fixed rolling evaluation for possession/non-possession scoring decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .scoring_decomposition import build_decomposed_walkforward

REMAINDER_SPECS: tuple[tuple[str, float | None], ...] = (
    ("constant", None),
    ("ridge_32", 32.0),
    ("ridge_128", 128.0),
)


@dataclass(frozen=True)
class DecompositionFoldMetrics:
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
class DecompositionCandidateMetrics:
    spec: str
    remainder_ridge: float | None
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
    folds: tuple[DecompositionFoldMetrics, ...]


@dataclass(frozen=True)
class DecompositionEvaluation:
    test_seasons: tuple[int, ...]
    selected_spec: str
    selected_remainder_ridge: float | None
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
    best_tested_spec: str
    best_tested_positive_folds: int
    best_tested_passed: bool
    candidates: tuple[DecompositionCandidateMetrics, ...]
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
        "candidate_home_margin",
        "candidate_total",
    }
    require_columns(frame, required, "decomposition_predictions")
    if frame.is_empty():
        raise DataContractError("cannot score an empty decomposition prediction frame")

    actual_margin = np.asarray(frame.get_column("actual_home_margin"), dtype=float)
    actual_total = np.asarray(frame.get_column("actual_total"), dtype=float)
    baseline_margin = np.asarray(frame.get_column("baseline_home_margin"), dtype=float)
    baseline_total = np.asarray(frame.get_column("baseline_total"), dtype=float)
    candidate_margin = np.asarray(frame.get_column("candidate_home_margin"), dtype=float)
    candidate_total = np.asarray(frame.get_column("candidate_total"), dtype=float)
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
    return (
        metrics[1] < metrics[0]
        and metrics[3] < metrics[2]
        and metrics[5] < metrics[4]
        and metrics[7] < metrics[6]
    )


def _objective(candidate: DecompositionCandidateMetrics) -> float:
    return (
        candidate.candidate_margin_mae / candidate.baseline_margin_mae
        + candidate.candidate_margin_rmse / candidate.baseline_margin_rmse
        + candidate.candidate_total_mae / candidate.baseline_total_mae
        + candidate.candidate_total_rmse / candidate.baseline_total_rmse
    )


def _evaluate_spec(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...],
    spec: str,
    remainder_ridge: float | None,
    start_week: int,
    end_week: int,
) -> DecompositionCandidateMetrics:
    frames: list[pl.DataFrame] = []
    folds: list[DecompositionFoldMetrics] = []
    for season in test_seasons:
        frame = build_decomposed_walkforward(
            schedules,
            pbp,
            season,
            remainder_ridge=remainder_ridge,
            start_week=start_week,
            end_week=end_week,
        )
        metrics = _frame_metrics(frame)
        folds.append(
            DecompositionFoldMetrics(
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
    return DecompositionCandidateMetrics(
        spec=spec,
        remainder_ridge=remainder_ridge,
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


def evaluate_scoring_decomposition(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    remainder_specs: tuple[tuple[str, float | None], ...] = REMAINDER_SPECS,
    start_week: int = 5,
    end_week: int = 18,
) -> DecompositionEvaluation:
    """Retain only one decomposition spec that wins every development fold."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if not remainder_specs:
        raise ValueError("remainder_specs must not be empty")
    names = [name for name, _ in remainder_specs]
    if len(names) != len(set(names)):
        raise ValueError("remainder spec names must be unique")
    for name, ridge in remainder_specs:
        if not name:
            raise ValueError("remainder spec name must not be empty")
        if ridge is not None and ridge <= 0:
            raise ValueError("remainder ridge must be positive")

    candidates = tuple(
        _evaluate_spec(
            schedules,
            pbp,
            test_seasons=test_seasons,
            spec=name,
            remainder_ridge=ridge,
            start_week=start_week,
            end_week=end_week,
        )
        for name, ridge in remainder_specs
    )
    best_tested = min(candidates, key=lambda candidate: (_objective(candidate), candidate.spec))
    eligible = [candidate for candidate in candidates if candidate.passed]
    selected = (
        min(eligible, key=lambda candidate: (_objective(candidate), candidate.spec))
        if eligible
        else None
    )
    reference = candidates[0]
    if selected is None:
        selected_spec = "canonical"
        selected_ridge = None
        selected_margin_mae = reference.baseline_margin_mae
        selected_margin_rmse = reference.baseline_margin_rmse
        selected_total_mae = reference.baseline_total_mae
        selected_total_rmse = reference.baseline_total_rmse
    else:
        selected_spec = selected.spec
        selected_ridge = selected.remainder_ridge
        selected_margin_mae = selected.candidate_margin_mae
        selected_margin_rmse = selected.candidate_margin_rmse
        selected_total_mae = selected.candidate_total_mae
        selected_total_rmse = selected.candidate_total_rmse

    return DecompositionEvaluation(
        test_seasons=test_seasons,
        selected_spec=selected_spec,
        selected_remainder_ridge=selected_ridge,
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
        best_tested_spec=best_tested.spec,
        best_tested_positive_folds=best_tested.positive_folds,
        best_tested_passed=best_tested.passed,
        candidates=candidates,
        canonical_score_change_enabled=False,
        promotion_eligible=False,
        meaning=(
            "2023-2025 are development folds only. One fixed possession/remainder specification "
            "must improve margin and total MAE/RMSE in every fold and in aggregate. Canonical "
            "direct-score projections remain active; any survivor may only enter a frozen 2026 "
            "shadow experiment."
        ),
    )
