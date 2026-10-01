"""Fixed multi-season gate for nonlinear NFL online-state residual candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .nonlinear_state import NONLINEAR_STATE_SPECS, NonlinearStateResidualModel, nonlinear_spec
from .online_state import ONLINE_STATE_FEATURES


@dataclass(frozen=True)
class NonlinearStateFold:
    season: int
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    passed: bool


@dataclass(frozen=True)
class NonlinearStateCandidate:
    state_config: str
    model_spec: str
    blend_weight: float
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    positive_folds: int
    total_folds: int
    passed_all_folds: bool
    folds: tuple[NonlinearStateFold, ...]


@dataclass(frozen=True)
class NonlinearStateTargetEvaluation:
    target: str
    baseline_mae: float
    baseline_rmse: float
    selected: NonlinearStateCandidate | None
    best_tested: NonlinearStateCandidate
    shadow_candidate: bool


@dataclass(frozen=True)
class NonlinearStateRollingEvaluation:
    test_seasons: tuple[int, ...]
    margin: NonlinearStateTargetEvaluation
    total: NonlinearStateTargetEvaluation
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for target in ("margin", "total"):
            target_result = result[target]
            selected = target_result["selected"]
            if selected is not None:
                selected["folds"] = [dict(row) for row in selected["folds"]]
            target_result["best_tested"]["folds"] = [
                dict(row) for row in target_result["best_tested"]["folds"]
            ]
        return result


def _target_spec(target: str) -> tuple[str, str]:
    if target == "margin_residual":
        return "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return "actual_total", "baseline_total"
    raise ValueError(f"unsupported nonlinear state target: {target}")


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _candidate(
    dataset: pl.DataFrame,
    *,
    state_config: str,
    target: str,
    test_seasons: tuple[int, ...],
    model_spec: str,
    blend_weight: float,
    min_training_games: int,
) -> NonlinearStateCandidate:
    actual_col, baseline_col = _target_spec(target)
    require_columns(
        dataset,
        {"season", actual_col, baseline_col, target, *ONLINE_STATE_FEATURES},
        "nonlinear_state_dataset",
    )
    if not 0 < blend_weight <= 1:
        raise ValueError("blend_weight must be in (0, 1]")

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[NonlinearStateFold] = []
    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"nonlinear state fold {season} requires at least "
                f"{min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(f"nonlinear state fold {season} has no evaluation games")

        model = NonlinearStateResidualModel(nonlinear_spec(model_spec)).fit(train, target)
        correction = blend_weight * model.predict(test)
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + correction
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            NonlinearStateFold(
                season=season,
                games=test.height,
                baseline_mae=baseline_mae,
                adjusted_mae=adjusted_mae,
                baseline_rmse=baseline_rmse,
                adjusted_rmse=adjusted_rmse,
                mae_improvement=mae_improvement,
                rmse_improvement=rmse_improvement,
                passed=mae_improvement > 0 and rmse_improvement > 0,
            )
        )
        actual_parts.append(actual)
        baseline_parts.append(baseline)
        adjusted_parts.append(adjusted)

    actual_all = np.concatenate(actual_parts)
    baseline_all = np.concatenate(baseline_parts)
    adjusted_all = np.concatenate(adjusted_parts)
    baseline_mae, baseline_rmse = _errors(actual_all, baseline_all)
    adjusted_mae, adjusted_rmse = _errors(actual_all, adjusted_all)
    positive_folds = sum(fold.passed for fold in folds)
    passed_all_folds = (
        positive_folds == len(folds)
        and adjusted_mae < baseline_mae
        and adjusted_rmse < baseline_rmse
    )
    return NonlinearStateCandidate(
        state_config=state_config,
        model_spec=model_spec,
        blend_weight=float(blend_weight),
        games=sum(fold.games for fold in folds),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=baseline_mae - adjusted_mae,
        rmse_improvement=baseline_rmse - adjusted_rmse,
        positive_folds=positive_folds,
        total_folds=len(folds),
        passed_all_folds=passed_all_folds,
        folds=tuple(folds),
    )


def _evaluate_target(
    datasets: dict[str, pl.DataFrame],
    *,
    target: str,
    test_seasons: tuple[int, ...],
    model_specs: tuple[str, ...],
    blend_grid: tuple[float, ...],
    min_training_games: int,
) -> NonlinearStateTargetEvaluation:
    candidates: list[NonlinearStateCandidate] = []
    for state_config, dataset in sorted(datasets.items()):
        for spec_name in model_specs:
            nonlinear_spec(spec_name)
            for blend_weight in blend_grid:
                candidates.append(
                    _candidate(
                        dataset,
                        state_config=state_config,
                        target=target,
                        test_seasons=test_seasons,
                        model_spec=spec_name,
                        blend_weight=blend_weight,
                        min_training_games=min_training_games,
                    )
                )
    if not candidates:
        raise ValueError("no nonlinear state candidates were evaluated")

    def objective(value: NonlinearStateCandidate) -> float:
        return value.adjusted_rmse + 0.10 * value.adjusted_mae

    best_tested = min(
        candidates,
        key=lambda value: (
            -value.positive_folds,
            objective(value),
            value.state_config,
            value.model_spec,
            value.blend_weight,
        ),
    )
    eligible = [value for value in candidates if value.passed_all_folds]
    selected = (
        min(
            eligible,
            key=lambda value: (
                objective(value),
                value.state_config,
                value.model_spec,
                value.blend_weight,
            ),
        )
        if eligible
        else None
    )
    return NonlinearStateTargetEvaluation(
        target=target,
        baseline_mae=best_tested.baseline_mae,
        baseline_rmse=best_tested.baseline_rmse,
        selected=selected,
        best_tested=best_tested,
        shadow_candidate=selected is not None,
    )


def evaluate_nonlinear_state_rolling(
    datasets: dict[str, pl.DataFrame],
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    model_specs: tuple[str, ...] = tuple(spec.name for spec in NONLINEAR_STATE_SPECS),
    blend_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    min_training_games: int = 150,
) -> NonlinearStateRollingEvaluation:
    """Evaluate fixed nonlinear state candidates while preserving weight-0 fallback."""

    if not datasets:
        raise ValueError("datasets must not be empty")
    if not model_specs:
        raise ValueError("model_specs must not be empty")
    if not blend_grid or any(value <= 0 or value > 1 for value in blend_grid):
        raise ValueError("blend_grid must contain values in (0, 1]")

    for name, dataset in datasets.items():
        require_columns(
            dataset,
            {
                "season",
                "margin_residual",
                "total_residual",
                "actual_home_margin",
                "baseline_home_margin",
                "actual_total",
                "baseline_total",
                *ONLINE_STATE_FEATURES,
            },
            f"nonlinear_state_dataset[{name}]",
        )
        seasons = set(int(value) for value in dataset.get_column("season").unique())
        if any(season not in seasons for season in test_seasons):
            raise ValueError(f"dataset {name} is missing a test season")
        if min(test_seasons) <= min(seasons):
            raise ValueError("test seasons must have at least one earlier training season")

    margin = _evaluate_target(
        datasets,
        target="margin_residual",
        test_seasons=test_seasons,
        model_specs=model_specs,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
    )
    total = _evaluate_target(
        datasets,
        target="total_residual",
        test_seasons=test_seasons,
        model_specs=model_specs,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
    )
    return NonlinearStateRollingEvaluation(
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        meaning=(
            "2022-2025 are development evidence only. A fixed nonlinear state profile/model/"
            "blend must improve MAE and RMSE in every 2023-2025 fold. Weight 0 remains "
            "canonical and 2026 is reserved for prospective evidence."
        ),
    )
