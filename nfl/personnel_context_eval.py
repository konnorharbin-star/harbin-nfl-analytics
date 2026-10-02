"""Fixed-spec rolling evaluation for non-QB NFL personnel residual candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .personnel_context import (
    MARGIN_PERSONNEL_FEATURE_SETS,
    TOTAL_PERSONNEL_FEATURE_SETS,
)
from .schedule_context_eval import ContextRidgeModel


@dataclass(frozen=True)
class PersonnelFoldMetrics:
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
class PersonnelTargetSelection:
    target: str
    feature_set: str
    features: tuple[str, ...]
    ridge_alpha: float | None
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    positive_folds: int
    total_folds: int
    shadow_candidate: bool
    best_tested_feature_set: str
    best_tested_alpha: float
    best_tested_positive_folds: int
    best_tested_adjusted_mae: float
    best_tested_adjusted_rmse: float
    folds: tuple[PersonnelFoldMetrics, ...]


@dataclass(frozen=True)
class PersonnelRollingEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin: PersonnelTargetSelection
    total: PersonnelTargetSelection
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _target_spec(
    target: str,
) -> tuple[dict[str, tuple[str, ...]], str, str]:
    if target == "margin_residual":
        return (
            MARGIN_PERSONNEL_FEATURE_SETS,
            "actual_home_margin",
            "baseline_home_margin",
        )
    if target == "total_residual":
        return (
            TOTAL_PERSONNEL_FEATURE_SETS,
            "actual_total",
            "baseline_total",
        )
    raise ValueError(f"unsupported personnel residual target: {target}")


def _errors(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    error = predicted - actual
    return (
        float(np.mean(np.abs(error))),
        float(sqrt(float(np.mean(np.square(error))))),
    )


def _candidate_metrics(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_set: str,
    ridge_alpha: float,
    min_training_games: int,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[PersonnelFoldMetrics, ...],
]:
    feature_sets, actual_col, baseline_col = _target_spec(target)
    try:
        features = feature_sets[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown personnel feature set {feature_set!r} for {target}"
        ) from exc
    require_columns(
        dataset,
        {"season", actual_col, baseline_col, target, *features},
        "personnel_rolling",
    )

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[PersonnelFoldMetrics] = []

    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"personnel rolling fold {season} requires at least "
                f"{min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(
                f"personnel rolling fold {season} has no evaluation games"
            )

        model = ContextRidgeModel(features, ridge_alpha).fit(
            train,
            target,
        )
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + model.predict(test)
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            PersonnelFoldMetrics(
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
    return (
        baseline_mae,
        baseline_rmse,
        adjusted_mae,
        adjusted_rmse,
        tuple(folds),
    )


def _disabled_selection(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    best_feature_set: str,
    best_alpha: float,
    best_folds: tuple[PersonnelFoldMetrics, ...],
    best_adjusted_mae: float,
    best_adjusted_rmse: float,
) -> PersonnelTargetSelection:
    feature_sets, actual_col, baseline_col = _target_spec(target)
    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    for season in test_seasons:
        test = dataset.filter(pl.col("season") == season)
        if test.is_empty():
            raise DataContractError(
                f"personnel rolling fold {season} has no evaluation games"
            )
        actual_parts.append(
            np.asarray(test.get_column(actual_col), dtype=float)
        )
        baseline_parts.append(
            np.asarray(test.get_column(baseline_col), dtype=float)
        )
    actual = np.concatenate(actual_parts)
    baseline = np.concatenate(baseline_parts)
    baseline_mae, baseline_rmse = _errors(actual, baseline)
    return PersonnelTargetSelection(
        target=target,
        feature_set="disabled",
        features=(),
        ridge_alpha=None,
        games=len(actual),
        baseline_mae=baseline_mae,
        adjusted_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=baseline_rmse,
        mae_improvement=0.0,
        rmse_improvement=0.0,
        positive_folds=0,
        total_folds=len(test_seasons),
        shadow_candidate=False,
        best_tested_feature_set=best_feature_set,
        best_tested_alpha=best_alpha,
        best_tested_positive_folds=sum(fold.passed for fold in best_folds),
        best_tested_adjusted_mae=best_adjusted_mae,
        best_tested_adjusted_rmse=best_adjusted_rmse,
        folds=best_folds,
    )


def _select_target(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_sets: tuple[str, ...],
    ridge_grid: tuple[float, ...],
    min_training_games: int,
) -> PersonnelTargetSelection:
    target_feature_sets, _, _ = _target_spec(target)
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")

    candidates: list[
        tuple[
            float,
            int,
            float,
            str,
            float,
            float,
            float,
            float,
            tuple[PersonnelFoldMetrics, ...],
        ]
    ] = []
    for feature_set in feature_sets:
        if feature_set not in target_feature_sets:
            raise ValueError(
                f"unknown personnel feature set: {feature_set}"
            )
        features = target_feature_sets[feature_set]
        for alpha in ridge_grid:
            (
                baseline_mae,
                baseline_rmse,
                adjusted_mae,
                adjusted_rmse,
                folds,
            ) = _candidate_metrics(
                dataset,
                target=target,
                test_seasons=test_seasons,
                feature_set=feature_set,
                ridge_alpha=alpha,
                min_training_games=min_training_games,
            )
            objective = adjusted_rmse + (0.10 * adjusted_mae)
            candidates.append(
                (
                    objective,
                    len(features),
                    float(alpha),
                    feature_set,
                    baseline_mae,
                    baseline_rmse,
                    adjusted_mae,
                    adjusted_rmse,
                    folds,
                )
            )

    candidates.sort(
        key=lambda item: (item[0], item[1], item[2], item[3])
    )
    best = candidates[0]
    (
        _,
        _,
        best_alpha,
        best_feature_set,
        _,
        _,
        best_adjusted_mae,
        best_adjusted_rmse,
        best_folds,
    ) = best

    eligible = [
        item
        for item in candidates
        if all(fold.passed for fold in item[-1])
        and (item[4] - item[6]) > 0
        and (item[5] - item[7]) > 0
    ]
    if not eligible:
        return _disabled_selection(
            dataset,
            target=target,
            test_seasons=test_seasons,
            best_feature_set=best_feature_set,
            best_alpha=best_alpha,
            best_folds=best_folds,
            best_adjusted_mae=best_adjusted_mae,
            best_adjusted_rmse=best_adjusted_rmse,
        )

    selected = min(
        eligible,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    (
        _,
        _,
        alpha,
        feature_set,
        baseline_mae,
        baseline_rmse,
        adjusted_mae,
        adjusted_rmse,
        folds,
    ) = selected
    features = target_feature_sets[feature_set]
    return PersonnelTargetSelection(
        target=target,
        feature_set=feature_set,
        features=features,
        ridge_alpha=alpha,
        games=sum(fold.games for fold in folds),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=baseline_mae - adjusted_mae,
        rmse_improvement=baseline_rmse - adjusted_rmse,
        positive_folds=sum(fold.passed for fold in folds),
        total_folds=len(folds),
        shadow_candidate=True,
        best_tested_feature_set=best_feature_set,
        best_tested_alpha=best_alpha,
        best_tested_positive_folds=sum(
            fold.passed for fold in best_folds
        ),
        best_tested_adjusted_mae=best_adjusted_mae,
        best_tested_adjusted_rmse=best_adjusted_rmse,
        folds=folds,
    )


def evaluate_fixed_personnel_rolling(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    feature_sets: tuple[str, ...] = ("starters", "groups", "all"),
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 150,
) -> PersonnelRollingEvaluation:
    """Select one constant non-QB personnel spec across all development folds."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if tuple(sorted(test_seasons)) != test_seasons:
        raise ValueError("test_seasons must be chronological")

    required = {
        "season",
        "margin_residual",
        "total_residual",
        "actual_home_margin",
        "baseline_home_margin",
        "actual_total",
        "baseline_total",
    }
    for name in feature_sets:
        required.update(MARGIN_PERSONNEL_FEATURE_SETS[name])
        required.update(TOTAL_PERSONNEL_FEATURE_SETS[name])
    require_columns(dataset, required, "personnel_rolling_dataset")

    seasons = tuple(
        sorted(
            int(value)
            for value in dataset.get_column("season").unique()
        )
    )
    if min(test_seasons) <= min(seasons):
        raise ValueError(
            "each test season requires at least one earlier training season"
        )

    margin = _select_target(
        dataset,
        target="margin_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        min_training_games=min_training_games,
    )
    total = _select_target(
        dataset,
        target="total_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        min_training_games=min_training_games,
    )
    return PersonnelRollingEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        meaning=(
            "A single non-QB personnel feature set and ridge must improve MAE "
            "and RMSE in every 2023-2025 development fold. Any nonzero result "
            "is only eligible for a frozen 2026 SHADOW ledger."
        ),
    )
