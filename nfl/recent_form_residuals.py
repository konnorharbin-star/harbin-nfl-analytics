"""Rolling-origin selection for recent-form PBP residual candidates.

This module deliberately treats 2022-2025 as development evidence because those
seasons have already informed model design elsewhere in the repository. A candidate
may be frozen for 2026 shadow evaluation, but these rolling folds are not labeled an
untouched promotion holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .oa_residuals import FeatureRidgeModel
from .recent_form import RECENT_PBP_METRICS, recent_feature_columns

RECENT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "epa": ("epa_per_play",),
    "epa_success": ("epa_per_play", "success_rate"),
    "core": (
        "epa_per_play",
        "success_rate",
        "pass_epa_per_dropback",
        "rush_epa_per_attempt",
    ),
    "all": RECENT_PBP_METRICS,
}


@dataclass(frozen=True)
class RecentFormFoldMetrics:
    season: int
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float


@dataclass(frozen=True)
class RecentFormTargetSelection:
    target: str
    recent_alpha: float | None
    feature_set: str
    ridge_alpha: float | None
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
    shadow_candidate: bool
    folds: tuple[RecentFormFoldMetrics, ...]


@dataclass(frozen=True)
class RecentFormRollingEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin: RecentFormTargetSelection
    total: RecentFormTargetSelection
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["margin"]["folds"] = [asdict(value) for value in self.margin.folds]
        result["total"]["folds"] = [asdict(value) for value in self.total.folds]
        return result


def _target_spec(target: str) -> tuple[str, str]:
    if target == "margin_residual":
        return "actual_home_margin", "baseline_home_margin"
    if target == "total_residual":
        return "actual_total", "baseline_total"
    raise ValueError(f"unsupported residual target: {target}")


def _features(feature_set: str, target: str) -> tuple[str, ...]:
    try:
        metrics = RECENT_FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(f"unknown recent-form feature set: {feature_set}") from exc
    return recent_feature_columns(metrics, target=target)


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _validate_dataset_map(datasets: dict[float, pl.DataFrame]) -> tuple[int, ...]:
    if not datasets:
        raise ValueError("datasets must not be empty")
    reference_identity: list[tuple[object, object, object]] | None = None
    seasons: tuple[int, ...] | None = None
    for alpha, frame in sorted(datasets.items()):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("recent alpha keys must be in (0, 1]")
        require_columns(
            frame,
            {
                "season",
                "week",
                "game_id",
                "margin_residual",
                "total_residual",
                "actual_home_margin",
                "baseline_home_margin",
                "actual_total",
                "baseline_total",
            },
            "recent_form_dataset",
        )
        identity = frame.select(["season", "week", "game_id"]).rows()
        if reference_identity is None:
            reference_identity = identity
            seasons = tuple(sorted(int(value) for value in frame.get_column("season").unique()))
        elif identity != reference_identity:
            raise DataContractError("recent-form alpha datasets do not share identical game rows")
    assert seasons is not None
    return seasons


def _candidate_metrics(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_set: str,
    ridge_alpha: float,
    blend_weight: float,
    min_training_games: int,
) -> tuple[float, float, float, float, tuple[RecentFormFoldMetrics, ...]]:
    actual_col, baseline_col = _target_spec(target)
    features = _features(feature_set, target)
    require_columns(dataset, {actual_col, baseline_col, *features}, "recent_form_candidate")

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[RecentFormFoldMetrics] = []

    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"recent-form fold {season} requires at least {min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(f"recent-form fold {season} has no evaluation games")

        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        if blend_weight == 0.0:
            adjusted = baseline.copy()
        else:
            model = FeatureRidgeModel(features, ridge_alpha).fit(train, target)
            correction = model.predict(test)
            adjusted = baseline + (float(blend_weight) * correction)

        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        folds.append(
            RecentFormFoldMetrics(
                season=season,
                games=test.height,
                baseline_mae=baseline_mae,
                adjusted_mae=adjusted_mae,
                baseline_rmse=baseline_rmse,
                adjusted_rmse=adjusted_rmse,
                mae_improvement=baseline_mae - adjusted_mae,
                rmse_improvement=baseline_rmse - adjusted_rmse,
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
    return baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, tuple(folds)


def _disabled_selection(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    min_training_games: int,
) -> RecentFormTargetSelection:
    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = _candidate_metrics(
        dataset,
        target=target,
        test_seasons=test_seasons,
        feature_set="epa",
        ridge_alpha=100.0,
        blend_weight=0.0,
        min_training_games=min_training_games,
    )
    return RecentFormTargetSelection(
        target=target,
        recent_alpha=None,
        feature_set="disabled",
        ridge_alpha=None,
        blend_weight=0.0,
        games=sum(value.games for value in folds),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=0.0,
        rmse_improvement=0.0,
        positive_folds=0,
        total_folds=len(folds),
        shadow_candidate=False,
        folds=folds,
    )


def _select_target(
    datasets: dict[float, pl.DataFrame],
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_sets: tuple[str, ...],
    ridge_grid: tuple[float, ...],
    blend_grid: tuple[float, ...],
    min_training_games: int,
    min_positive_folds: int,
) -> RecentFormTargetSelection:
    first_dataset = datasets[sorted(datasets)[0]]
    disabled = _disabled_selection(
        first_dataset,
        target=target,
        test_seasons=test_seasons,
        min_training_games=min_training_games,
    )

    candidates: list[
        tuple[
            float,
            int,
            float,
            float,
            float,
            str,
            float,
            float,
            float,
            float,
            tuple[RecentFormFoldMetrics, ...],
        ]
    ] = []
    for recent_alpha, dataset in sorted(datasets.items()):
        for feature_set in feature_sets:
            feature_count = len(_features(feature_set, target))
            for ridge_alpha in ridge_grid:
                for blend_weight in blend_grid:
                    if blend_weight <= 0.0:
                        continue
                    metrics = _candidate_metrics(
                        dataset,
                        target=target,
                        test_seasons=test_seasons,
                        feature_set=feature_set,
                        ridge_alpha=ridge_alpha,
                        blend_weight=blend_weight,
                        min_training_games=min_training_games,
                    )
                    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = metrics
                    positive_folds = sum(
                        value.mae_improvement > 0 and value.rmse_improvement > 0
                        for value in folds
                    )
                    if not (
                        adjusted_mae < baseline_mae
                        and adjusted_rmse < baseline_rmse
                        and positive_folds >= min_positive_folds
                    ):
                        continue
                    objective = adjusted_rmse + (0.10 * adjusted_mae)
                    candidates.append(
                        (
                            objective,
                            feature_count,
                            float(blend_weight),
                            float(ridge_alpha),
                            float(recent_alpha),
                            feature_set,
                            baseline_mae,
                            baseline_rmse,
                            adjusted_mae,
                            adjusted_rmse,
                            folds,
                        )
                    )

    if not candidates:
        return disabled
    candidates.sort(key=lambda item: item[:6])
    (
        _,
        _,
        blend_weight,
        ridge_alpha,
        recent_alpha,
        feature_set,
        baseline_mae,
        baseline_rmse,
        adjusted_mae,
        adjusted_rmse,
        folds,
    ) = candidates[0]
    positive_folds = sum(
        value.mae_improvement > 0 and value.rmse_improvement > 0 for value in folds
    )
    return RecentFormTargetSelection(
        target=target,
        recent_alpha=recent_alpha,
        feature_set=feature_set,
        ridge_alpha=ridge_alpha,
        blend_weight=blend_weight,
        games=sum(value.games for value in folds),
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=baseline_mae - adjusted_mae,
        rmse_improvement=baseline_rmse - adjusted_rmse,
        positive_folds=positive_folds,
        total_folds=len(folds),
        shadow_candidate=True,
        folds=folds,
    )


def evaluate_recent_form_rolling(
    datasets: dict[float, pl.DataFrame],
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    feature_sets: tuple[str, ...] = ("epa", "epa_success", "core", "all"),
    ridge_grid: tuple[float, ...] = (1.0, 10.0, 100.0),
    blend_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    min_training_games: int = 150,
    min_positive_folds: int = 2,
) -> RecentFormRollingEvaluation:
    """Select recent-form research candidates across expanding chronological folds.

    This is development selection, not an untouched holdout. Non-zero selection requires
    aggregate MAE/RMSE improvement and improvement in at least ``min_positive_folds``.
    The canonical fair score remains unchanged; any selected candidate is intended only
    to be frozen for independent 2026 shadow evidence.
    """

    seasons = _validate_dataset_map(datasets)
    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if any(season not in seasons for season in test_seasons):
        raise ValueError("every test season must exist in the dataset map")
    if min_positive_folds < 1 or min_positive_folds > len(test_seasons):
        raise ValueError("min_positive_folds is outside the available fold count")
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")
    if not blend_grid or any(not 0.0 < value <= 1.0 for value in blend_grid):
        raise ValueError("blend_grid must contain values in (0, 1]")

    margin = _select_target(
        datasets,
        target="margin_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
        min_positive_folds=min_positive_folds,
    )
    total = _select_target(
        datasets,
        target="total_residual",
        test_seasons=test_seasons,
        feature_sets=feature_sets,
        ridge_grid=ridge_grid,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
        min_positive_folds=min_positive_folds,
    )
    return RecentFormRollingEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        promotion_eligible=False,
        meaning=(
            "Rolling-origin 2022-2025 development evidence only. A non-zero candidate may "
            "be frozen for 2026 shadow evaluation, but is not promoted from this result."
        ),
    )
