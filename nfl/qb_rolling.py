"""Fixed-spec rolling-origin selection for NFL quarterback residual candidates.

Stage 14 showed that the QB architecture was the only margin layer to clear both
nested development folds, but the selected feature set/ridge varied by fold. This
module asks a stricter question: can one QB specification, held fixed across all
2023-2025 development folds, beat the canonical baseline on both MAE and RMSE every
time? Baseline/zero adjustment is always a valid outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .qb_residuals import FEATURE_SETS, QBRidgeModel, qb_feature_columns


@dataclass(frozen=True)
class QBFixedFoldMetrics:
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
class QBFixedSelection:
    target: str
    feature_set: str
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
    folds: tuple[QBFixedFoldMetrics, ...]


@dataclass(frozen=True)
class QBFixedRollingEvaluation:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    margin: QBFixedSelection
    total: QBFixedSelection
    canonical_score_adjustment_enabled: bool
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
    raise ValueError(f"unsupported QB rolling target: {target}")


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _candidate_metrics(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_set: str,
    ridge_alpha: float,
    min_training_games: int,
) -> tuple[float, float, float, float, tuple[QBFixedFoldMetrics, ...]]:
    actual_col, baseline_col = _target_spec(target)
    features = qb_feature_columns(feature_set, target)
    require_columns(dataset, {"season", actual_col, baseline_col, target, *features}, "qb_rolling")

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[QBFixedFoldMetrics] = []

    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"QB rolling fold {season} requires at least {min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(f"QB rolling fold {season} has no evaluation games")

        model = QBRidgeModel(features, ridge_alpha).fit(train, target)
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + model.predict(test)
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            QBFixedFoldMetrics(
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
    return baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, tuple(folds)


def _disabled_selection(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
) -> QBFixedSelection:
    actual_col, baseline_col = _target_spec(target)
    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    folds: list[QBFixedFoldMetrics] = []
    for season in test_seasons:
        test = dataset.filter(pl.col("season") == season)
        if test.is_empty():
            raise DataContractError(f"QB rolling fold {season} has no evaluation games")
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        mae, rmse = _errors(actual, baseline)
        folds.append(
            QBFixedFoldMetrics(
                season=season,
                games=test.height,
                baseline_mae=mae,
                adjusted_mae=mae,
                baseline_rmse=rmse,
                adjusted_rmse=rmse,
                mae_improvement=0.0,
                rmse_improvement=0.0,
                passed=False,
            )
        )
        actual_parts.append(actual)
        baseline_parts.append(baseline)

    actual_all = np.concatenate(actual_parts)
    baseline_all = np.concatenate(baseline_parts)
    mae, rmse = _errors(actual_all, baseline_all)
    return QBFixedSelection(
        target=target,
        feature_set="disabled",
        ridge_alpha=None,
        games=sum(value.games for value in folds),
        baseline_mae=mae,
        adjusted_mae=mae,
        baseline_rmse=rmse,
        adjusted_rmse=rmse,
        mae_improvement=0.0,
        rmse_improvement=0.0,
        positive_folds=0,
        total_folds=len(folds),
        shadow_candidate=False,
        folds=tuple(folds),
    )


def _select_target(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    feature_sets: tuple[str, ...],
    ridge_grid: tuple[float, ...],
    min_training_games: int,
) -> QBFixedSelection:
    disabled = _disabled_selection(dataset, target=target, test_seasons=test_seasons)
    candidates: list[tuple[float, int, float, str, QBFixedSelection]] = []

    for feature_set in feature_sets:
        feature_count = len(qb_feature_columns(feature_set, target))
        for ridge_alpha in ridge_grid:
            baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = (
                _candidate_metrics(
                    dataset,
                    target=target,
                    test_seasons=test_seasons,
                    feature_set=feature_set,
                    ridge_alpha=ridge_alpha,
                    min_training_games=min_training_games,
                )
            )
            positive_folds = sum(value.passed for value in folds)
            if positive_folds != len(folds):
                continue
            mae_improvement = baseline_mae - adjusted_mae
            rmse_improvement = baseline_rmse - adjusted_rmse
            if mae_improvement <= 0 or rmse_improvement <= 0:
                continue
            selection = QBFixedSelection(
                target=target,
                feature_set=feature_set,
                ridge_alpha=float(ridge_alpha),
                games=sum(value.games for value in folds),
                baseline_mae=baseline_mae,
                adjusted_mae=adjusted_mae,
                baseline_rmse=baseline_rmse,
                adjusted_rmse=adjusted_rmse,
                mae_improvement=mae_improvement,
                rmse_improvement=rmse_improvement,
                positive_folds=positive_folds,
                total_folds=len(folds),
                shadow_candidate=True,
                folds=folds,
            )
            objective = adjusted_rmse + (0.10 * adjusted_mae)
            candidates.append(
                (objective, feature_count, float(ridge_alpha), feature_set, selection)
            )

    if not candidates:
        return disabled
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][4]


def evaluate_fixed_qb_rolling(
    dataset: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    feature_sets: tuple[str, ...] = tuple(FEATURE_SETS),
    ridge_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    min_training_games: int = 150,
) -> QBFixedRollingEvaluation:
    """Select one constant QB spec across every development fold or disable it."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if tuple(sorted(test_seasons)) != test_seasons:
        raise ValueError("test_seasons must be chronological")
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")
    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")

    seasons = tuple(sorted(int(value) for value in dataset.get_column("season").unique()))
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
    return QBFixedRollingEvaluation(
        seasons=seasons,
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        meaning=(
            "A fixed QB feature/ridge must improve MAE and RMSE in every 2023-2025 "
            "development fold. Any nonzero result is eligible only to be frozen for "
            "prospective 2026 SHADOW evaluation; the canonical fair score is unchanged."
        ),
    )
