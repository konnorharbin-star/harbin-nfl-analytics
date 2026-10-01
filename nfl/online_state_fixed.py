"""Fixed rolling gate for NCAA-style NFL online-state residual candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .oa_residuals import FeatureRidgeModel
from .online_state import ONLINE_STATE_FEATURES


@dataclass(frozen=True)
class OnlineStateFoldMetrics:
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
class OnlineStateTargetSelection:
    target: str
    state_config: str
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
    folds: tuple[OnlineStateFoldMetrics, ...]


@dataclass(frozen=True)
class OnlineStateRollingEvaluation:
    test_seasons: tuple[int, ...]
    margin: OnlineStateTargetSelection
    total: OnlineStateTargetSelection
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
    raise ValueError(f"unsupported online-state target: {target}")


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(sqrt(float(np.mean(np.square(error)))))


def _candidate_metrics(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...],
    ridge_alpha: float,
    blend_weight: float,
    min_training_games: int,
) -> tuple[float, float, float, float, tuple[OnlineStateFoldMetrics, ...]]:
    actual_col, baseline_col = _target_spec(target)
    require_columns(
        dataset,
        {
            "season",
            actual_col,
            baseline_col,
            target,
            *ONLINE_STATE_FEATURES,
        },
        "online_state_dataset",
    )

    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    folds: list[OnlineStateFoldMetrics] = []
    for season in test_seasons:
        train = dataset.filter(pl.col("season") < season)
        test = dataset.filter(pl.col("season") == season)
        if train.height < min_training_games:
            raise DataContractError(
                f"online-state fold {season} requires at least "
                f"{min_training_games} training games"
            )
        if test.is_empty():
            raise DataContractError(f"online-state fold {season} has no evaluation games")

        model = FeatureRidgeModel(ONLINE_STATE_FEATURES, ridge_alpha).fit(train, target)
        correction = blend_weight * model.predict(test)
        actual = np.asarray(test.get_column(actual_col), dtype=float)
        baseline = np.asarray(test.get_column(baseline_col), dtype=float)
        adjusted = baseline + correction
        baseline_mae, baseline_rmse = _errors(actual, baseline)
        adjusted_mae, adjusted_rmse = _errors(actual, adjusted)
        mae_improvement = baseline_mae - adjusted_mae
        rmse_improvement = baseline_rmse - adjusted_rmse
        folds.append(
            OnlineStateFoldMetrics(
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
) -> OnlineStateTargetSelection:
    actual_col, baseline_col = _target_spec(target)
    test = dataset.filter(pl.col("season").is_in(test_seasons))
    if test.is_empty():
        raise DataContractError("online-state disabled selection has no evaluation games")
    actual = np.asarray(test.get_column(actual_col), dtype=float)
    baseline = np.asarray(test.get_column(baseline_col), dtype=float)
    baseline_mae, baseline_rmse = _errors(actual, baseline)
    folds: list[OnlineStateFoldMetrics] = []
    for season in test_seasons:
        fold = test.filter(pl.col("season") == season)
        fold_actual = np.asarray(fold.get_column(actual_col), dtype=float)
        fold_baseline = np.asarray(fold.get_column(baseline_col), dtype=float)
        mae, rmse = _errors(fold_actual, fold_baseline)
        folds.append(
            OnlineStateFoldMetrics(
                season=season,
                games=fold.height,
                baseline_mae=mae,
                adjusted_mae=mae,
                baseline_rmse=rmse,
                adjusted_rmse=rmse,
                mae_improvement=0.0,
                rmse_improvement=0.0,
                passed=False,
            )
        )
    return OnlineStateTargetSelection(
        target=target,
        state_config="disabled",
        ridge_alpha=None,
        blend_weight=0.0,
        games=test.height,
        baseline_mae=baseline_mae,
        adjusted_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=baseline_rmse,
        mae_improvement=0.0,
        rmse_improvement=0.0,
        positive_folds=0,
        total_folds=len(test_seasons),
        shadow_candidate=False,
        folds=tuple(folds),
    )


def _select_target(
    datasets: dict[str, pl.DataFrame],
    *,
    target: str,
    test_seasons: tuple[int, ...],
    ridge_grid: tuple[float, ...],
    blend_grid: tuple[float, ...],
    min_training_games: int,
) -> OnlineStateTargetSelection:
    if not datasets:
        raise ValueError("datasets must not be empty")
    baseline_dataset = next(iter(datasets.values()))
    disabled = _disabled_selection(
        baseline_dataset,
        target=target,
        test_seasons=test_seasons,
    )

    candidates: list[
        tuple[
            float,
            str,
            float,
            float,
            tuple[float, float, float, float, tuple[OnlineStateFoldMetrics, ...]],
        ]
    ] = []
    for config_name, dataset in sorted(datasets.items()):
        for ridge_alpha in ridge_grid:
            for blend_weight in blend_grid:
                metrics = _candidate_metrics(
                    dataset,
                    target=target,
                    test_seasons=test_seasons,
                    ridge_alpha=ridge_alpha,
                    blend_weight=blend_weight,
                    min_training_games=min_training_games,
                )
                baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = metrics
                if not all(fold.passed for fold in folds):
                    continue
                if adjusted_mae >= baseline_mae or adjusted_rmse >= baseline_rmse:
                    continue
                objective = adjusted_rmse + (0.10 * adjusted_mae)
                candidates.append(
                    (
                        objective,
                        config_name,
                        float(ridge_alpha),
                        float(blend_weight),
                        metrics,
                    )
                )

    if not candidates:
        return disabled

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, config_name, ridge_alpha, blend_weight, metrics = candidates[0]
    baseline_mae, baseline_rmse, adjusted_mae, adjusted_rmse, folds = metrics
    return OnlineStateTargetSelection(
        target=target,
        state_config=config_name,
        ridge_alpha=ridge_alpha,
        blend_weight=blend_weight,
        games=sum(fold.games for fold in folds),
        baseline_mae=float(baseline_mae),
        adjusted_mae=float(adjusted_mae),
        baseline_rmse=float(baseline_rmse),
        adjusted_rmse=float(adjusted_rmse),
        mae_improvement=float(baseline_mae - adjusted_mae),
        rmse_improvement=float(baseline_rmse - adjusted_rmse),
        positive_folds=sum(fold.passed for fold in folds),
        total_folds=len(folds),
        shadow_candidate=True,
        folds=folds,
    )


def evaluate_online_state_rolling(
    datasets: dict[str, pl.DataFrame],
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    ridge_grid: tuple[float, ...] = (1.0, 10.0, 100.0),
    blend_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    min_training_games: int = 150,
) -> OnlineStateRollingEvaluation:
    """Select fixed online-state candidates only when every development fold improves."""

    if not ridge_grid or any(value <= 0 for value in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")
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
            f"online_state_dataset[{name}]",
        )
        seasons = set(int(value) for value in dataset.get_column("season").unique())
        if any(season not in seasons for season in test_seasons):
            raise ValueError(f"dataset {name} is missing a test season")
        if min(test_seasons) <= min(seasons):
            raise ValueError("test seasons must have at least one earlier training season")

    margin = _select_target(
        datasets,
        target="margin_residual",
        test_seasons=test_seasons,
        ridge_grid=ridge_grid,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
    )
    total = _select_target(
        datasets,
        target="total_residual",
        test_seasons=test_seasons,
        ridge_grid=ridge_grid,
        blend_grid=blend_grid,
        min_training_games=min_training_games,
    )
    return OnlineStateRollingEvaluation(
        test_seasons=test_seasons,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        meaning=(
            "2022-2025 are development evidence only. A fixed NCAA-style NFL online-state "
            "profile/ridge/blend must improve MAE and RMSE in every 2023-2025 fold. "
            "Baseline/weight 0 remains canonical and 2026 is reserved for prospective evidence."
        ),
    )
