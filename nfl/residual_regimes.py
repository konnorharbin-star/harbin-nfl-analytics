"""Diagnostic regime analysis for canonical NFL fair-score residuals.

This module does not fit or apply score adjustments. It only identifies whether
predeclared, pregame-known regimes show repeatable signed residual bias across
completed development seasons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns

REGIME_COLUMNS = (
    "regime_margin_band",
    "regime_total_band",
    "regime_season_phase",
    "regime_rest_edge",
    "regime_neutral_site",
)


@dataclass(frozen=True)
class RegimeSeasonMetrics:
    season: int
    games: int
    mean_residual: float
    mae: float
    rmse: float


@dataclass(frozen=True)
class RegimeSummary:
    target: str
    dimension: str
    regime: str
    games: int
    mean_residual: float
    mae: float
    rmse: float
    bias_direction: str
    bootstrap_low: float
    bootstrap_high: float
    minimum_season_games: int
    same_direction_all_seasons: bool
    sample_sufficient: bool
    interval_excludes_zero: bool
    persistent_bias: bool
    seasons: tuple[RegimeSeasonMetrics, ...]


@dataclass(frozen=True)
class ResidualRegimeReport:
    test_seasons: tuple[int, ...]
    minimum_games_per_season: int
    bootstrap_iterations: int
    margin: tuple[RegimeSummary, ...]
    total: tuple[RegimeSummary, ...]
    persistent_margin_regimes: tuple[str, ...]
    persistent_total_regimes: tuple[str, ...]
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def annotate_residual_regimes(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach fixed, predeclared regime labels using only pregame-known fields."""

    require_columns(
        frame,
        {
            "week",
            "baseline_home_margin",
            "baseline_total",
            "ctx_rest_diff_days",
            "ctx_neutral_site",
        },
        "residual_regime_frame",
    )
    return frame.with_columns(
        pl.when(pl.col("baseline_home_margin").abs() <= 3.0)
        .then(pl.lit("close_0_3"))
        .when(pl.col("baseline_home_margin").abs() <= 7.0)
        .then(pl.lit("medium_3_7"))
        .otherwise(pl.lit("large_7_plus"))
        .alias("regime_margin_band"),
        pl.when(pl.col("baseline_total") < 42.0)
        .then(pl.lit("low_under_42"))
        .when(pl.col("baseline_total") <= 48.0)
        .then(pl.lit("mid_42_48"))
        .otherwise(pl.lit("high_over_48"))
        .alias("regime_total_band"),
        pl.when(pl.col("week") <= 8)
        .then(pl.lit("early_5_8"))
        .when(pl.col("week") <= 13)
        .then(pl.lit("middle_9_13"))
        .otherwise(pl.lit("late_14_18"))
        .alias("regime_season_phase"),
        pl.when(pl.col("ctx_rest_diff_days") >= 2.0)
        .then(pl.lit("home_rest_edge"))
        .when(pl.col("ctx_rest_diff_days") <= -2.0)
        .then(pl.lit("away_rest_edge"))
        .otherwise(pl.lit("balanced_rest"))
        .alias("regime_rest_edge"),
        pl.when(pl.col("ctx_neutral_site") > 0.5)
        .then(pl.lit("neutral"))
        .otherwise(pl.lit("standard_home"))
        .alias("regime_neutral_site"),
    )


def _errors(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    residual = actual - predicted
    mean_residual = float(np.mean(residual))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(sqrt(float(np.mean(np.square(residual)))))
    return mean_residual, mae, rmse


def _direction(value: float, *, epsilon: float = 1e-12) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _direction_name(value: float) -> str:
    direction = _direction(value)
    if direction > 0:
        return "baseline_underpredicts"
    if direction < 0:
        return "baseline_overpredicts"
    return "flat"


def _block_bootstrap_mean_interval(
    frame: pl.DataFrame,
    residual_col: str,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap mean residual by season/week blocks to retain weekly dependence."""

    if iterations < 200:
        raise ValueError("bootstrap iterations must be >= 200")
    require_columns(frame, {"season", "week", residual_col}, "bootstrap_frame")
    blocks = [
        np.asarray(group.get_column(residual_col), dtype=float)
        for group in frame.partition_by(["season", "week"], maintain_order=True)
    ]
    if not blocks:
        raise DataContractError("cannot bootstrap an empty regime")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    block_count = len(blocks)
    for index in range(iterations):
        selected = rng.integers(0, block_count, size=block_count)
        values = np.concatenate([blocks[int(block)] for block in selected])
        estimates[index] = float(np.mean(values))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _summarize_one(
    frame: pl.DataFrame,
    *,
    target: str,
    dimension: str,
    regime: str,
    test_seasons: tuple[int, ...],
    minimum_games_per_season: int,
    bootstrap_iterations: int,
    seed: int,
) -> RegimeSummary:
    if target == "margin":
        actual_col = "actual_home_margin"
        baseline_col = "baseline_home_margin"
    elif target == "total":
        actual_col = "actual_total"
        baseline_col = "baseline_total"
    else:
        raise ValueError(f"unsupported target: {target}")

    subset = frame.filter(pl.col(dimension) == regime).with_columns(
        (pl.col(actual_col) - pl.col(baseline_col)).alias("_residual")
    )
    if subset.is_empty():
        raise DataContractError(f"empty regime {dimension}={regime}")

    season_metrics: list[RegimeSeasonMetrics] = []
    directions: list[int] = []
    minimum_games = subset.height
    for season in test_seasons:
        season_frame = subset.filter(pl.col("season") == season)
        actual = np.asarray(season_frame.get_column(actual_col), dtype=float)
        predicted = np.asarray(season_frame.get_column(baseline_col), dtype=float)
        if season_frame.is_empty():
            metrics = RegimeSeasonMetrics(
                season=season,
                games=0,
                mean_residual=float("nan"),
                mae=float("nan"),
                rmse=float("nan"),
            )
            directions.append(0)
            minimum_games = 0
        else:
            mean_residual, mae, rmse = _errors(actual, predicted)
            metrics = RegimeSeasonMetrics(
                season=season,
                games=season_frame.height,
                mean_residual=mean_residual,
                mae=mae,
                rmse=rmse,
            )
            directions.append(_direction(mean_residual))
            minimum_games = min(minimum_games, season_frame.height)
        season_metrics.append(metrics)

    actual = np.asarray(subset.get_column(actual_col), dtype=float)
    predicted = np.asarray(subset.get_column(baseline_col), dtype=float)
    mean_residual, mae, rmse = _errors(actual, predicted)
    low, high = _block_bootstrap_mean_interval(
        subset,
        "_residual",
        iterations=bootstrap_iterations,
        seed=seed,
    )
    nonzero_directions = {direction for direction in directions if direction != 0}
    same_direction = len(nonzero_directions) == 1 and all(
        direction != 0 for direction in directions
    )
    sample_sufficient = minimum_games >= minimum_games_per_season
    interval_excludes_zero = low > 0.0 or high < 0.0
    persistent = same_direction and sample_sufficient and interval_excludes_zero

    return RegimeSummary(
        target=target,
        dimension=dimension,
        regime=regime,
        games=subset.height,
        mean_residual=mean_residual,
        mae=mae,
        rmse=rmse,
        bias_direction=_direction_name(mean_residual),
        bootstrap_low=low,
        bootstrap_high=high,
        minimum_season_games=minimum_games,
        same_direction_all_seasons=same_direction,
        sample_sufficient=sample_sufficient,
        interval_excludes_zero=interval_excludes_zero,
        persistent_bias=persistent,
        seasons=tuple(season_metrics),
    )


def analyze_residual_regimes(
    frame: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    minimum_games_per_season: int = 24,
    bootstrap_iterations: int = 2000,
    seed: int = 20261001,
) -> ResidualRegimeReport:
    """Describe stable canonical residual regimes without fitting a correction."""

    require_columns(
        frame,
        {
            "season",
            "week",
            "actual_home_margin",
            "baseline_home_margin",
            "actual_total",
            "baseline_total",
            "ctx_rest_diff_days",
            "ctx_neutral_site",
        },
        "residual_regime_dataset",
    )
    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if minimum_games_per_season < 1:
        raise ValueError("minimum_games_per_season must be >= 1")

    evaluation = annotate_residual_regimes(
        frame.filter(pl.col("season").is_in(list(test_seasons)))
    )
    if evaluation.is_empty():
        raise DataContractError("residual-regime evaluation has no rows")

    summaries: dict[str, list[RegimeSummary]] = {"margin": [], "total": []}
    for target_index, target in enumerate(("margin", "total")):
        for dimension_index, dimension in enumerate(REGIME_COLUMNS):
            regimes = sorted(str(value) for value in evaluation.get_column(dimension).unique())
            for regime_index, regime in enumerate(regimes):
                regime_seed = seed + target_index * 1000 + dimension_index * 100 + regime_index
                summaries[target].append(
                    _summarize_one(
                        evaluation,
                        target=target,
                        dimension=dimension,
                        regime=regime,
                        test_seasons=test_seasons,
                        minimum_games_per_season=minimum_games_per_season,
                        bootstrap_iterations=bootstrap_iterations,
                        seed=regime_seed,
                    )
                )

    margin = tuple(summaries["margin"])
    total = tuple(summaries["total"])
    persistent_margin = tuple(
        f"{item.dimension}={item.regime}" for item in margin if item.persistent_bias
    )
    persistent_total = tuple(
        f"{item.dimension}={item.regime}" for item in total if item.persistent_bias
    )
    return ResidualRegimeReport(
        test_seasons=test_seasons,
        minimum_games_per_season=minimum_games_per_season,
        bootstrap_iterations=bootstrap_iterations,
        margin=margin,
        total=total,
        persistent_margin_regimes=persistent_margin,
        persistent_total_regimes=persistent_total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
    )
