"""Real-source Stage 22 audit for the nonlinear NFL residual ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .ensemble_dataset import build_ensemble_walkforward_dataset
from .ensemble_residuals import NonlinearEnsembleEvaluation, evaluate_nonlinear_ensemble


@dataclass(frozen=True)
class NonlinearEnsembleAudit:
    seasons: tuple[int, ...]
    rows: int
    feature_count: int
    margin_positive_folds: int
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_architecture_candidate: bool
    total_positive_folds: int
    total_baseline_mae: float
    total_adjusted_mae: float
    total_baseline_rmse: float
    total_adjusted_rmse: float
    total_architecture_candidate: bool
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_nonlinear_ensemble_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> NonlinearEnsembleAudit:
    """Build the joined point-in-time matrix and run nested rolling evaluation."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        sorted({min(seasons) - 1, *seasons}),
        refresh=refresh,
    )
    pbp = source.load_pbp(list(seasons), refresh=refresh)
    player_stats = source.load_player_stats(list(seasons), refresh=refresh)

    frames = [
        build_ensemble_walkforward_dataset(
            schedules,
            pbp.filter(pl.col("season") == season),
            player_stats,
            season,
            start_week=start_week,
            end_week=end_week,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    evaluation: NonlinearEnsembleEvaluation = evaluate_nonlinear_ensemble(dataset)
    margin = evaluation.margin
    total = evaluation.total
    return NonlinearEnsembleAudit(
        seasons=seasons,
        rows=dataset.height,
        feature_count=evaluation.feature_count,
        margin_positive_folds=margin.positive_folds,
        margin_baseline_mae=margin.aggregate_baseline_mae,
        margin_adjusted_mae=margin.aggregate_adjusted_mae,
        margin_baseline_rmse=margin.aggregate_baseline_rmse,
        margin_adjusted_rmse=margin.aggregate_adjusted_rmse,
        margin_architecture_candidate=margin.architecture_candidate,
        total_positive_folds=total.positive_folds,
        total_baseline_mae=total.aggregate_baseline_mae,
        total_adjusted_mae=total.aggregate_adjusted_mae,
        total_baseline_rmse=total.aggregate_baseline_rmse,
        total_adjusted_rmse=total.aggregate_adjusted_rmse,
        total_architecture_candidate=total.architecture_candidate,
        canonical_score_adjustment_enabled=evaluation.canonical_score_adjustment_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
