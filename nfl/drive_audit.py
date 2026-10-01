"""Real-source rolling audit for NFL drive-efficiency candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .drive_dataset import build_drive_walkforward_dataset
from .drive_features import DRIVE_PBP_REQUIRED
from .drive_fixed import DriveRollingEvaluation, evaluate_drive_rolling


@dataclass(frozen=True)
class DriveAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    source_columns_verified: tuple[str, ...]
    margin_feature_set: str
    margin_ridge_alpha: float | None
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_positive_folds: int
    margin_shadow_candidate: bool
    total_feature_set: str
    total_ridge_alpha: float | None
    total_baseline_mae: float
    total_adjusted_mae: float
    total_baseline_rmse: float
    total_adjusted_rmse: float
    total_positive_folds: int
    total_shadow_candidate: bool
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_drive_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> DriveAudit:
    """Build point-in-time drive datasets and run the fixed rolling gate."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )
    pbp = source.load_pbp(list(seasons), refresh=refresh)
    missing = sorted(DRIVE_PBP_REQUIRED - set(pbp.columns))
    if missing:
        raise ValueError(f"nflverse PBP is missing drive source columns: {missing}")

    frames = [
        build_drive_walkforward_dataset(
            schedules,
            pbp,
            season,
            start_week=start_week,
            end_week=end_week,
            score_ridge=score_ridge,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
    evaluation: DriveRollingEvaluation = evaluate_drive_rolling(
        dataset,
        test_seasons=test_seasons,
    )
    margin = evaluation.margin
    total = evaluation.total
    return DriveAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        source_columns_verified=tuple(sorted(DRIVE_PBP_REQUIRED)),
        margin_feature_set=margin.feature_set,
        margin_ridge_alpha=margin.ridge_alpha,
        margin_baseline_mae=margin.baseline_mae,
        margin_adjusted_mae=margin.adjusted_mae,
        margin_baseline_rmse=margin.baseline_rmse,
        margin_adjusted_rmse=margin.adjusted_rmse,
        margin_positive_folds=margin.positive_folds,
        margin_shadow_candidate=margin.shadow_candidate,
        total_feature_set=total.feature_set,
        total_ridge_alpha=total.ridge_alpha,
        total_baseline_mae=total.baseline_mae,
        total_adjusted_mae=total.adjusted_mae,
        total_baseline_rmse=total.baseline_rmse,
        total_adjusted_rmse=total.adjusted_rmse,
        total_positive_folds=total.positive_folds,
        total_shadow_candidate=total.shadow_candidate,
        canonical_score_adjustment_enabled=evaluation.canonical_score_adjustment_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
