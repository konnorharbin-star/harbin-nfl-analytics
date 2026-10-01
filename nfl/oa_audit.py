"""Live multi-season holdout audit for opponent-adjusted PBP candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .oa_dataset import build_oa_walkforward_dataset
from .oa_residuals import evaluate_oa_nested_holdout


@dataclass(frozen=True)
class OAAudit:
    seasons: tuple[int, ...]
    rows: int
    validation_season: int
    holdout_season: int
    pbp_ridge: float
    margin_feature_set: str
    margin_alpha: float
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_candidate_pass: bool
    total_feature_set: str
    total_alpha: float
    total_baseline_mae: float
    total_adjusted_mae: float
    total_baseline_rmse: float
    total_adjusted_rmse: float
    total_candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_oa_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    validation_season: int = 2024,
    holdout_season: int = 2025,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    pbp_ridge: float = 12.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> OAAudit:
    """Build point-in-time OA PBP datasets and score the untouched holdout."""

    if validation_season not in seasons or holdout_season not in seasons:
        raise ValueError("validation and holdout seasons must be included in seasons")

    source = client or NFLDataClient()
    schedule_seasons = sorted({min(seasons) - 1, *seasons})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    pbp = source.load_pbp(list(seasons), refresh=refresh)

    frames = [
        build_oa_walkforward_dataset(
            schedules,
            pbp.filter(pl.col("season") == season),
            season,
            start_week=start_week,
            end_week=end_week,
            score_ridge=score_ridge,
            pbp_ridge=pbp_ridge,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
    evaluation = evaluate_oa_nested_holdout(
        dataset,
        validation_season=validation_season,
        holdout_season=holdout_season,
    )

    return OAAudit(
        seasons=seasons,
        rows=dataset.height,
        validation_season=validation_season,
        holdout_season=holdout_season,
        pbp_ridge=pbp_ridge,
        margin_feature_set=evaluation.margin.feature_set,
        margin_alpha=evaluation.margin.alpha,
        margin_baseline_mae=evaluation.margin.baseline_mae,
        margin_adjusted_mae=evaluation.margin.adjusted_mae,
        margin_baseline_rmse=evaluation.margin.baseline_rmse,
        margin_adjusted_rmse=evaluation.margin.adjusted_rmse,
        margin_candidate_pass=evaluation.margin.candidate_pass,
        total_feature_set=evaluation.total.feature_set,
        total_alpha=evaluation.total.alpha,
        total_baseline_mae=evaluation.total.baseline_mae,
        total_adjusted_mae=evaluation.total.adjusted_mae,
        total_baseline_rmse=evaluation.total.baseline_rmse,
        total_adjusted_rmse=evaluation.total.adjusted_rmse,
        total_candidate_pass=evaluation.total.candidate_pass,
    )
