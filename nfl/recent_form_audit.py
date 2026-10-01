"""Real-source rolling-origin audit for NFL recent-form PBP candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .recent_form_dataset import build_recent_form_walkforward_dataset
from .recent_form_residuals import RecentFormRollingEvaluation, evaluate_recent_form_rolling


@dataclass(frozen=True)
class RecentFormAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    recent_alpha_grid: tuple[float, ...]
    rows_per_alpha: int
    margin_recent_alpha: float | None
    margin_feature_set: str
    margin_ridge_alpha: float | None
    margin_blend_weight: float
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_positive_folds: int
    margin_shadow_candidate: bool
    total_recent_alpha: float | None
    total_feature_set: str
    total_ridge_alpha: float | None
    total_blend_weight: float
    total_baseline_mae: float
    total_adjusted_mae: float
    total_baseline_rmse: float
    total_adjusted_rmse: float
    total_positive_folds: int
    total_shadow_candidate: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_recent_form_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    recent_alpha_grid: tuple[float, ...] = (0.20, 0.35, 0.50),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> RecentFormAudit:
    """Build each alpha chronologically and select only shadow-eligible research state."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )
    pbp = source.load_pbp(list(seasons), refresh=refresh)

    datasets: dict[float, pl.DataFrame] = {}
    for recent_alpha in recent_alpha_grid:
        frames = [
            build_recent_form_walkforward_dataset(
                schedules,
                pbp,
                season,
                recent_alpha=recent_alpha,
                start_week=start_week,
                end_week=end_week,
                score_ridge=score_ridge,
            )
            for season in seasons
        ]
        datasets[float(recent_alpha)] = pl.concat(frames, how="vertical_relaxed").sort(
            ["season", "week", "game_id"]
        )

    evaluation: RecentFormRollingEvaluation = evaluate_recent_form_rolling(
        datasets,
        test_seasons=test_seasons,
    )
    rows_per_alpha = next(iter(datasets.values())).height
    margin = evaluation.margin
    total = evaluation.total
    return RecentFormAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        recent_alpha_grid=recent_alpha_grid,
        rows_per_alpha=rows_per_alpha,
        margin_recent_alpha=margin.recent_alpha,
        margin_feature_set=margin.feature_set,
        margin_ridge_alpha=margin.ridge_alpha,
        margin_blend_weight=margin.blend_weight,
        margin_baseline_mae=margin.baseline_mae,
        margin_adjusted_mae=margin.adjusted_mae,
        margin_baseline_rmse=margin.baseline_rmse,
        margin_adjusted_rmse=margin.adjusted_rmse,
        margin_positive_folds=margin.positive_folds,
        margin_shadow_candidate=margin.shadow_candidate,
        total_recent_alpha=total.recent_alpha,
        total_feature_set=total.feature_set,
        total_ridge_alpha=total.ridge_alpha,
        total_blend_weight=total.blend_weight,
        total_baseline_mae=total.baseline_mae,
        total_adjusted_mae=total.adjusted_mae,
        total_baseline_rmse=total.baseline_rmse,
        total_adjusted_rmse=total.adjusted_rmse,
        total_positive_folds=total.positive_folds,
        total_shadow_candidate=total.shadow_candidate,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
