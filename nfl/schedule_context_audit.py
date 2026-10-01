"""Real-source rolling audit for historical schedule/venue context."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .schedule_context import (
    SCHEDULE_CONTEXT_REQUIRED,
    build_schedule_context_walkforward,
)
from .schedule_context_eval import ScheduleContextEvaluation, evaluate_schedule_context


@dataclass(frozen=True)
class ScheduleContextAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    source_columns_verified: tuple[str, ...]
    margin_selected_alpha: float | None
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_positive_folds: int
    margin_shadow_candidate: bool
    margin_best_tested_alpha: float
    margin_best_tested_positive_folds: int
    total_selected_alpha: float | None
    total_baseline_mae: float
    total_adjusted_mae: float
    total_baseline_rmse: float
    total_adjusted_rmse: float
    total_positive_folds: int
    total_shadow_candidate: bool
    total_best_tested_alpha: float
    total_best_tested_positive_folds: int
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_schedule_context_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    alpha_grid: tuple[float, ...] = (1.0, 10.0, 100.0),
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> ScheduleContextAudit:
    """Build chronological schedule-context rows and run the fixed rolling gate."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )
    missing = sorted(SCHEDULE_CONTEXT_REQUIRED - set(schedules.columns))
    if missing:
        raise ValueError(f"nflverse schedules missing context source columns: {missing}")

    frames = [
        build_schedule_context_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
            score_ridge=score_ridge,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    evaluation: ScheduleContextEvaluation = evaluate_schedule_context(
        dataset,
        test_seasons=test_seasons,
        alpha_grid=alpha_grid,
    )
    margin = evaluation.margin
    total = evaluation.total
    return ScheduleContextAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        source_columns_verified=tuple(sorted(SCHEDULE_CONTEXT_REQUIRED)),
        margin_selected_alpha=margin.selected_alpha,
        margin_baseline_mae=margin.baseline_mae,
        margin_adjusted_mae=margin.adjusted_mae,
        margin_baseline_rmse=margin.baseline_rmse,
        margin_adjusted_rmse=margin.adjusted_rmse,
        margin_positive_folds=margin.positive_folds,
        margin_shadow_candidate=margin.shadow_candidate,
        margin_best_tested_alpha=margin.best_tested_alpha,
        margin_best_tested_positive_folds=margin.best_tested_positive_folds,
        total_selected_alpha=total.selected_alpha,
        total_baseline_mae=total.baseline_mae,
        total_adjusted_mae=total.adjusted_mae,
        total_baseline_rmse=total.baseline_rmse,
        total_adjusted_rmse=total.adjusted_rmse,
        total_positive_folds=total.positive_folds,
        total_shadow_candidate=total.shadow_candidate,
        total_best_tested_alpha=total.best_tested_alpha,
        total_best_tested_positive_folds=total.best_tested_positive_folds,
        canonical_score_adjustment_enabled=evaluation.canonical_score_adjustment_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
