"""Live validation/holdout audit for the recency-weighted fair-score candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .recency import evaluate_recency_holdout


@dataclass(frozen=True)
class RecencyAudit:
    validation_season: int
    holdout_season: int
    selected_half_life_weeks: float
    validation_games: int
    validation_baseline_margin_rmse: float
    validation_recency_margin_rmse: float
    validation_baseline_total_rmse: float
    validation_recency_total_rmse: float
    holdout_games: int
    holdout_baseline_margin_mae: float
    holdout_recency_margin_mae: float
    holdout_baseline_margin_rmse: float
    holdout_recency_margin_rmse: float
    holdout_baseline_total_mae: float
    holdout_recency_total_mae: float
    holdout_baseline_total_rmse: float
    holdout_recency_total_rmse: float
    margin_mae_improvement: float
    margin_rmse_improvement: float
    total_mae_improvement: float
    total_rmse_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_recency_audit(
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    start_week: int = 5,
    ridge: float = 8.0,
    half_life_grid: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0),
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> RecencyAudit:
    """Select a half-life on validation data, then score one later holdout season."""

    source = client or NFLDataClient()
    schedules = source.load_schedules(
        [validation_season, holdout_season],
        refresh=refresh,
    )
    evaluation = evaluate_recency_holdout(
        schedules,
        validation_season=validation_season,
        holdout_season=holdout_season,
        half_life_grid=half_life_grid,
        start_week=start_week,
        ridge=ridge,
    )

    return RecencyAudit(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_half_life_weeks=evaluation.selected_half_life_weeks,
        validation_games=evaluation.validation_baseline.games,
        validation_baseline_margin_rmse=evaluation.validation_baseline.margin_rmse,
        validation_recency_margin_rmse=evaluation.validation_recency.margin_rmse,
        validation_baseline_total_rmse=evaluation.validation_baseline.total_rmse,
        validation_recency_total_rmse=evaluation.validation_recency.total_rmse,
        holdout_games=evaluation.holdout_baseline.games,
        holdout_baseline_margin_mae=evaluation.holdout_baseline.margin_mae,
        holdout_recency_margin_mae=evaluation.holdout_recency.margin_mae,
        holdout_baseline_margin_rmse=evaluation.holdout_baseline.margin_rmse,
        holdout_recency_margin_rmse=evaluation.holdout_recency.margin_rmse,
        holdout_baseline_total_mae=evaluation.holdout_baseline.total_mae,
        holdout_recency_total_mae=evaluation.holdout_recency.total_mae,
        holdout_baseline_total_rmse=evaluation.holdout_baseline.total_rmse,
        holdout_recency_total_rmse=evaluation.holdout_recency.total_rmse,
        margin_mae_improvement=evaluation.margin_mae_improvement,
        margin_rmse_improvement=evaluation.margin_rmse_improvement,
        total_mae_improvement=evaluation.total_mae_improvement,
        total_rmse_improvement=evaluation.total_rmse_improvement,
        candidate_pass=evaluation.candidate_pass,
    )
