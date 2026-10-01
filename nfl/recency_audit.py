"""Live validation/holdout audit for the recency-weighted fair-score candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .recency import evaluate_recency_holdout


@dataclass(frozen=True)
class RecencyAudit:
    validation_season: int
    holdout_season: int
    selected_half_life_weeks: float | None
    validation_games: int
    validation_baseline_margin_mae: float
    validation_baseline_margin_rmse: float
    validation_baseline_total_mae: float
    validation_baseline_total_rmse: float
    validation_recency_margin_mae: float | None
    validation_recency_margin_rmse: float | None
    validation_recency_total_mae: float | None
    validation_recency_total_rmse: float | None
    holdout_games: int
    holdout_baseline_margin_mae: float
    holdout_baseline_margin_rmse: float
    holdout_baseline_total_mae: float
    holdout_baseline_total_rmse: float
    holdout_recency_margin_mae: float | None
    holdout_recency_margin_rmse: float | None
    holdout_recency_total_mae: float | None
    holdout_recency_total_rmse: float | None
    margin_mae_improvement: float | None
    margin_rmse_improvement: float | None
    total_mae_improvement: float | None
    total_rmse_improvement: float | None
    validation_candidate_pass: bool
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
        list(range(validation_season - 1, holdout_season + 1)),
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

    validation_recency = evaluation.validation_recency
    holdout_recency = evaluation.holdout_recency
    return RecencyAudit(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_half_life_weeks=evaluation.selected_half_life_weeks,
        validation_games=evaluation.validation_baseline.games,
        validation_baseline_margin_mae=evaluation.validation_baseline.margin_mae,
        validation_baseline_margin_rmse=evaluation.validation_baseline.margin_rmse,
        validation_baseline_total_mae=evaluation.validation_baseline.total_mae,
        validation_baseline_total_rmse=evaluation.validation_baseline.total_rmse,
        validation_recency_margin_mae=(
            validation_recency.margin_mae if validation_recency is not None else None
        ),
        validation_recency_margin_rmse=(
            validation_recency.margin_rmse if validation_recency is not None else None
        ),
        validation_recency_total_mae=(
            validation_recency.total_mae if validation_recency is not None else None
        ),
        validation_recency_total_rmse=(
            validation_recency.total_rmse if validation_recency is not None else None
        ),
        holdout_games=evaluation.holdout_baseline.games,
        holdout_baseline_margin_mae=evaluation.holdout_baseline.margin_mae,
        holdout_baseline_margin_rmse=evaluation.holdout_baseline.margin_rmse,
        holdout_baseline_total_mae=evaluation.holdout_baseline.total_mae,
        holdout_baseline_total_rmse=evaluation.holdout_baseline.total_rmse,
        holdout_recency_margin_mae=(
            holdout_recency.margin_mae if holdout_recency is not None else None
        ),
        holdout_recency_margin_rmse=(
            holdout_recency.margin_rmse if holdout_recency is not None else None
        ),
        holdout_recency_total_mae=(
            holdout_recency.total_mae if holdout_recency is not None else None
        ),
        holdout_recency_total_rmse=(
            holdout_recency.total_rmse if holdout_recency is not None else None
        ),
        margin_mae_improvement=evaluation.margin_mae_improvement,
        margin_rmse_improvement=evaluation.margin_rmse_improvement,
        total_mae_improvement=evaluation.total_mae_improvement,
        total_rmse_improvement=evaluation.total_rmse_improvement,
        validation_candidate_pass=evaluation.validation_candidate_pass,
        candidate_pass=evaluation.candidate_pass,
    )
