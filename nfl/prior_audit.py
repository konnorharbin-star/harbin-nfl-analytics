"""Live validation/holdout audit for the prior-season scoring candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .priors import evaluate_prior_holdout


@dataclass(frozen=True)
class PriorAudit:
    validation_season: int
    holdout_season: int
    selected_prior_weight: float
    validation_games: int
    validation_baseline_margin_rmse: float
    validation_prior_margin_rmse: float
    validation_baseline_total_rmse: float
    validation_prior_total_rmse: float
    holdout_games: int
    holdout_baseline_margin_mae: float
    holdout_prior_margin_mae: float
    holdout_baseline_margin_rmse: float
    holdout_prior_margin_rmse: float
    holdout_baseline_total_mae: float
    holdout_prior_total_mae: float
    holdout_baseline_total_rmse: float
    holdout_prior_total_rmse: float
    margin_mae_improvement: float
    margin_rmse_improvement: float
    total_mae_improvement: float
    total_rmse_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_prior_audit(
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    start_week: int = 2,
    ridge: float = 8.0,
    prior_weight_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35),
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> PriorAudit:
    """Select prior weight on validation, then score the later holdout season."""

    source = client or NFLDataClient()
    seasons = sorted({validation_season - 1, validation_season, holdout_season})
    schedules = source.load_schedules(seasons, refresh=refresh)
    evaluation = evaluate_prior_holdout(
        schedules,
        validation_season=validation_season,
        holdout_season=holdout_season,
        prior_weight_grid=prior_weight_grid,
        start_week=start_week,
        ridge=ridge,
    )

    return PriorAudit(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_prior_weight=evaluation.selected_prior_weight,
        validation_games=evaluation.validation_baseline.games,
        validation_baseline_margin_rmse=evaluation.validation_baseline.margin_rmse,
        validation_prior_margin_rmse=evaluation.validation_prior.margin_rmse,
        validation_baseline_total_rmse=evaluation.validation_baseline.total_rmse,
        validation_prior_total_rmse=evaluation.validation_prior.total_rmse,
        holdout_games=evaluation.holdout_baseline.games,
        holdout_baseline_margin_mae=evaluation.holdout_baseline.margin_mae,
        holdout_prior_margin_mae=evaluation.holdout_prior.margin_mae,
        holdout_baseline_margin_rmse=evaluation.holdout_baseline.margin_rmse,
        holdout_prior_margin_rmse=evaluation.holdout_prior.margin_rmse,
        holdout_baseline_total_mae=evaluation.holdout_baseline.total_mae,
        holdout_prior_total_mae=evaluation.holdout_prior.total_mae,
        holdout_baseline_total_rmse=evaluation.holdout_baseline.total_rmse,
        holdout_prior_total_rmse=evaluation.holdout_prior.total_rmse,
        margin_mae_improvement=evaluation.margin_mae_improvement,
        margin_rmse_improvement=evaluation.margin_rmse_improvement,
        total_mae_improvement=evaluation.total_mae_improvement,
        total_rmse_improvement=evaluation.total_rmse_improvement,
        candidate_pass=evaluation.candidate_pass,
    )
