"""Live chronological audit for logistic NFL home-win calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .recency import build_score_walkforward
from .win_probability import evaluate_win_probability_holdout


@dataclass(frozen=True)
class WinProbabilityAudit:
    seasons: tuple[int, ...]
    rows: int
    validation_season: int
    holdout_season: int
    alpha: float
    training_games: int
    validation_games: int
    holdout_games: int
    gaussian_brier: float
    logistic_brier: float
    gaussian_log_loss: float
    logistic_log_loss: float
    brier_improvement: float
    log_loss_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_win_probability_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    validation_season: int = 2024,
    holdout_season: int = 2025,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> WinProbabilityAudit:
    """Reconstruct canonical forecasts and score logistic win calibration."""

    if validation_season not in seasons or holdout_season not in seasons:
        raise ValueError("validation and holdout seasons must be included in seasons")
    source = client or NFLDataClient()
    schedule_seasons = sorted({min(seasons) - 1, *seasons})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    frames = [
        build_score_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
            ridge=score_ridge,
        ).with_columns(pl.lit(season).alias("season"))
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    evaluation = evaluate_win_probability_holdout(
        dataset,
        validation_season=validation_season,
        holdout_season=holdout_season,
    )
    return WinProbabilityAudit(
        seasons=seasons,
        rows=dataset.height,
        validation_season=validation_season,
        holdout_season=holdout_season,
        alpha=evaluation.alpha,
        training_games=evaluation.training_games,
        validation_games=evaluation.validation_games,
        holdout_games=evaluation.holdout_games,
        gaussian_brier=evaluation.gaussian.brier,
        logistic_brier=evaluation.logistic.brier,
        gaussian_log_loss=evaluation.gaussian.log_loss,
        logistic_log_loss=evaluation.logistic.log_loss,
        brier_improvement=evaluation.brier_improvement,
        log_loss_improvement=evaluation.log_loss_improvement,
        candidate_pass=evaluation.candidate_pass,
    )
