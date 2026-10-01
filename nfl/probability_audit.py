"""Live chronological holdout audit for score-distribution calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .probability import evaluate_probability_holdout
from .recency import build_score_walkforward


@dataclass(frozen=True)
class ProbabilityAudit:
    seasons: tuple[int, ...]
    rows: int
    validation_season: int
    holdout_season: int
    margin_scale: float
    total_scale: float
    training_games: int
    validation_games: int
    holdout_games: int
    uncalibrated_margin_nll: float
    calibrated_margin_nll: float
    uncalibrated_total_nll: float
    calibrated_total_nll: float
    uncalibrated_home_win_brier: float
    calibrated_home_win_brier: float
    margin_50_coverage: float
    margin_80_coverage: float
    total_50_coverage: float
    total_80_coverage: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_probability_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    validation_season: int = 2024,
    holdout_season: int = 2025,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> ProbabilityAudit:
    """Reconstruct score forecasts and audit Gaussian residual calibration."""

    if validation_season not in seasons or holdout_season not in seasons:
        raise ValueError("validation and holdout seasons must be included in seasons")
    if min(seasons) >= validation_season:
        raise ValueError("at least one pre-validation season is required")

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
    evaluation = evaluate_probability_holdout(
        dataset,
        validation_season=validation_season,
        holdout_season=holdout_season,
    )
    calibrated = evaluation.holdout_calibrated
    baseline = evaluation.holdout_uncalibrated
    return ProbabilityAudit(
        seasons=seasons,
        rows=dataset.height,
        validation_season=validation_season,
        holdout_season=holdout_season,
        margin_scale=evaluation.margin_scale,
        total_scale=evaluation.total_scale,
        training_games=evaluation.training_games,
        validation_games=evaluation.validation_games,
        holdout_games=evaluation.holdout_games,
        uncalibrated_margin_nll=baseline.margin_nll,
        calibrated_margin_nll=calibrated.margin_nll,
        uncalibrated_total_nll=baseline.total_nll,
        calibrated_total_nll=calibrated.total_nll,
        uncalibrated_home_win_brier=baseline.home_win_brier,
        calibrated_home_win_brier=calibrated.home_win_brier,
        margin_50_coverage=calibrated.margin_50_coverage,
        margin_80_coverage=calibrated.margin_80_coverage,
        total_50_coverage=calibrated.total_50_coverage,
        total_80_coverage=calibrated.total_80_coverage,
        candidate_pass=evaluation.candidate_pass,
    )
