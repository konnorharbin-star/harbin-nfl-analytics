"""Forward out-of-time shadow audit for the frozen QB adjustment structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .qb_dataset import build_qb_walkforward_dataset
from .qb_state import QB_PRIOR_DROPBACKS
from .qb_validated import ValidatedQBAdjustment, score_qb_adjustment
from .shadow_gate import TargetShadowEvidence, paired_bootstrap_evidence


@dataclass(frozen=True)
class QBShadowAudit:
    training_seasons: tuple[int, ...]
    shadow_season: int
    training_games: int
    shadow_games: int
    shadow_start_week: int
    shadow_end_week: int
    qb_prior_dropbacks: float
    baseline_margin_mae: float
    adjusted_margin_mae: float
    baseline_margin_rmse: float
    adjusted_margin_rmse: float
    margin_mae_improvement: float
    margin_rmse_improvement: float
    margin_pass: bool
    margin_evidence: TargetShadowEvidence
    baseline_total_mae: float
    adjusted_total_mae: float
    baseline_total_rmse: float
    adjusted_total_rmse: float
    total_mae_improvement: float
    total_rmse_improvement: float
    total_pass: bool
    total_evidence: TargetShadowEvidence

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_qb_shadow_audit(
    *,
    training_seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    shadow_season: int = 2026,
    training_start_week: int = 5,
    shadow_start_week: int = 3,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    minimum_shadow_games: int = 128,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> QBShadowAudit:
    """Fit only on 2022-2025 and score completed 2026 weeks without retuning."""

    if max(training_seasons) >= shadow_season:
        raise ValueError("all training seasons must precede the shadow season")

    source = client or NFLDataClient()
    schedule_seasons = sorted({min(training_seasons) - 1, *training_seasons, shadow_season})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    stats_seasons = sorted({*training_seasons, shadow_season})
    player_stats = source.load_player_stats(stats_seasons, refresh=refresh)

    training_frames = [
        build_qb_walkforward_dataset(
            schedules,
            player_stats.filter(pl.col("season") == season),
            season,
            start_week=training_start_week,
            end_week=18,
            score_ridge=score_ridge,
            qb_prior_dropbacks=qb_prior_dropbacks,
        )
        for season in training_seasons
    ]
    training = pl.concat(training_frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )

    shadow = build_qb_walkforward_dataset(
        schedules,
        player_stats.filter(pl.col("season") == shadow_season),
        shadow_season,
        start_week=shadow_start_week,
        end_week=None,
        score_ridge=score_ridge,
        qb_prior_dropbacks=qb_prior_dropbacks,
    )
    model = ValidatedQBAdjustment().fit(training)
    scored = model.apply(shadow)
    metrics = score_qb_adjustment(scored)

    margin_evidence = paired_bootstrap_evidence(
        scored.get_column("actual_home_margin"),
        scored.get_column("baseline_home_margin"),
        scored.get_column("qb_adjusted_home_margin"),
        iterations=bootstrap_iterations,
        confidence=confidence,
        minimum_games=minimum_shadow_games,
        seed=20260930,
    )
    total_evidence = paired_bootstrap_evidence(
        scored.get_column("actual_total"),
        scored.get_column("baseline_total"),
        scored.get_column("qb_adjusted_total"),
        iterations=bootstrap_iterations,
        confidence=confidence,
        minimum_games=minimum_shadow_games,
        seed=20260931,
    )

    return QBShadowAudit(
        training_seasons=training_seasons,
        shadow_season=shadow_season,
        training_games=training.height,
        shadow_games=metrics.games,
        shadow_start_week=int(shadow.get_column("week").min()),
        shadow_end_week=int(shadow.get_column("week").max()),
        qb_prior_dropbacks=qb_prior_dropbacks,
        baseline_margin_mae=metrics.baseline_margin_mae,
        adjusted_margin_mae=metrics.adjusted_margin_mae,
        baseline_margin_rmse=metrics.baseline_margin_rmse,
        adjusted_margin_rmse=metrics.adjusted_margin_rmse,
        margin_mae_improvement=metrics.margin_mae_improvement,
        margin_rmse_improvement=metrics.margin_rmse_improvement,
        margin_pass=metrics.margin_pass,
        margin_evidence=margin_evidence,
        baseline_total_mae=metrics.baseline_total_mae,
        adjusted_total_mae=metrics.adjusted_total_mae,
        baseline_total_rmse=metrics.baseline_total_rmse,
        adjusted_total_rmse=metrics.adjusted_total_rmse,
        total_mae_improvement=metrics.total_mae_improvement,
        total_rmse_improvement=metrics.total_rmse_improvement,
        total_pass=metrics.total_pass,
        total_evidence=total_evidence,
    )
