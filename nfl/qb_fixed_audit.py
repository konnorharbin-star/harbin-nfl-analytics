"""Real-source audit for one fixed QB residual specification across development folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .qb_dataset import build_qb_walkforward_dataset
from .qb_rolling import QBFixedRollingEvaluation, evaluate_fixed_qb_rolling
from .qb_state import QB_PRIOR_DROPBACKS


@dataclass(frozen=True)
class QBFixedAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    qb_prior_dropbacks: float
    margin_feature_set: str
    margin_ridge_alpha: float | None
    margin_games: int
    margin_baseline_mae: float
    margin_adjusted_mae: float
    margin_baseline_rmse: float
    margin_adjusted_rmse: float
    margin_positive_folds: int
    margin_shadow_candidate: bool
    total_feature_set: str
    total_ridge_alpha: float | None
    total_games: int
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


def run_fixed_qb_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> QBFixedAudit:
    """Build point-in-time QB datasets once, then select one fixed development spec."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    if any(season not in seasons for season in test_seasons):
        raise ValueError("every test season must be included in seasons")

    source = client or NFLDataClient()
    schedule_seasons = sorted({min(seasons) - 1, *seasons})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    player_stats = source.load_player_stats(list(seasons), refresh=refresh)

    frames = [
        build_qb_walkforward_dataset(
            schedules,
            player_stats.filter(pl.col("season") == season),
            season,
            start_week=start_week,
            end_week=end_week,
            score_ridge=score_ridge,
            qb_prior_dropbacks=qb_prior_dropbacks,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
    evaluation: QBFixedRollingEvaluation = evaluate_fixed_qb_rolling(
        dataset,
        test_seasons=test_seasons,
    )
    margin = evaluation.margin
    total = evaluation.total
    return QBFixedAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        qb_prior_dropbacks=qb_prior_dropbacks,
        margin_feature_set=margin.feature_set,
        margin_ridge_alpha=margin.ridge_alpha,
        margin_games=margin.games,
        margin_baseline_mae=margin.baseline_mae,
        margin_adjusted_mae=margin.adjusted_mae,
        margin_baseline_rmse=margin.baseline_rmse,
        margin_adjusted_rmse=margin.adjusted_rmse,
        margin_positive_folds=margin.positive_folds,
        margin_shadow_candidate=margin.shadow_candidate,
        total_feature_set=total.feature_set,
        total_ridge_alpha=total.ridge_alpha,
        total_games=total.games,
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
