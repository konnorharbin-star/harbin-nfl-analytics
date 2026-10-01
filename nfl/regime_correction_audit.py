"""Real-source audit for Stage 22 targeted residual-regime corrections."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .regime_corrections import RegimeCorrectionEvaluation, evaluate_regime_corrections
from .schedule_context import (
    SCHEDULE_CONTEXT_REQUIRED,
    build_schedule_context_walkforward,
)


@dataclass(frozen=True)
class RegimeCorrectionAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    evaluation_games: int
    source_columns_verified: tuple[str, ...]
    margin_shadow_candidate: bool
    margin_positive_folds: int
    margin_mae_improvement: float
    margin_rmse_improvement: float
    total_shadow_candidate: bool
    total_positive_folds: int
    total_mae_improvement: float
    total_rmse_improvement: float
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_regime_correction_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> RegimeCorrectionAudit:
    """Replay fixed Stage 21 corrections on expanding earlier-season histories."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )
    missing = sorted(SCHEDULE_CONTEXT_REQUIRED - set(schedules.columns))
    if missing:
        raise ValueError(f"nflverse schedules missing correction source columns: {missing}")

    frames = [
        build_schedule_context_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    evaluation: RegimeCorrectionEvaluation = evaluate_regime_corrections(
        dataset,
        test_seasons=test_seasons,
    )
    margin = evaluation.margin_away_rest
    total = evaluation.total_low_projection
    evaluation_games = dataset.filter(
        pl.col("season").is_in(list(test_seasons))
    ).height
    return RegimeCorrectionAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        evaluation_games=evaluation_games,
        source_columns_verified=tuple(sorted(SCHEDULE_CONTEXT_REQUIRED)),
        margin_shadow_candidate=margin.shadow_candidate,
        margin_positive_folds=margin.positive_folds,
        margin_mae_improvement=margin.mae_improvement,
        margin_rmse_improvement=margin.rmse_improvement,
        total_shadow_candidate=total.shadow_candidate,
        total_positive_folds=total.positive_folds,
        total_mae_improvement=total.mae_improvement,
        total_rmse_improvement=total.rmse_improvement,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        evaluation=evaluation.to_dict(),
    )
