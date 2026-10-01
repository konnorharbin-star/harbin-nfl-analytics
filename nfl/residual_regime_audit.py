"""Real-source audit for canonical NFL residual regimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .residual_regimes import ResidualRegimeReport, analyze_residual_regimes
from .schedule_context import (
    SCHEDULE_CONTEXT_REQUIRED,
    build_schedule_context_walkforward,
)


@dataclass(frozen=True)
class ResidualRegimeAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    evaluation_games: int
    source_columns_verified: tuple[str, ...]
    persistent_margin_regimes: tuple[str, ...]
    persistent_total_regimes: tuple[str, ...]
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_residual_regime_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    minimum_games_per_season: int = 24,
    bootstrap_iterations: int = 2000,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> ResidualRegimeAudit:
    """Build canonical pregame rows and diagnose stable residual regimes."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )
    missing = sorted(SCHEDULE_CONTEXT_REQUIRED - set(schedules.columns))
    if missing:
        raise ValueError(f"nflverse schedules missing regime source columns: {missing}")

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
    report: ResidualRegimeReport = analyze_residual_regimes(
        dataset,
        test_seasons=test_seasons,
        minimum_games_per_season=minimum_games_per_season,
        bootstrap_iterations=bootstrap_iterations,
    )
    evaluation_games = dataset.filter(
        pl.col("season").is_in(list(test_seasons))
    ).height
    return ResidualRegimeAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        evaluation_games=evaluation_games,
        source_columns_verified=tuple(sorted(SCHEDULE_CONTEXT_REQUIRED)),
        persistent_margin_regimes=report.persistent_margin_regimes,
        persistent_total_regimes=report.persistent_total_regimes,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        report=report.to_dict(),
    )
