"""Real-source audit for the NCAA-style NFL residual ensemble."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .dataset import build_walkforward_dataset
from .ensemble_residuals import evaluate_rolling_ensemble


@dataclass(frozen=True)
class EnsembleAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    margin: dict[str, object]
    total: dict[str, object]
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    reserved_forward_season: int
    meaning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_ensemble_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> EnsembleAudit:
    """Build leak-free historical rows and evaluate the ensemble architecture."""

    if not seasons or min(test_seasons) <= min(seasons):
        raise ValueError("seasons must include at least one pre-evaluation training season")

    source = client or NFLDataClient()
    schedule_seasons = tuple(range(min(seasons) - 1, max(seasons) + 1))
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    pbp = source.load_pbp(seasons, refresh=refresh)

    frames = [
        build_walkforward_dataset(
            schedules,
            pbp,
            season,
            start_week=start_week,
            end_week=end_week,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])

    margin = evaluate_rolling_ensemble(dataset, target="margin", test_seasons=test_seasons)
    total = evaluate_rolling_ensemble(dataset, target="total", test_seasons=test_seasons)

    return EnsembleAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        margin=margin.to_dict(),
        total=total.to_dict(),
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        reserved_forward_season=2026,
        meaning=(
            "The ridge plus histogram-boosting architecture is development-only. Each "
            "fold selects its residual mix and weight using strictly earlier football "
            "data. A target may justify a 2026 shadow only if every 2023-2025 "
            "evaluation fold improves both MAE and RMSE. Canonical scores never change "
            "in this audit."
        ),
    )
