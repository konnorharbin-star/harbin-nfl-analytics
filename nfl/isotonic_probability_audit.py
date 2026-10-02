"""Real-source audit for Stage 25 isotonic home-win calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .isotonic_probability import (
    IsotonicProbabilityEvaluation,
    evaluate_isotonic_probability,
)
from .recency import build_score_walkforward


@dataclass(frozen=True)
class IsotonicProbabilityAudit:
    seasons: tuple[int, ...]
    rows: int
    evaluation: IsotonicProbabilityEvaluation
    source_contract_pass: bool
    canonical_probability_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["evaluation"] = self.evaluation.to_dict()
        return out


def run_isotonic_probability_audit(
    *,
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> IsotonicProbabilityAudit:
    source = client or NFLDataClient()
    schedules = source.load_schedules(
        tuple(sorted({min(seasons) - 1, *seasons})),
        refresh=refresh,
    )
    frames = [
        build_score_walkforward(
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
    evaluation = evaluate_isotonic_probability(dataset)
    return IsotonicProbabilityAudit(
        seasons=seasons,
        rows=dataset.height,
        evaluation=evaluation,
        source_contract_pass=True,
        canonical_probability_change_enabled=False,
        promotion_eligible=False,
    )
