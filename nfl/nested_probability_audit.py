"""Real-source audit for whole-week nested NFL probability validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient
from .nested_probability import NestedProbabilityEvaluation, evaluate_nested_probability
from .recency import build_score_walkforward


@dataclass(frozen=True)
class NestedProbabilityAudit:
    seasons: tuple[int, ...]
    rows: int
    evaluation: NestedProbabilityEvaluation
    source_contract_pass: bool
    canonical_probability_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evaluation"] = self.evaluation.to_dict()
        return result


def run_nested_probability_audit(
    *,
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> NestedProbabilityAudit:
    """Reconstruct canonical pregame scores and apply the Stage 23 protocol."""

    source = client or NFLDataClient()
    schedule_seasons = tuple(sorted({min(seasons) - 1, *seasons}))
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    frames = [
        build_score_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
            ridge=score_ridge,
        )
        for season in seasons
    ]
    dataset = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    evaluation = evaluate_nested_probability(dataset)
    return NestedProbabilityAudit(
        seasons=seasons,
        rows=dataset.height,
        evaluation=evaluation,
        source_contract_pass=True,
        canonical_probability_change_enabled=False,
        promotion_eligible=False,
    )
