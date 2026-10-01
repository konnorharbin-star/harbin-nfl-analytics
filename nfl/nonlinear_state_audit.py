"""Real-source audit for nonlinear NFL online-state residual research."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .nonlinear_state_fixed import (
    NonlinearStateRollingEvaluation,
    evaluate_nonlinear_state_rolling,
)
from .online_state import ONLINE_STATE_CONFIGS
from .online_state_dataset import build_online_state_dataset


@dataclass(frozen=True)
class NonlinearStateAudit:
    source_seasons: tuple[int, ...]
    dataset_rows_by_config: dict[str, int]
    evaluation_games: int
    evaluation: NonlinearStateRollingEvaluation
    source_contract_pass: bool
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evaluation"] = self.evaluation.to_dict()
        return result


def run_nonlinear_state_audit(
    *,
    source_seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    start_season: int = 2022,
    end_season: int = 2025,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> NonlinearStateAudit:
    """Build every online-state profile and evaluate nonlinear fixed candidates."""

    source = client or NFLDataClient()
    schedules = source.load_schedules(source_seasons, refresh=refresh)
    datasets = {
        config.name: build_online_state_dataset(
            schedules,
            config,
            start_season=start_season,
            end_season=end_season,
            start_week=5,
            end_week=18,
        )
        for config in ONLINE_STATE_CONFIGS
    }
    evaluation = evaluate_nonlinear_state_rolling(
        datasets,
        test_seasons=test_seasons,
    )
    first = next(iter(datasets.values()))
    evaluation_games = first.filter(first.get_column("season").is_in(test_seasons)).height
    return NonlinearStateAudit(
        source_seasons=source_seasons,
        dataset_rows_by_config={name: frame.height for name, frame in datasets.items()},
        evaluation_games=evaluation_games,
        evaluation=evaluation,
        source_contract_pass=True,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
    )
