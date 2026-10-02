"""Real-source rolling audit for the decoupled score architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .decoupled_score import DecoupledArchitectureEvaluation, evaluate_decoupled_architecture


@dataclass(frozen=True)
class DecoupledScoreAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    games: int
    selected_margin_ridge: float | None
    selected_total_ridge: float | None
    margin_shadow_candidate: bool
    total_shadow_candidate: bool
    margin_baseline_mae: float
    margin_selected_mae: float
    margin_baseline_rmse: float
    margin_selected_rmse: float
    total_baseline_mae: float
    total_selected_mae: float
    total_baseline_rmse: float
    total_selected_rmse: float
    canonical_score_change_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_decoupled_score_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> DecoupledScoreAudit:
    """Load nflverse schedules and run the fixed multi-season architecture gate."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(list(seasons), refresh=refresh)
    evaluation: DecoupledArchitectureEvaluation = evaluate_decoupled_architecture(
        schedules,
        test_seasons=test_seasons,
        start_week=start_week,
        end_week=end_week,
    )
    return DecoupledScoreAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        games=evaluation.games,
        selected_margin_ridge=evaluation.selected_margin_ridge,
        selected_total_ridge=evaluation.selected_total_ridge,
        margin_shadow_candidate=evaluation.margin_shadow_candidate,
        total_shadow_candidate=evaluation.total_shadow_candidate,
        margin_baseline_mae=evaluation.margin_baseline_mae,
        margin_selected_mae=evaluation.margin_selected_mae,
        margin_baseline_rmse=evaluation.margin_baseline_rmse,
        margin_selected_rmse=evaluation.margin_selected_rmse,
        total_baseline_mae=evaluation.total_baseline_mae,
        total_selected_mae=evaluation.total_selected_mae,
        total_baseline_rmse=evaluation.total_baseline_rmse,
        total_selected_rmse=evaluation.total_selected_rmse,
        canonical_score_change_enabled=evaluation.canonical_score_change_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
