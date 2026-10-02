"""Real-source rolling audit for NCAA-style online NFL ratings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .online_ratings import OnlineRatingsEvaluation, evaluate_online_ratings


@dataclass(frozen=True)
class OnlineRatingsAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    games: int
    selected_margin_config: dict[str, float] | None
    selected_total_config: dict[str, float] | None
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


def _config_dict(config) -> dict[str, float] | None:
    if config is None:
        return None
    return {
        "update_alpha": float(config.update_alpha),
        "offseason_carry": float(config.offseason_carry),
        "home_field": float(config.home_field),
    }


def run_online_ratings_audit(
    *,
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> OnlineRatingsAudit:
    """Load nflverse schedules and run the fixed NCAA-style architecture gate."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    source = client or NFLDataClient()
    schedules = source.load_schedules(list(seasons), refresh=refresh)
    evaluation: OnlineRatingsEvaluation = evaluate_online_ratings(
        schedules,
        test_seasons=test_seasons,
        warmup_start=min(seasons),
        start_week=start_week,
        end_week=end_week,
    )
    return OnlineRatingsAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        games=evaluation.games,
        selected_margin_config=_config_dict(evaluation.selected_margin_config),
        selected_total_config=_config_dict(evaluation.selected_total_config),
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
