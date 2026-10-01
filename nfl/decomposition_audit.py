"""Real-source audit for possession/non-possession NFL scoring decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .decomposition_eval import evaluate_scoring_decomposition
from .scoring_decomposition import SCORING_DECOMPOSITION_REQUIRED


@dataclass(frozen=True)
class DecompositionAudit:
    test_seasons: tuple[int, ...]
    games: int
    source_columns_verified: tuple[str, ...]
    selected_spec: str
    selected_remainder_ridge: float | None
    shadow_candidate: bool
    baseline_margin_mae: float
    selected_margin_mae: float
    baseline_margin_rmse: float
    selected_margin_rmse: float
    baseline_total_mae: float
    selected_total_mae: float
    baseline_total_rmse: float
    selected_total_rmse: float
    best_tested_spec: str
    best_tested_positive_folds: int
    best_tested_passed: bool
    canonical_score_change_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_decomposition_audit(
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> DecompositionAudit:
    """Run the fixed Stage 20 score decomposition audit on real nflverse data."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    source = client or NFLDataClient()
    first_history_season = min(test_seasons) - 1
    seasons = list(range(first_history_season, max(test_seasons) + 1))
    schedules = source.load_schedules(seasons, refresh=refresh)
    pbp = source.load_pbp(seasons, refresh=refresh)
    missing = sorted(SCORING_DECOMPOSITION_REQUIRED - set(pbp.columns))
    if missing:
        raise ValueError(f"nflverse PBP is missing decomposition source columns: {missing}")

    evaluation = evaluate_scoring_decomposition(
        schedules,
        pbp,
        test_seasons=test_seasons,
        start_week=start_week,
        end_week=end_week,
    )
    return DecompositionAudit(
        test_seasons=test_seasons,
        games=evaluation.games,
        source_columns_verified=tuple(sorted(SCORING_DECOMPOSITION_REQUIRED)),
        selected_spec=evaluation.selected_spec,
        selected_remainder_ridge=evaluation.selected_remainder_ridge,
        shadow_candidate=evaluation.shadow_candidate,
        baseline_margin_mae=evaluation.baseline_margin_mae,
        selected_margin_mae=evaluation.selected_margin_mae,
        baseline_margin_rmse=evaluation.baseline_margin_rmse,
        selected_margin_rmse=evaluation.selected_margin_rmse,
        baseline_total_mae=evaluation.baseline_total_mae,
        selected_total_mae=evaluation.selected_total_mae,
        baseline_total_rmse=evaluation.baseline_total_rmse,
        selected_total_rmse=evaluation.selected_total_rmse,
        best_tested_spec=evaluation.best_tested_spec,
        best_tested_positive_folds=evaluation.best_tested_positive_folds,
        best_tested_passed=evaluation.best_tested_passed,
        canonical_score_change_enabled=evaluation.canonical_score_change_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
