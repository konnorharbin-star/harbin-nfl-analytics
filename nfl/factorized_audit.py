"""Real-source audit for the factorized possession × efficiency score architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .factorized_eval import evaluate_factorized_architecture
from .factorized_score import DRIVE_COUNT_REQUIRED


@dataclass(frozen=True)
class FactorizedAudit:
    test_seasons: tuple[int, ...]
    games: int
    source_columns_verified: tuple[str, ...]
    ridge_grid: tuple[float, ...]
    selected_ridge: float | None
    shadow_candidate: bool
    baseline_margin_mae: float
    selected_margin_mae: float
    baseline_margin_rmse: float
    selected_margin_rmse: float
    baseline_total_mae: float
    selected_total_mae: float
    baseline_total_rmse: float
    selected_total_rmse: float
    best_tested_ridge: float
    best_tested_positive_folds: int
    best_tested_passed: bool
    canonical_score_change_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_factorized_audit(
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    ridge_grid: tuple[float, ...] = (2.0, 8.0, 32.0),
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> FactorizedAudit:
    """Run the fixed 2023-2025 architecture comparison on real nflverse data."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    source = client or NFLDataClient()
    first_history_season = min(test_seasons) - 1
    seasons = list(range(first_history_season, max(test_seasons) + 1))
    schedules = source.load_schedules(seasons, refresh=refresh)
    pbp = source.load_pbp(seasons, refresh=refresh)
    missing = sorted(DRIVE_COUNT_REQUIRED - set(pbp.columns))
    if missing:
        raise ValueError(f"nflverse PBP is missing factorized source columns: {missing}")

    evaluation = evaluate_factorized_architecture(
        schedules,
        pbp,
        test_seasons=test_seasons,
        ridge_grid=ridge_grid,
        start_week=start_week,
        end_week=end_week,
    )
    return FactorizedAudit(
        test_seasons=test_seasons,
        games=evaluation.games,
        source_columns_verified=tuple(sorted(DRIVE_COUNT_REQUIRED)),
        ridge_grid=evaluation.ridge_grid,
        selected_ridge=evaluation.selected_ridge,
        shadow_candidate=evaluation.shadow_candidate,
        baseline_margin_mae=evaluation.baseline_margin_mae,
        selected_margin_mae=evaluation.selected_margin_mae,
        baseline_margin_rmse=evaluation.baseline_margin_rmse,
        selected_margin_rmse=evaluation.selected_margin_rmse,
        baseline_total_mae=evaluation.baseline_total_mae,
        selected_total_mae=evaluation.selected_total_mae,
        baseline_total_rmse=evaluation.baseline_total_rmse,
        selected_total_rmse=evaluation.selected_total_rmse,
        best_tested_ridge=evaluation.best_tested_ridge,
        best_tested_positive_folds=evaluation.best_tested_positive_folds,
        best_tested_passed=evaluation.best_tested_passed,
        canonical_score_change_enabled=evaluation.canonical_score_change_enabled,
        promotion_eligible=evaluation.promotion_eligible,
        evaluation=evaluation.to_dict(),
    )
