"""Real-source audit for fixed non-QB personnel context across development folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .data import NFLDataClient, completed_games
from .personnel_context import build_personnel_walkforward_dataset
from .personnel_context_eval import (
    PersonnelRollingEvaluation,
    evaluate_fixed_personnel_rolling,
)

MIN_PERSONNEL_COVERAGE = 0.90


@dataclass(frozen=True)
class PersonnelSeasonCoverage:
    season: int
    expected_games: int
    reconstructed_games: int
    coverage: float
    injury_rows: int
    depth_rows: int
    timestamped_depth_games: int
    weekly_depth_games: int
    season_only_depth_games: int
    source_error: str | None


@dataclass(frozen=True)
class PersonnelContextAudit:
    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    rows: int
    minimum_coverage: float
    source_status: str
    coverage: tuple[PersonnelSeasonCoverage, ...]
    margin_feature_set: str
    margin_ridge_alpha: float | None
    margin_shadow_candidate: bool
    total_feature_set: str
    total_ridge_alpha: float | None
    total_shadow_candidate: bool
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool
    evaluation: dict[str, object] | None
    meaning: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["coverage"] = [asdict(row) for row in self.coverage]
        return value


def _expected_games(
    schedules: pl.DataFrame,
    *,
    season: int,
    start_week: int,
    end_week: int,
) -> int:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") >= start_week)
        & (pl.col("week") <= end_week)
    ).height


def _mode_games(
    frame: pl.DataFrame,
    mode: str,
) -> int:
    if frame.is_empty():
        return 0
    return frame.filter(
        (pl.col("home_depth_temporal_mode") == mode)
        | (pl.col("away_depth_temporal_mode") == mode)
    ).height


def run_personnel_context_audit(
    *,
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    minimum_coverage: float = MIN_PERSONNEL_COVERAGE,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> PersonnelContextAudit:
    """Rebuild historical personnel state, then apply the fixed rolling gate."""

    if not seasons:
        raise ValueError("seasons must not be empty")
    if tuple(sorted(seasons)) != seasons:
        raise ValueError("seasons must be chronological")
    if any(season not in seasons for season in test_seasons):
        raise ValueError("every test season must be included in seasons")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")

    source = client or NFLDataClient()
    schedules = source.load_schedules(
        list(range(min(seasons) - 1, max(seasons) + 1)),
        refresh=refresh,
    )

    frames: list[pl.DataFrame] = []
    coverage_rows: list[PersonnelSeasonCoverage] = []
    for season in seasons:
        expected = _expected_games(
            schedules,
            season=season,
            start_week=start_week,
            end_week=end_week,
        )
        injury_rows = 0
        depth_rows = 0
        error: str | None = None
        frame = pl.DataFrame()
        try:
            injuries = source.load_injuries(
                [season],
                refresh=refresh,
            )
            depth = source.load_depth_charts(
                [season],
                refresh=refresh,
            )
            injury_rows = injuries.height
            depth_rows = depth.height
            frame = build_personnel_walkforward_dataset(
                schedules,
                injuries,
                depth,
                season,
                start_week=start_week,
                end_week=end_week,
                score_ridge=score_ridge,
            )
            frames.append(frame)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        reconstructed = frame.height
        coverage = reconstructed / expected if expected else 0.0
        coverage_rows.append(
            PersonnelSeasonCoverage(
                season=season,
                expected_games=expected,
                reconstructed_games=reconstructed,
                coverage=coverage,
                injury_rows=injury_rows,
                depth_rows=depth_rows,
                timestamped_depth_games=_mode_games(
                    frame,
                    "timestamped",
                ),
                weekly_depth_games=_mode_games(frame, "weekly"),
                season_only_depth_games=_mode_games(
                    frame,
                    "season_only",
                ),
                source_error=error,
            )
        )

    dataset = (
        pl.concat(frames, how="diagonal_relaxed").sort(
            ["season", "week", "game_id"]
        )
        if frames
        else pl.DataFrame()
    )
    source_ready = (
        len(coverage_rows) == len(seasons)
        and all(
            row.source_error is None
            and row.coverage >= minimum_coverage
            for row in coverage_rows
        )
    )

    if not source_ready:
        return PersonnelContextAudit(
            seasons=seasons,
            test_seasons=test_seasons,
            rows=dataset.height,
            minimum_coverage=minimum_coverage,
            source_status="SOURCE_INSUFFICIENT",
            coverage=tuple(coverage_rows),
            margin_feature_set="disabled",
            margin_ridge_alpha=None,
            margin_shadow_candidate=False,
            total_feature_set="disabled",
            total_ridge_alpha=None,
            total_shadow_candidate=False,
            canonical_score_adjustment_enabled=False,
            promotion_eligible=False,
            evaluation=None,
            meaning=(
                "Historical non-QB personnel coverage did not clear the "
                "predeclared 90% development-season gate. No predictive "
                "selection was permitted."
            ),
        )

    evaluation: PersonnelRollingEvaluation = (
        evaluate_fixed_personnel_rolling(
            dataset,
            test_seasons=test_seasons,
        )
    )
    return PersonnelContextAudit(
        seasons=seasons,
        test_seasons=test_seasons,
        rows=dataset.height,
        minimum_coverage=minimum_coverage,
        source_status="READY",
        coverage=tuple(coverage_rows),
        margin_feature_set=evaluation.margin.feature_set,
        margin_ridge_alpha=evaluation.margin.ridge_alpha,
        margin_shadow_candidate=evaluation.margin.shadow_candidate,
        total_feature_set=evaluation.total.feature_set,
        total_ridge_alpha=evaluation.total.ridge_alpha,
        total_shadow_candidate=evaluation.total.shadow_candidate,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
        evaluation=evaluation.to_dict(),
        meaning=(
            "Non-QB starter, offensive-line, skill-position, and defensive "
            "availability were reconstructed without sportsbook inputs. A "
            "fixed spec must improve MAE and RMSE in every development fold; "
            "any survivor remains 2026 SHADOW-only."
        ),
    )
