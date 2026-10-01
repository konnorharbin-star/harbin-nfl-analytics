"""Chronological recency validation for the NFL fair-score baseline.

Recency is treated as a candidate model change, not an assumption. Half-life choices
are selected on one completed validation season and then evaluated once on a later
untouched holdout season. Sportsbook prices are never inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .ratings import fit_pregame_fair_score


@dataclass(frozen=True)
class ScoreMetrics:
    games: int
    margin_mae: float
    margin_rmse: float
    margin_bias: float
    total_mae: float
    total_rmse: float
    total_bias: float


@dataclass(frozen=True)
class RecencyHoldoutEvaluation:
    validation_season: int
    holdout_season: int
    selected_half_life_weeks: float | None
    validation_baseline: ScoreMetrics
    validation_recency: ScoreMetrics | None
    holdout_baseline: ScoreMetrics
    holdout_recency: ScoreMetrics | None
    margin_mae_improvement: float | None
    margin_rmse_improvement: float | None
    total_mae_improvement: float | None
    total_rmse_improvement: float | None
    validation_candidate_pass: bool
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_week_score_predictions(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    ridge: float = 8.0,
    half_life_weeks: float | None = None,
) -> pl.DataFrame:
    """Project one completed week from the canonical pregame scoring history."""

    if week < 2:
        raise ValueError("week must be >= 2")
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()

    model = fit_pregame_fair_score(
        schedules,
        season,
        week,
        ridge=ridge,
        current_season_half_life=half_life_weeks,
    )
    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        projection = model.project(home_team, away_team)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        rows.append(
            {
                "season": season,
                "week": week,
                "game_id": str(game["game_id"]),
                "home_team": home_team,
                "away_team": away_team,
                "projected_home_margin": projection.home_margin,
                "projected_total": projection.total,
                "actual_home_margin": home_score - away_score,
                "actual_total": home_score + away_score,
            }
        )
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_score_walkforward(
    schedules: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = None,
    ridge: float = 8.0,
    half_life_weeks: float | None = None,
) -> pl.DataFrame:
    """Reconstruct a season with an explicit weekly pregame boundary."""

    if start_week < 2:
        raise ValueError("start_week must be >= 2")
    regular = completed_games(schedules).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if regular.is_empty():
        raise DataContractError(f"no completed regular-season games found for {season}")

    max_week = int(regular.get_column("week").max())
    final_week = max_week if end_week is None else min(int(end_week), max_week)
    if final_week < start_week:
        raise ValueError("end_week is before start_week")

    frames: list[pl.DataFrame] = []
    for week in range(start_week, final_week + 1):
        frame = build_week_score_predictions(
            schedules,
            season,
            week,
            ridge=ridge,
            half_life_weeks=half_life_weeks,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("score walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])


def score_predictions(frame: pl.DataFrame) -> ScoreMetrics:
    required = {
        "projected_home_margin",
        "projected_total",
        "actual_home_margin",
        "actual_total",
    }
    require_columns(frame, required, "score_predictions")
    if frame.is_empty():
        raise DataContractError("cannot score an empty prediction frame")

    margin_error = np.asarray(
        frame.get_column("projected_home_margin") - frame.get_column("actual_home_margin"),
        dtype=float,
    )
    total_error = np.asarray(
        frame.get_column("projected_total") - frame.get_column("actual_total"),
        dtype=float,
    )
    return ScoreMetrics(
        games=frame.height,
        margin_mae=float(np.mean(np.abs(margin_error))),
        margin_rmse=float(sqrt(float(np.mean(np.square(margin_error))))),
        margin_bias=float(np.mean(margin_error)),
        total_mae=float(np.mean(np.abs(total_error))),
        total_rmse=float(sqrt(float(np.mean(np.square(total_error))))),
        total_bias=float(np.mean(total_error)),
    )


def _strictly_improves(candidate: ScoreMetrics, baseline: ScoreMetrics) -> bool:
    return (
        candidate.margin_mae < baseline.margin_mae
        and candidate.margin_rmse < baseline.margin_rmse
        and candidate.total_mae < baseline.total_mae
        and candidate.total_rmse < baseline.total_rmse
    )


def _relative_score(candidate: ScoreMetrics, baseline: ScoreMetrics) -> float:
    return (
        candidate.margin_mae / baseline.margin_mae
        + candidate.margin_rmse / baseline.margin_rmse
        + candidate.total_mae / baseline.total_mae
        + candidate.total_rmse / baseline.total_rmse
    )


def evaluate_recency_holdout(
    schedules: pl.DataFrame,
    *,
    validation_season: int,
    holdout_season: int,
    half_life_grid: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0),
    start_week: int = 5,
    ridge: float = 8.0,
) -> RecencyHoldoutEvaluation:
    """Tune recency on one season, then score the later holdout exactly once.

    A half-life becomes eligible only if it improves all four primary score metrics on
    the validation season. The later holdout uses the selected half-life exactly once,
    and promotion requires all four metrics to improve there as well.
    """

    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")
    if not half_life_grid or any(value <= 0 for value in half_life_grid):
        raise ValueError("half_life_grid must contain positive values")

    validation_baseline = score_predictions(
        build_score_walkforward(
            schedules,
            validation_season,
            start_week=start_week,
            ridge=ridge,
        )
    )

    eligible: list[tuple[float, float, ScoreMetrics]] = []
    for half_life in half_life_grid:
        metrics = score_predictions(
            build_score_walkforward(
                schedules,
                validation_season,
                start_week=start_week,
                ridge=ridge,
                half_life_weeks=half_life,
            )
        )
        if _strictly_improves(metrics, validation_baseline):
            eligible.append(
                (_relative_score(metrics, validation_baseline), float(half_life), metrics)
            )

    selected_half_life: float | None = None
    validation_recency: ScoreMetrics | None = None
    if eligible:
        eligible.sort(key=lambda item: (item[0], item[1]))
        _, selected_half_life, validation_recency = eligible[0]

    holdout_baseline = score_predictions(
        build_score_walkforward(
            schedules,
            holdout_season,
            start_week=start_week,
            ridge=ridge,
        )
    )

    holdout_recency: ScoreMetrics | None = None
    improvements: tuple[float | None, float | None, float | None, float | None] = (
        None,
        None,
        None,
        None,
    )
    candidate_pass = False
    if selected_half_life is not None:
        holdout_recency = score_predictions(
            build_score_walkforward(
                schedules,
                holdout_season,
                start_week=start_week,
                ridge=ridge,
                half_life_weeks=selected_half_life,
            )
        )
        improvements = (
            holdout_baseline.margin_mae - holdout_recency.margin_mae,
            holdout_baseline.margin_rmse - holdout_recency.margin_rmse,
            holdout_baseline.total_mae - holdout_recency.total_mae,
            holdout_baseline.total_rmse - holdout_recency.total_rmse,
        )
        candidate_pass = all(value is not None and value > 0 for value in improvements)

    return RecencyHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_half_life_weeks=selected_half_life,
        validation_baseline=validation_baseline,
        validation_recency=validation_recency,
        holdout_baseline=holdout_baseline,
        holdout_recency=holdout_recency,
        margin_mae_improvement=improvements[0],
        margin_rmse_improvement=improvements[1],
        total_mae_improvement=improvements[2],
        total_rmse_improvement=improvements[3],
        validation_candidate_pass=selected_half_life is not None,
        candidate_pass=candidate_pass,
    )
