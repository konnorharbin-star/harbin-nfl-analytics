"""Chronological recency-weighting experiments for the NFL fair-score baseline.

This module tests one narrow hypothesis: recent completed games may deserve more
weight than older games when estimating team scoring strength. The experiment stays
inside the football-only fair-score layer. Sportsbook prices are never inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games, schedule_to_team_games
from .ratings import FairScoreModel


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
    selected_half_life_weeks: float
    validation_baseline: ScoreMetrics
    validation_recency: ScoreMetrics
    holdout_baseline: ScoreMetrics
    holdout_recency: ScoreMetrics
    margin_mae_improvement: float
    margin_rmse_improvement: float
    total_mae_improvement: float
    total_rmse_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def exponential_week_weights(
    team_games: pl.DataFrame,
    *,
    target_week: int,
    half_life_weeks: float,
) -> np.ndarray:
    """Return positive exponential weights anchored to the latest eligible week.

    A game from ``target_week - 1`` receives weight 1.0. Every ``half_life_weeks``
    of additional age halves the weight. The function fails closed if target/future
    rows are supplied.
    """

    require_columns(team_games, {"week"}, "team_games")
    if target_week < 2:
        raise ValueError("target_week must be >= 2")
    if half_life_weeks <= 0:
        raise ValueError("half_life_weeks must be > 0")

    weeks = np.asarray(team_games.get_column("week"), dtype=float)
    ages = (target_week - 1) - weeks
    if not np.isfinite(ages).all() or np.any(ages < 0):
        raise DataContractError("recency history contains target/future or invalid week rows")
    return np.power(0.5, ages / float(half_life_weeks))


def _regular_history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    require_columns(
        schedules,
        {"season", "week", "game_type", "game_id"},
        "schedules",
    )
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") < week)
    )


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
    """Project one completed week from strictly earlier same-season games."""

    if week < 2:
        raise ValueError("week must be >= 2")
    history = _regular_history(schedules, season, week)
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if history.height < 2:
        raise DataContractError("at least two prior regular-season games are required")

    team_games = schedule_to_team_games(history)
    sample_weight = None
    if half_life_weeks is not None:
        sample_weight = exponential_week_weights(
            team_games,
            target_week=week,
            half_life_weeks=half_life_weeks,
        )
    model = FairScoreModel(ridge=ridge).fit(team_games, sample_weight=sample_weight)

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
    """Reconstruct a season of score projections with an explicit pregame boundary."""

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


def _selection_objective(metrics: ScoreMetrics) -> float:
    """Joint score objective used only on the validation season."""

    return metrics.margin_rmse + metrics.total_rmse


def evaluate_recency_holdout(
    schedules: pl.DataFrame,
    *,
    validation_season: int,
    holdout_season: int,
    half_life_grid: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0),
    start_week: int = 5,
    ridge: float = 8.0,
) -> RecencyHoldoutEvaluation:
    """Tune recency on one season, then score the later holdout exactly once."""

    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")
    if not half_life_grid or any(value <= 0 for value in half_life_grid):
        raise ValueError("half_life_grid must contain positive values")

    validation_baseline_frame = build_score_walkforward(
        schedules,
        validation_season,
        start_week=start_week,
        ridge=ridge,
    )
    validation_baseline = score_predictions(validation_baseline_frame)

    candidates: list[tuple[float, float, ScoreMetrics]] = []
    for half_life in half_life_grid:
        frame = build_score_walkforward(
            schedules,
            validation_season,
            start_week=start_week,
            ridge=ridge,
            half_life_weeks=half_life,
        )
        metrics = score_predictions(frame)
        candidates.append((_selection_objective(metrics), float(half_life), metrics))
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, selected_half_life, validation_recency = candidates[0]

    holdout_baseline = score_predictions(
        build_score_walkforward(
            schedules,
            holdout_season,
            start_week=start_week,
            ridge=ridge,
        )
    )
    holdout_recency = score_predictions(
        build_score_walkforward(
            schedules,
            holdout_season,
            start_week=start_week,
            ridge=ridge,
            half_life_weeks=selected_half_life,
        )
    )

    margin_mae_improvement = holdout_baseline.margin_mae - holdout_recency.margin_mae
    margin_rmse_improvement = holdout_baseline.margin_rmse - holdout_recency.margin_rmse
    total_mae_improvement = holdout_baseline.total_mae - holdout_recency.total_mae
    total_rmse_improvement = holdout_baseline.total_rmse - holdout_recency.total_rmse
    candidate_pass = all(
        value > 0
        for value in (
            margin_mae_improvement,
            margin_rmse_improvement,
            total_mae_improvement,
            total_rmse_improvement,
        )
    )

    return RecencyHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_half_life_weeks=selected_half_life,
        validation_baseline=validation_baseline,
        validation_recency=validation_recency,
        holdout_baseline=holdout_baseline,
        holdout_recency=holdout_recency,
        margin_mae_improvement=margin_mae_improvement,
        margin_rmse_improvement=margin_rmse_improvement,
        total_mae_improvement=total_mae_improvement,
        total_rmse_improvement=total_rmse_improvement,
        candidate_pass=candidate_pass,
    )
