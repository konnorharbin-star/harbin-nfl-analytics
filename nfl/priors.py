"""Prior-season information experiments for the NFL fair-score baseline.

The production baseline currently fits only the active season. This module tests a
narrow early-season prior: completed regular-season games from the immediately prior
season may enter with a small fixed sample weight, while current-season games retain
full weight. Sportsbook data is never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games, schedule_to_team_games
from .ratings import FairScoreModel
from .recency import ScoreMetrics, build_score_walkforward, score_predictions


@dataclass(frozen=True)
class PriorHoldoutEvaluation:
    validation_season: int
    holdout_season: int
    selected_prior_weight: float
    validation_baseline: ScoreMetrics
    validation_prior: ScoreMetrics
    holdout_baseline: ScoreMetrics
    holdout_prior: ScoreMetrics
    margin_mae_improvement: float
    margin_rmse_improvement: float
    total_mae_improvement: float
    total_rmse_improvement: float
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def prior_season_weights(
    team_games: pl.DataFrame,
    *,
    season: int,
    prior_weight: float,
) -> np.ndarray:
    """Give current-season rows weight 1 and previous-season rows ``prior_weight``."""

    require_columns(team_games, {"season"}, "team_games")
    if not 0 < prior_weight <= 1:
        raise ValueError("prior_weight must be in (0, 1]")

    seasons = np.asarray(team_games.get_column("season"), dtype=int)
    valid = (seasons == season) | (seasons == season - 1)
    if not valid.all():
        raise DataContractError("prior-season history contains an unsupported season")
    return np.where(seasons == season, 1.0, float(prior_weight))


def _history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    require_columns(schedules, {"season", "week", "game_type"}, "schedules")
    if week < 2:
        raise ValueError("week must be >= 2")
    games = completed_games(schedules).filter(pl.col("game_type") == "REG")
    return games.filter(
        (pl.col("season") == season - 1)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def _targets(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_week_prior_predictions(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    prior_weight: float,
    ridge: float = 8.0,
) -> pl.DataFrame:
    """Project one week using prior-season games as a downweighted football prior."""

    history = _history(schedules, season, week)
    targets = _targets(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if history.is_empty():
        raise DataContractError("no prior/current history is available")

    team_games = schedule_to_team_games(history)
    weights = prior_season_weights(team_games, season=season, prior_weight=prior_weight)
    model = FairScoreModel(ridge=ridge).fit(team_games, sample_weight=weights)

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


def build_prior_walkforward(
    schedules: pl.DataFrame,
    season: int,
    *,
    prior_weight: float,
    start_week: int = 2,
    end_week: int | None = None,
    ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct a season with an immediately-prior-season scoring prior."""

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
        frame = build_week_prior_predictions(
            schedules,
            season,
            week,
            prior_weight=prior_weight,
            ridge=ridge,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("prior walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])


def _selection_objective(metrics: ScoreMetrics) -> float:
    return metrics.margin_rmse + metrics.total_rmse


def evaluate_prior_holdout(
    schedules: pl.DataFrame,
    *,
    validation_season: int,
    holdout_season: int,
    prior_weight_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.35),
    start_week: int = 2,
    ridge: float = 8.0,
) -> PriorHoldoutEvaluation:
    """Select prior strength on validation, then score one later holdout once."""

    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")
    if not prior_weight_grid or any(not 0 < value <= 1 for value in prior_weight_grid):
        raise ValueError("prior_weight_grid must contain values in (0, 1]")

    validation_baseline = score_predictions(
        build_score_walkforward(
            schedules,
            validation_season,
            start_week=start_week,
            ridge=ridge,
        )
    )

    candidates: list[tuple[float, float, ScoreMetrics]] = []
    for weight in prior_weight_grid:
        metrics = score_predictions(
            build_prior_walkforward(
                schedules,
                validation_season,
                prior_weight=weight,
                start_week=start_week,
                ridge=ridge,
            )
        )
        candidates.append((_selection_objective(metrics), float(weight), metrics))
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, selected_weight, validation_prior = candidates[0]

    holdout_baseline = score_predictions(
        build_score_walkforward(
            schedules,
            holdout_season,
            start_week=start_week,
            ridge=ridge,
        )
    )
    holdout_prior = score_predictions(
        build_prior_walkforward(
            schedules,
            holdout_season,
            prior_weight=selected_weight,
            start_week=start_week,
            ridge=ridge,
        )
    )

    margin_mae_improvement = holdout_baseline.margin_mae - holdout_prior.margin_mae
    margin_rmse_improvement = holdout_baseline.margin_rmse - holdout_prior.margin_rmse
    total_mae_improvement = holdout_baseline.total_mae - holdout_prior.total_mae
    total_rmse_improvement = holdout_baseline.total_rmse - holdout_prior.total_rmse
    candidate_pass = all(
        value > 0
        for value in (
            margin_mae_improvement,
            margin_rmse_improvement,
            total_mae_improvement,
            total_rmse_improvement,
        )
    )

    return PriorHoldoutEvaluation(
        validation_season=validation_season,
        holdout_season=holdout_season,
        selected_prior_weight=selected_weight,
        validation_baseline=validation_baseline,
        validation_prior=validation_prior,
        holdout_baseline=holdout_baseline,
        holdout_prior=holdout_prior,
        margin_mae_improvement=margin_mae_improvement,
        margin_rmse_improvement=margin_rmse_improvement,
        total_mae_improvement=total_mae_improvement,
        total_rmse_improvement=total_rmse_improvement,
        candidate_pass=candidate_pass,
    )
