"""NFL scoring decomposition research model.

Final team points mix repeatable possession scoring with noisier defensive and special-
teams scoring. This module reconstructs possession-team scoreboard gains from PBP,
models that component with the canonical opponent-adjusted score structure, and then
adds a heavily shrunk non-possession remainder. Sportsbook information is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games, schedule_to_team_games
from .ratings import (
    VALIDATED_PRIOR_SEASON_WEIGHT,
    FairScoreModel,
    fair_score_history,
    fit_pregame_fair_score,
    pregame_sample_weights,
)

SCORING_DECOMPOSITION_REQUIRED = {
    "season",
    "week",
    "game_id",
    "posteam",
    "posteam_score",
    "posteam_score_post",
}


@dataclass(frozen=True)
class DecomposedProjection:
    home_team: str
    away_team: str
    home_possession_points: float
    away_possession_points: float
    home_remainder_points: float
    away_remainder_points: float

    @property
    def home_points(self) -> float:
        return self.home_possession_points + self.home_remainder_points

    @property
    def away_points(self) -> float:
        return self.away_possession_points + self.away_remainder_points

    @property
    def home_margin(self) -> float:
        return self.home_points - self.away_points

    @property
    def total(self) -> float:
        return self.home_points + self.away_points


class ConstantRemainderModel:
    """Weighted league-mean forecast for volatile non-possession points."""

    def __init__(self) -> None:
        self.mean_points: float | None = None

    def fit(
        self,
        team_games: pl.DataFrame,
        sample_weight: np.ndarray,
    ) -> ConstantRemainderModel:
        require_columns(team_games, {"points_for"}, "remainder_team_games")
        values = np.asarray(team_games.get_column("points_for"), dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if values.shape != weights.shape:
            raise DataContractError("remainder values and weights have different shapes")
        if not np.isfinite(values).all() or np.any(values < -1e-9):
            raise DataContractError("remainder points must be finite and non-negative")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise DataContractError("remainder weights must be finite and positive")
        self.mean_points = max(0.0, float(np.average(values, weights=weights)))
        return self

    def project(self, home_team: str, away_team: str) -> tuple[float, float]:
        del home_team, away_team
        if self.mean_points is None:
            raise RuntimeError("constant remainder model has not been fitted")
        return self.mean_points, self.mean_points


class RidgeRemainderModel:
    """Strongly regularized team/opponent model for non-possession points."""

    def __init__(self, ridge: float) -> None:
        self.model = FairScoreModel(ridge=ridge)

    def fit(
        self,
        team_games: pl.DataFrame,
        sample_weight: np.ndarray,
    ) -> RidgeRemainderModel:
        self.model.fit(team_games, sample_weight=sample_weight)
        return self

    def project(self, home_team: str, away_team: str) -> tuple[float, float]:
        projection = self.model.project(home_team, away_team)
        return projection.home_points, projection.away_points


class DecomposedScoreModel:
    """Possession-scoring model plus a shrunk non-possession remainder."""

    def __init__(self, *, offensive_ridge: float = 8.0, remainder_ridge: float | None = None) -> None:
        if offensive_ridge <= 0:
            raise ValueError("offensive_ridge must be > 0")
        if remainder_ridge is not None and remainder_ridge <= 0:
            raise ValueError("remainder_ridge must be > 0")
        self.possession = FairScoreModel(ridge=offensive_ridge)
        self.remainder: ConstantRemainderModel | RidgeRemainderModel
        if remainder_ridge is None:
            self.remainder = ConstantRemainderModel()
        else:
            self.remainder = RidgeRemainderModel(remainder_ridge)

    def fit(
        self,
        possession_games: pl.DataFrame,
        remainder_games: pl.DataFrame,
        sample_weight: np.ndarray,
    ) -> DecomposedScoreModel:
        self.possession.fit(possession_games, sample_weight=sample_weight)
        self.remainder.fit(remainder_games, sample_weight=sample_weight)
        return self

    def project(self, home_team: str, away_team: str) -> DecomposedProjection:
        possession = self.possession.project(home_team, away_team)
        home_remainder, away_remainder = self.remainder.project(home_team, away_team)
        return DecomposedProjection(
            home_team=home_team,
            away_team=away_team,
            home_possession_points=possession.home_points,
            away_possession_points=possession.away_points,
            home_remainder_points=max(0.0, float(home_remainder)),
            away_remainder_points=max(0.0, float(away_remainder)),
        )


def possession_scoring_by_team_game(pbp: pl.DataFrame) -> pl.DataFrame:
    """Sum positive possession-team scoreboard deltas for every observed team-game."""

    require_columns(pbp, SCORING_DECOMPOSITION_REQUIRED, "scoring_decomposition_pbp")
    plays = pbp.filter(pl.col("game_id").is_not_null() & pl.col("posteam").is_not_null())
    if plays.is_empty():
        raise DataContractError("no PBP rows are available for possession scoring")
    return (
        plays.with_columns(
            (
                pl.col("posteam_score_post").cast(pl.Float64)
                - pl.col("posteam_score").cast(pl.Float64)
            )
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .alias("_score_delta")
        )
        .group_by(["game_id", "posteam"])
        .agg(pl.col("_score_delta").sum().alias("possession_points"))
        .rename({"posteam": "team"})
    )


def decomposition_training_tables(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return possession and non-possession team-game targets at the pregame boundary."""

    history = fair_score_history(schedules, season, week)
    if history.is_empty():
        raise DataContractError("scoring decomposition has no eligible game history")
    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = pbp.filter(pl.col("game_id").cast(pl.String).is_in(history_ids))
    if history_pbp.is_empty():
        raise DataContractError("scoring decomposition has no PBP for eligible history")

    team_games = schedule_to_team_games(history)
    possession = possession_scoring_by_team_game(history_pbp)
    joined = team_games.join(possession, on=["game_id", "team"], how="left")
    missing = joined.filter(pl.col("possession_points").is_null())
    if missing.height:
        raise DataContractError(
            f"possession scoring is missing for {missing.height} historical team-games"
        )
    joined = joined.with_columns(
        (pl.col("points_for") - pl.col("possession_points")).alias("remainder_points")
    )
    invalid = joined.filter(pl.col("remainder_points") < -1e-6)
    if invalid.height:
        raise DataContractError(
            f"possession scoring exceeds final score for {invalid.height} team-games"
        )
    joined = joined.with_columns(pl.col("remainder_points").clip(lower_bound=0.0))

    possession_games = joined.with_columns(
        pl.col("possession_points").alias("points_for")
    ).drop("possession_points", "remainder_points")
    remainder_games = joined.with_columns(
        pl.col("remainder_points").alias("points_for")
    ).drop("possession_points", "remainder_points")
    return possession_games, remainder_games


def fit_pregame_decomposed_score(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    offensive_ridge: float = 8.0,
    remainder_ridge: float | None = None,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> DecomposedScoreModel:
    """Fit the possession/remainder score decomposition at the pregame boundary."""

    possession_games, remainder_games = decomposition_training_tables(
        schedules,
        pbp,
        season,
        week,
    )
    weights = pregame_sample_weights(
        possession_games,
        season,
        week,
        prior_season_weight=prior_season_weight,
    )
    return DecomposedScoreModel(
        offensive_ridge=offensive_ridge,
        remainder_ridge=remainder_ridge,
    ).fit(possession_games, remainder_games, weights)


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_decomposed_week_predictions(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    remainder_ridge: float | None,
    baseline_ridge: float = 8.0,
    offensive_ridge: float = 8.0,
) -> pl.DataFrame:
    """Compare one decomposition specification with the canonical direct score."""

    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    baseline = fit_pregame_fair_score(schedules, season, week, ridge=baseline_ridge)
    candidate = fit_pregame_decomposed_score(
        schedules,
        pbp,
        season,
        week,
        offensive_ridge=offensive_ridge,
        remainder_ridge=remainder_ridge,
    )

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        direct = baseline.project(home_team, away_team)
        decomposed = candidate.project(home_team, away_team)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        rows.append(
            {
                "season": season,
                "week": week,
                "game_id": str(game["game_id"]),
                "home_team": home_team,
                "away_team": away_team,
                "actual_home_margin": home_score - away_score,
                "actual_total": home_score + away_score,
                "baseline_home_margin": direct.home_margin,
                "baseline_total": direct.total,
                "candidate_home_margin": decomposed.home_margin,
                "candidate_total": decomposed.total,
                "candidate_home_possession_points": decomposed.home_possession_points,
                "candidate_away_possession_points": decomposed.away_possession_points,
                "candidate_home_remainder_points": decomposed.home_remainder_points,
                "candidate_away_remainder_points": decomposed.away_remainder_points,
            }
        )
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_decomposed_walkforward(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    *,
    remainder_ridge: float | None,
    start_week: int = 5,
    end_week: int | None = 18,
    baseline_ridge: float = 8.0,
    offensive_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct one season of decomposition-vs-canonical predictions."""

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
    for target_week in range(start_week, final_week + 1):
        frame = build_decomposed_week_predictions(
            schedules,
            pbp,
            season,
            target_week,
            remainder_ridge=remainder_ridge,
            baseline_ridge=baseline_ridge,
            offensive_ridge=offensive_ridge,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("scoring decomposition walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])
