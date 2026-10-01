"""Factorized NFL fair-score research model.

Instead of regressing final points directly, this module decomposes expected scoring as
shared game possessions times team scoring efficiency per possession. All inputs are
football results or play-by-play known before the target week; sportsbook information
is deliberately excluded.
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

DRIVE_COUNT_REQUIRED = {"season", "week", "game_id", "drive", "posteam"}


@dataclass(frozen=True)
class FactorizedProjection:
    home_team: str
    away_team: str
    shared_drives: float
    home_points_per_drive: float
    away_points_per_drive: float
    home_points: float
    away_points: float

    @property
    def home_margin(self) -> float:
        return self.home_points - self.away_points

    @property
    def total(self) -> float:
        return self.home_points + self.away_points


class SharedPossessionModel:
    """Ridge model for shared game possessions using additive team pace effects."""

    def __init__(self, ridge: float = 8.0) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be > 0")
        self.ridge = float(ridge)
        self.teams: tuple[str, ...] = ()
        self.league_drives: float | None = None
        self.pace: dict[str, float] = {}

    def fit(
        self,
        games: pl.DataFrame,
        sample_weight: np.ndarray | None = None,
    ) -> SharedPossessionModel:
        required = {"home_team", "away_team", "shared_drives"}
        require_columns(games, required, "shared_possession_games")
        if games.height < 4:
            raise DataContractError("shared-possession model requires at least four games")

        teams = sorted(
            set(games.get_column("home_team").drop_nulls().to_list())
            | set(games.get_column("away_team").drop_nulls().to_list())
        )
        if len(teams) < 2:
            raise DataContractError("shared-possession model requires at least two teams")
        index = {team: position for position, team in enumerate(teams)}
        n_rows = games.height
        x = np.zeros((n_rows, 1 + len(teams)), dtype=float)
        x[:, 0] = 1.0
        for row, game in enumerate(games.iter_rows(named=True)):
            x[row, 1 + index[str(game["home_team"])]] += 1.0
            x[row, 1 + index[str(game["away_team"])]] += 1.0
        y = np.asarray(games.get_column("shared_drives"), dtype=float)
        if not np.isfinite(y).all() or np.any(y <= 0):
            raise DataContractError("shared possessions must be finite and positive")

        if sample_weight is None:
            weights = np.ones(n_rows, dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (n_rows,):
                raise DataContractError("shared-possession sample weights have wrong shape")
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise DataContractError("shared-possession weights must be finite and positive")

        sqrt_weight = np.sqrt(weights)
        weighted_x = x * sqrt_weight[:, None]
        weighted_y = y * sqrt_weight
        penalty = np.eye(x.shape[1], dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(
            weighted_x.T @ weighted_x + penalty,
            weighted_x.T @ weighted_y,
        )

        raw_pace = beta[1:]
        pace_mean = float(np.mean(raw_pace))
        self.teams = tuple(teams)
        self.league_drives = float(beta[0] + (2.0 * pace_mean))
        self.pace = {
            team: float(raw_pace[position] - pace_mean)
            for team, position in index.items()
        }
        return self

    def project(self, home_team: str, away_team: str) -> float:
        if self.league_drives is None:
            raise RuntimeError("shared-possession model has not been fitted")
        if home_team not in self.pace or away_team not in self.pace:
            raise DataContractError("target team is missing from shared-possession history")
        value = self.league_drives + self.pace[home_team] + self.pace[away_team]
        return max(1.0, float(value))


class FactorizedScoreModel:
    """Pregame score model equal to shared possessions times opponent-adjusted PPD."""

    def __init__(self, efficiency_ridge: float = 8.0, pace_ridge: float = 8.0) -> None:
        self.efficiency = FairScoreModel(ridge=efficiency_ridge)
        self.pace = SharedPossessionModel(ridge=pace_ridge)

    def fit(
        self,
        efficiency_games: pl.DataFrame,
        possession_games: pl.DataFrame,
        *,
        efficiency_weights: np.ndarray,
        possession_weights: np.ndarray,
    ) -> FactorizedScoreModel:
        self.efficiency.fit(efficiency_games, sample_weight=efficiency_weights)
        self.pace.fit(possession_games, sample_weight=possession_weights)
        return self

    def project(self, home_team: str, away_team: str) -> FactorizedProjection:
        efficiency = self.efficiency.project(home_team, away_team)
        shared_drives = self.pace.project(home_team, away_team)
        home_points = shared_drives * efficiency.home_points
        away_points = shared_drives * efficiency.away_points
        return FactorizedProjection(
            home_team=home_team,
            away_team=away_team,
            shared_drives=shared_drives,
            home_points_per_drive=efficiency.home_points,
            away_points_per_drive=efficiency.away_points,
            home_points=max(0.0, float(home_points)),
            away_points=max(0.0, float(away_points)),
        )


def _drive_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    require_columns(pbp, DRIVE_COUNT_REQUIRED, "factorized_pbp")
    counts = (
        pbp.filter(
            pl.col("game_id").is_not_null()
            & pl.col("drive").is_not_null()
            & pl.col("posteam").is_not_null()
        )
        .group_by(["game_id", "posteam"])
        .agg(pl.col("drive").n_unique().cast(pl.Float64).alias("drives"))
        .rename({"posteam": "team"})
    )
    if counts.is_empty():
        raise DataContractError("no offensive drive counts are available")
    return counts


def _possession_weights(
    games: pl.DataFrame,
    season: int,
    *,
    prior_season_weight: float,
) -> np.ndarray:
    require_columns(games, {"season"}, "possession_games")
    seasons = np.asarray(games.get_column("season"), dtype=int)
    allowed = (seasons == season - 1) | (seasons == season)
    if not np.all(allowed):
        raise DataContractError("possession history contains an unsupported season")
    return np.where(seasons == season, 1.0, float(prior_season_weight))


def factorized_training_tables(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build efficiency and shared-possession training tables at the pregame boundary."""

    history = fair_score_history(schedules, season, week)
    if history.is_empty():
        raise DataContractError("factorized model has no eligible scoring history")
    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = pbp.filter(pl.col("game_id").cast(pl.String).is_in(history_ids))
    if history_pbp.is_empty():
        raise DataContractError("factorized model has no PBP for eligible history games")

    team_games = schedule_to_team_games(history)
    counts = _drive_counts(history_pbp)
    efficiency_games = team_games.join(counts, on=["game_id", "team"], how="left")
    missing = efficiency_games.filter(pl.col("drives").is_null() | (pl.col("drives") <= 0))
    if missing.height:
        raise DataContractError(
            f"factorized history is missing positive drive counts for {missing.height} team-games"
        )
    efficiency_games = efficiency_games.with_columns(
        (pl.col("points_for") / pl.col("drives")).alias("points_for")
    )

    possession_games = (
        efficiency_games.group_by(["season", "week", "game_id"])
        .agg(
            pl.col("drives").mean().alias("shared_drives"),
            pl.col("team").filter(pl.col("is_home")).first().alias("home_team"),
            pl.col("team").filter(~pl.col("is_home")).first().alias("away_team"),
        )
        .sort(["season", "week", "game_id"])
    )
    malformed = possession_games.filter(
        pl.col("home_team").is_null()
        | pl.col("away_team").is_null()
        | pl.col("shared_drives").is_null()
        | (pl.col("shared_drives") <= 0)
    )
    if malformed.height:
        raise DataContractError("factorized possession table contains malformed games")
    return efficiency_games, possession_games


def fit_pregame_factorized_score(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    ridge: float = 8.0,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> FactorizedScoreModel:
    """Fit the factorized model from prior/current completed regular-season history."""

    efficiency_games, possession_games = factorized_training_tables(
        schedules,
        pbp,
        season,
        week,
    )
    efficiency_weights = pregame_sample_weights(
        efficiency_games,
        season,
        week,
        prior_season_weight=prior_season_weight,
    )
    possession_weights = _possession_weights(
        possession_games,
        season,
        prior_season_weight=prior_season_weight,
    )
    return FactorizedScoreModel(efficiency_ridge=ridge, pace_ridge=ridge).fit(
        efficiency_games,
        possession_games,
        efficiency_weights=efficiency_weights,
        possession_weights=possession_weights,
    )


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_factorized_week_predictions(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    factorized_ridge: float = 8.0,
    baseline_ridge: float = 8.0,
) -> pl.DataFrame:
    """Compare direct-score and factorized projections for one completed target week."""

    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    baseline = fit_pregame_fair_score(schedules, season, week, ridge=baseline_ridge)
    factorized = fit_pregame_factorized_score(
        schedules,
        pbp,
        season,
        week,
        ridge=factorized_ridge,
    )

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        direct = baseline.project(home_team, away_team)
        decomposed = factorized.project(home_team, away_team)
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
                "factorized_home_margin": decomposed.home_margin,
                "factorized_total": decomposed.total,
                "factorized_home_points": decomposed.home_points,
                "factorized_away_points": decomposed.away_points,
                "factorized_shared_drives": decomposed.shared_drives,
                "factorized_home_ppd": decomposed.home_points_per_drive,
                "factorized_away_ppd": decomposed.away_points_per_drive,
            }
        )
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_factorized_walkforward(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = 18,
    factorized_ridge: float = 8.0,
    baseline_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct one season of direct and factorized pregame projections."""

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
        frame = build_factorized_week_predictions(
            schedules,
            pbp,
            season,
            target_week,
            factorized_ridge=factorized_ridge,
            baseline_ridge=baseline_ridge,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("factorized walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])
