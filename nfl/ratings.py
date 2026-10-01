"""Independent NFL fair-score baseline.

This module intentionally has no sportsbook inputs. It estimates league scoring,
home-field advantage, team offense, and team defense from completed football games
available before the target week.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games, schedule_to_team_games

VALIDATED_PRIOR_SEASON_WEIGHT = 0.10


@dataclass(frozen=True)
class FairScoreProjection:
    home_team: str
    away_team: str
    home_points: float
    away_points: float

    @property
    def home_margin(self) -> float:
        return self.home_points - self.away_points

    @property
    def total(self) -> float:
        return self.home_points + self.away_points


class FairScoreModel:
    """Ridge-regularized offense/defense scoring model.

    Expected team points are modeled as::

        league_points + offense(team) - defense(opponent) + home_field

    Team effects are shrunk toward zero with ridge regularization. The intercept and
    home-field coefficient are not penalized. Optional positive sample weights allow
    later games to receive more influence without changing the model specification.
    """

    def __init__(self, ridge: float = 8.0) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be > 0")
        self.ridge = float(ridge)
        self.teams: tuple[str, ...] = ()
        self.league_points: float | None = None
        self.home_field: float | None = None
        self.offense: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.residual_std: float | None = None
        self.training_rows: int = 0
        self.training_weight: float = 0.0

    @property
    def fitted(self) -> bool:
        return self.league_points is not None

    @staticmethod
    def _weights(
        n_rows: int,
        sample_weight: Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_rows, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n_rows,):
            raise DataContractError(
                f"sample_weight must have shape ({n_rows},), got {weights.shape}"
            )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise DataContractError("sample_weight must contain finite positive values")
        return weights

    def fit(
        self,
        team_games: pl.DataFrame,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> FairScoreModel:
        required = {"team", "opponent", "points_for", "is_home"}
        require_columns(team_games, required, "team_games")
        if team_games.height < 4:
            raise DataContractError("fair-score model requires at least four team-game rows")

        teams = sorted(
            set(team_games.get_column("team").drop_nulls().to_list())
            | set(team_games.get_column("opponent").drop_nulls().to_list())
        )
        if len(teams) < 2:
            raise DataContractError("fair-score model requires at least two teams")

        team_index = {team: index for index, team in enumerate(teams)}
        n_rows = team_games.height
        n_teams = len(teams)
        n_columns = 2 + (2 * n_teams)

        x = np.zeros((n_rows, n_columns), dtype=float)
        y = np.asarray(team_games.get_column("points_for").cast(pl.Float64), dtype=float)
        home = np.asarray(team_games.get_column("is_home").cast(pl.Float64), dtype=float)
        team_values = team_games.get_column("team").to_list()
        opponent_values = team_games.get_column("opponent").to_list()
        weights = self._weights(n_rows, sample_weight)

        x[:, 0] = 1.0
        x[:, 1] = home
        for row, (team, opponent) in enumerate(zip(team_values, opponent_values, strict=True)):
            x[row, 2 + team_index[team]] = 1.0
            x[row, 2 + n_teams + team_index[opponent]] = -1.0

        sqrt_weight = np.sqrt(weights)
        weighted_x = x * sqrt_weight[:, None]
        weighted_y = y * sqrt_weight

        penalty = np.eye(n_columns, dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        penalty[1, 1] = 0.0
        beta = np.linalg.solve(
            weighted_x.T @ weighted_x + penalty,
            weighted_x.T @ weighted_y,
        )

        raw_offense = beta[2 : 2 + n_teams]
        raw_defense = beta[2 + n_teams :]
        offense_mean = float(np.mean(raw_offense))
        defense_mean = float(np.mean(raw_defense))

        self.teams = tuple(teams)
        self.league_points = float(beta[0] + offense_mean - defense_mean)
        self.home_field = float(beta[1])
        self.offense = {
            team: float(raw_offense[index] - offense_mean)
            for team, index in team_index.items()
        }
        self.defense = {
            team: float(raw_defense[index] - defense_mean)
            for team, index in team_index.items()
        }
        fitted_values = x @ beta
        residuals = y - fitted_values
        self.residual_std = float(
            np.sqrt(np.sum(weights * np.square(residuals)) / np.sum(weights))
        )
        self.training_rows = n_rows
        self.training_weight = float(np.sum(weights))
        return self

    def _check_team(self, team: str) -> None:
        if not self.fitted:
            raise RuntimeError("fair-score model has not been fitted")
        if team not in self.offense:
            raise DataContractError(f"team {team!r} was not present in the training history")

    def project(self, home_team: str, away_team: str) -> FairScoreProjection:
        self._check_team(home_team)
        self._check_team(away_team)
        assert self.league_points is not None
        assert self.home_field is not None

        home_points = (
            self.league_points
            + self.offense[home_team]
            - self.defense[away_team]
            + self.home_field
        )
        away_points = self.league_points + self.offense[away_team] - self.defense[home_team]

        return FairScoreProjection(
            home_team=home_team,
            away_team=away_team,
            home_points=max(0.0, float(home_points)),
            away_points=max(0.0, float(away_points)),
        )

    def ratings_table(self) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("fair-score model has not been fitted")
        return pl.DataFrame(
            {
                "team": list(self.teams),
                "offense_points": [self.offense[team] for team in self.teams],
                "defense_points": [self.defense[team] for team in self.teams],
            }
        ).with_columns(
            (pl.col("offense_points") + pl.col("defense_points")).alias("net_rating")
        )


def fair_score_history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return canonical scoring history for a target season/week.

    The active season contributes only completed regular-season games strictly before
    the target week. The immediately prior regular season is allowed as a validated
    low-weight prior because all of it was known before the active season began.
    """

    if week < 1:
        raise DataContractError("week must be >= 1")
    history = completed_games(schedules).filter(pl.col("game_type") == "REG")
    return history.filter(
        (pl.col("season") == season - 1)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def pregame_sample_weights(
    team_games: pl.DataFrame,
    season: int,
    week: int,
    *,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
    current_season_half_life: float | None = None,
) -> np.ndarray:
    """Return leakage-safe weights for a pregame scoring history.

    Prior-season rows retain a fixed low weight. When ``current_season_half_life`` is
    supplied, current-season games decay exponentially by completed-week age, with the
    immediately preceding week receiving weight 1.0. A half-life is deliberately not
    enabled by default; it must first earn promotion through chronological validation.
    """

    require_columns(team_games, {"season", "week"}, "team_games")
    if week < 1:
        raise DataContractError("week must be >= 1")
    if not 0 < prior_season_weight <= 1:
        raise ValueError("prior_season_weight must be in (0, 1]")
    if current_season_half_life is not None and current_season_half_life <= 0:
        raise ValueError("current_season_half_life must be > 0")

    seasons = np.asarray(team_games.get_column("season"), dtype=int)
    weeks = np.asarray(team_games.get_column("week"), dtype=int)
    allowed = (seasons == season - 1) | ((seasons == season) & (weeks < week))
    if not np.all(allowed):
        raise DataContractError("team_games contains rows outside the canonical pregame history")

    weights = np.full(team_games.height, float(prior_season_weight), dtype=float)
    current = seasons == season
    if current_season_half_life is None:
        weights[current] = 1.0
    else:
        age = (week - 1) - weeks[current]
        if np.any(age < 0):
            raise DataContractError("current-season history contains target/future week rows")
        weights[current] = np.power(0.5, age / float(current_season_half_life))
    return weights


def fit_pregame_fair_score(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    ridge: float = 8.0,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
    current_season_half_life: float | None = None,
) -> FairScoreModel:
    """Fit the canonical fair-score model at the pregame information boundary."""

    history = fair_score_history(schedules, season, week)
    team_games = schedule_to_team_games(history)
    weights = pregame_sample_weights(
        team_games,
        season,
        week,
        prior_season_weight=prior_season_weight,
        current_season_half_life=current_season_half_life,
    )
    return FairScoreModel(ridge=ridge).fit(team_games, sample_weight=weights)
