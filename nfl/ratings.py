"""Independent NFL fair-score baseline.

This module intentionally has no sportsbook inputs. It estimates league scoring,
home-field advantage, team offense, and team defense from completed football games
available before the target week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import pregame_history, schedule_to_team_games


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
    home-field coefficient are not penalized.
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

    @property
    def fitted(self) -> bool:
        return self.league_points is not None

    def fit(self, team_games: pl.DataFrame) -> FairScoreModel:
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

        x[:, 0] = 1.0
        x[:, 1] = home
        for row, (team, opponent) in enumerate(zip(team_values, opponent_values, strict=True)):
            x[row, 2 + team_index[team]] = 1.0
            x[row, 2 + n_teams + team_index[opponent]] = -1.0

        penalty = np.eye(n_columns, dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        penalty[1, 1] = 0.0
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)

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
        self.residual_std = float(np.sqrt(np.mean(np.square(residuals))))
        self.training_rows = n_rows
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


def fit_pregame_fair_score(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    ridge: float = 8.0,
) -> FairScoreModel:
    """Fit the fair-score baseline using only games strictly before target week."""

    history = pregame_history(schedules, season, week)
    team_games = schedule_to_team_games(history)
    return FairScoreModel(ridge=ridge).fit(team_games)
