"""Opponent-adjusted play-by-play efficiency ratings.

Raw team PBP averages confound team quality with schedule strength. This module fits
small ridge offense/defense decompositions to game-level efficiency observations so
pregame matchup signals are adjusted for the opponents already faced. Sportsbook
prices are not inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .advanced import _eligible_scrimmage_plays
from .contracts import DataContractError, require_columns

PBP_METRICS = (
    "epa_per_play",
    "success_rate",
    "pass_epa_per_dropback",
    "rush_epa_per_attempt",
    "explosive_rate",
    "early_down_epa",
)

METRIC_COUNT_COLUMNS = {
    "epa_per_play": "plays",
    "success_rate": "plays",
    "pass_epa_per_dropback": "pass_plays",
    "rush_epa_per_attempt": "rush_plays",
    "explosive_rate": "plays",
    "early_down_epa": "early_down_plays",
}


def team_game_pbp_metrics(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate eligible PBP to one offense-vs-defense observation per game."""

    plays = _eligible_scrimmage_plays(pbp)
    if plays.is_empty():
        raise DataContractError("no eligible scrimmage plays are available")

    return (
        plays.group_by(["game_id", "posteam", "defteam"])
        .agg(
            pl.len().alias("plays"),
            pl.col("_dropback").cast(pl.Int64).sum().alias("pass_plays"),
            pl.col("_rush").cast(pl.Int64).sum().alias("rush_plays"),
            pl.col("_early_down").cast(pl.Int64).sum().alias("early_down_plays"),
            pl.col("epa").mean().alias("epa_per_play"),
            pl.col("_success").mean().alias("success_rate"),
            pl.when(pl.col("_dropback"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("pass_epa_per_dropback"),
            pl.when(pl.col("_rush"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("rush_epa_per_attempt"),
            pl.col("_explosive").mean().alias("explosive_rate"),
            pl.when(pl.col("_early_down"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("early_down_epa"),
        )
        .rename({"posteam": "offense", "defteam": "defense"})
        .sort(["game_id", "offense"])
    )


@dataclass(frozen=True)
class MetricRating:
    metric: str
    league_value: float
    offense: dict[str, float]
    defense: dict[str, float]
    observations: int
    effective_weight: float

    def matchup_value(self, offense_team: str, defense_team: str) -> float:
        try:
            return (
                self.league_value
                + self.offense[offense_team]
                - self.defense[defense_team]
            )
        except KeyError as exc:
            raise DataContractError(
                f"team {exc.args[0]!r} missing from opponent-adjusted {self.metric} state"
            ) from exc


class OpponentAdjustedPBPModel:
    """Ridge offense/defense decomposition for several PBP efficiency metrics."""

    def __init__(self, ridge: float = 12.0) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be > 0")
        self.ridge = float(ridge)
        self.ratings: dict[str, MetricRating] = {}

    @property
    def fitted(self) -> bool:
        return len(self.ratings) == len(PBP_METRICS)

    def _fit_metric(self, frame: pl.DataFrame, metric: str) -> MetricRating:
        count_col = METRIC_COUNT_COLUMNS[metric]
        data = frame.filter(
            pl.col(metric).is_not_null()
            & pl.col(metric).is_finite()
            & (pl.col(count_col) > 0)
        )
        if data.height < 4:
            raise DataContractError(f"insufficient observations for {metric}")

        teams = sorted(
            set(data.get_column("offense").to_list())
            | set(data.get_column("defense").to_list())
        )
        if len(teams) < 2:
            raise DataContractError(f"{metric} requires at least two teams")
        team_index = {team: index for index, team in enumerate(teams)}
        n_teams = len(teams)

        x = np.zeros((data.height, 1 + (2 * n_teams)), dtype=float)
        x[:, 0] = 1.0
        for row, (offense, defense) in enumerate(
            zip(
                data.get_column("offense").to_list(),
                data.get_column("defense").to_list(),
                strict=True,
            )
        ):
            x[row, 1 + team_index[offense]] = 1.0
            x[row, 1 + n_teams + team_index[defense]] = -1.0

        y = np.asarray(data.get_column(metric), dtype=float)
        weights = np.asarray(data.get_column(count_col), dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise DataContractError(f"invalid observations for {metric}")

        sqrt_weight = np.sqrt(weights)
        weighted_x = x * sqrt_weight[:, None]
        weighted_y = y * sqrt_weight
        penalty = np.eye(x.shape[1], dtype=float) * self.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(
            weighted_x.T @ weighted_x + penalty,
            weighted_x.T @ weighted_y,
        )

        raw_offense = beta[1 : 1 + n_teams]
        raw_defense = beta[1 + n_teams :]
        offense_mean = float(np.mean(raw_offense))
        defense_mean = float(np.mean(raw_defense))
        league_value = float(beta[0] + offense_mean - defense_mean)
        offense = {
            team: float(raw_offense[index] - offense_mean)
            for team, index in team_index.items()
        }
        defense = {
            team: float(raw_defense[index] - defense_mean)
            for team, index in team_index.items()
        }
        return MetricRating(
            metric=metric,
            league_value=league_value,
            offense=offense,
            defense=defense,
            observations=data.height,
            effective_weight=float(np.sum(weights)),
        )

    def fit(self, pbp: pl.DataFrame) -> OpponentAdjustedPBPModel:
        frame = team_game_pbp_metrics(pbp)
        self.ratings = {
            metric: self._fit_metric(frame, metric)
            for metric in PBP_METRICS
        }
        return self

    def matchup_signals(self, home_team: str, away_team: str) -> dict[str, float]:
        """Return structurally distinct margin and total signals for one matchup.

        Margin signals compare the two expected efficiency matchups. Total signals
        sum their deviations from league average, allowing a high-offense/weak-defense
        game to differ from a low-offense/strong-defense game even when the side
        advantage is similar.
        """

        if not self.fitted:
            raise RuntimeError("opponent-adjusted PBP model has not been fitted")

        signals: dict[str, float] = {}
        for metric in PBP_METRICS:
            rating = self.ratings[metric]
            home_value = rating.matchup_value(home_team, away_team)
            away_value = rating.matchup_value(away_team, home_team)
            signals[f"oa_{metric}_margin_signal"] = home_value - away_value
            signals[f"oa_{metric}_total_signal"] = (
                home_value + away_value - (2.0 * rating.league_value)
            )
        return signals


def opponent_adjusted_feature_columns(
    metrics: tuple[str, ...],
    *,
    target: str,
) -> tuple[str, ...]:
    """Return target-specific feature columns for a chosen metric subset."""

    unknown = set(metrics) - set(PBP_METRICS)
    if unknown:
        raise ValueError(f"unsupported PBP metrics: {sorted(unknown)}")
    if target == "margin_residual":
        suffix = "margin_signal"
    elif target == "total_residual":
        suffix = "total_signal"
    else:
        raise ValueError(f"unsupported residual target: {target}")
    return tuple(f"oa_{metric}_{suffix}" for metric in metrics)
