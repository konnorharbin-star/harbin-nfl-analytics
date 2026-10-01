"""Leakage-safe recent-form NFL play-by-play efficiency state.

The NCAA model maintains both season-to-date and recent-form efficiency. This module
adds the NFL-native equivalent without copying a college coefficient. It computes one
pregame EWMA state per team from completed historical game-level PBP observations.
Sportsbook prices are not inputs.
"""

from __future__ import annotations

import math

import polars as pl

from .advanced import _eligible_scrimmage_plays
from .contracts import DataContractError

RECENT_PBP_METRICS = (
    "epa_per_play",
    "success_rate",
    "pass_epa_per_dropback",
    "rush_epa_per_attempt",
    "explosive_rate",
    "early_down_epa",
)


def team_game_recent_metrics(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate eligible PBP to chronological offense-vs-defense game observations."""

    plays = _eligible_scrimmage_plays(pbp)
    if plays.is_empty():
        raise DataContractError("no eligible scrimmage plays are available")

    return (
        plays.group_by(["season", "week", "game_id", "posteam", "defteam"])
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
        .sort(["season", "week", "game_id", "offense"])
    )


def _ewma_role(
    games: pl.DataFrame,
    *,
    team_column: str,
    prefix: str,
    alpha: float,
) -> pl.DataFrame:
    state: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    for row in games.sort(["season", "week", "game_id"]).iter_rows(named=True):
        team = str(row[team_column])
        team_state = state.setdefault(team, {})
        counts[team] = counts.get(team, 0) + 1
        for metric in RECENT_PBP_METRICS:
            value = row.get(metric)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            previous = team_state.get(metric)
            team_state[metric] = (
                number if previous is None else ((1.0 - alpha) * previous) + (alpha * number)
            )

    rows: list[dict[str, object]] = []
    for team in sorted(state):
        team_state = state[team]
        row: dict[str, object] = {"team": team, f"recent_{prefix}_games": counts[team]}
        for metric in RECENT_PBP_METRICS:
            if metric not in team_state:
                raise DataContractError(
                    f"team {team!r} has no finite recent-form observation for {metric}"
                )
            suffix = f"{metric}_allowed" if prefix == "def" else metric
            row[f"recent_{prefix}_{suffix}"] = team_state[metric]
        rows.append(row)
    if not rows:
        raise DataContractError(f"no recent-form {prefix} state is available")
    return pl.DataFrame(rows)


def team_recent_pbp_features(pbp: pl.DataFrame, *, alpha: float = 0.35) -> pl.DataFrame:
    """Return one recent-form offense/defense row per team.

    ``alpha`` controls the weight on the newest completed game. It is exposed as a
    research parameter and is not promoted merely because the NCAA stack uses a
    similar recent-form concept.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    games = team_game_recent_metrics(pbp)
    offense = _ewma_role(games, team_column="offense", prefix="off", alpha=alpha)
    defense = _ewma_role(games, team_column="defense", prefix="def", alpha=alpha)
    return offense.join(defense, on="team", how="full", coalesce=True).sort("team")


def recent_matchup_signals(
    home: dict[str, object],
    away: dict[str, object],
) -> dict[str, float]:
    """Return recent-form margin and total signals for one matchup."""

    values: dict[str, float] = {}
    for metric in RECENT_PBP_METRICS:
        offense_key = f"recent_off_{metric}"
        defense_key = f"recent_def_{metric}_allowed"
        try:
            home_matchup = float(home[offense_key]) - float(away[defense_key])
            away_matchup = float(away[offense_key]) - float(home[defense_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(f"missing/non-numeric recent-form feature for {metric}") from exc
        if not math.isfinite(home_matchup) or not math.isfinite(away_matchup):
            raise DataContractError(f"non-finite recent-form matchup value for {metric}")
        values[f"recent_{metric}_margin_signal"] = home_matchup - away_matchup
        values[f"recent_{metric}_total_signal"] = home_matchup + away_matchup
    return values


def recent_feature_columns(
    metrics: tuple[str, ...],
    *,
    target: str,
) -> tuple[str, ...]:
    """Return target-specific recent-form feature names for a metric subset."""

    unknown = set(metrics) - set(RECENT_PBP_METRICS)
    if unknown:
        raise ValueError(f"unsupported recent-form metrics: {sorted(unknown)}")
    if target == "margin_residual":
        suffix = "margin_signal"
    elif target == "total_residual":
        suffix = "total_signal"
    else:
        raise ValueError(f"unsupported residual target: {target}")
    return tuple(f"recent_{metric}_{suffix}" for metric in metrics)
