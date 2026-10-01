"""Chronological NFL modeling-dataset construction.

Every row produced here represents a game as it would have looked before that week's
kickoff. Football features are built only from completed regular-season games in the
same season and strictly earlier weeks. Sportsbook data is intentionally absent.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .advanced import team_pbp_features
from .contracts import DataContractError, require_columns
from .data import completed_games
from .ratings import fit_pregame_fair_score

FEATURE_NAMES = (
    "epa_per_play",
    "success_rate",
    "pass_epa_per_dropback",
    "rush_epa_per_attempt",
    "explosive_rate",
    "early_down_epa",
)


def _regular_season_history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return same-season regular games completed strictly before ``week``."""

    require_columns(schedules, {"season", "week", "game_type", "game_id"}, "schedules")
    if week < 1:
        raise DataContractError("week must be >= 1")
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") < week)
    )


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return completed regular-season games in the target week."""

    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def _pbp_for_games(pbp: pl.DataFrame, game_ids: Iterable[str]) -> pl.DataFrame:
    require_columns(pbp, {"game_id"}, "pbp")
    ids = list(game_ids)
    if not ids:
        raise DataContractError("no historical game IDs are available for PBP features")
    return pbp.filter(pl.col("game_id").is_in(ids))


def _team_feature_map(features: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {row["team"]: row for row in features.iter_rows(named=True)}


def _matchup_feature_row(
    home: dict[str, object],
    away: dict[str, object],
) -> dict[str, float]:
    """Return symmetric home-minus-away matchup advantages.

    For an efficiency metric ``m`` the matchup value is::

        (home offense m - away defense m allowed)
        - (away offense m - home defense m allowed)

    Positive values therefore indicate a football-efficiency advantage for the home
    team before home-field value is considered by the fair-score baseline.
    """

    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        offense_key = f"off_{name}"
        defense_key = f"def_{name}_allowed"
        try:
            home_matchup = float(home[offense_key]) - float(away[defense_key])
            away_matchup = float(away[offense_key]) - float(home[defense_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataContractError(f"missing/non-numeric PBP feature for {name}") from exc
        values[f"{name}_matchup_advantage"] = home_matchup - away_matchup
    return values


def build_week_snapshot(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    ridge: float = 8.0,
) -> pl.DataFrame:
    """Build pregame features and canonical baseline projections for one NFL week."""

    history = _regular_season_history(schedules, season, week)
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if history.height < 2:
        raise DataContractError("at least two prior regular-season games are required")

    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = _pbp_for_games(pbp, history_ids)
    team_features = team_pbp_features(history_pbp)
    feature_map = _team_feature_map(team_features)

    model = fit_pregame_fair_score(schedules, season, week, ridge=ridge)
    rows: list[dict[str, object]] = []

    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        if home_team not in feature_map or away_team not in feature_map:
            raise DataContractError(
                f"missing pregame PBP feature state for {away_team} at {home_team}"
            )

        projection = model.project(home_team, away_team)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        actual_margin = home_score - away_score
        actual_total = home_score + away_score

        row: dict[str, object] = {
            "season": season,
            "week": week,
            "game_id": str(game["game_id"]),
            "gameday": game["gameday"],
            "away_team": away_team,
            "home_team": home_team,
            "baseline_home_points": projection.home_points,
            "baseline_away_points": projection.away_points,
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - projection.home_margin,
            "total_residual": actual_total - projection.total,
            "home_off_plays": int(feature_map[home_team]["off_plays"]),
            "away_off_plays": int(feature_map[away_team]["off_plays"]),
            "home_def_plays": int(feature_map[home_team]["def_plays"]),
            "away_def_plays": int(feature_map[away_team]["def_plays"]),
        }
        row.update(_matchup_feature_row(feature_map[home_team], feature_map[away_team]))
        rows.append(row)

    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_walkforward_dataset(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = None,
    ridge: float = 8.0,
) -> pl.DataFrame:
    """Create a season dataset by reconstructing each target week independently.

    Recomputing each week is intentional: it makes the information boundary explicit
    and prevents end-of-season aggregates from being reused as if they were pregame.
    """

    if start_week < 2:
        raise DataContractError("start_week must be >= 2 for a pregame training history")

    regular = completed_games(schedules).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if regular.is_empty():
        raise DataContractError(f"no completed regular-season games found for {season}")

    max_week = int(regular.get_column("week").max())
    final_week = max_week if end_week is None else min(end_week, max_week)
    if final_week < start_week:
        raise DataContractError("end_week is before start_week")

    snapshots: list[pl.DataFrame] = []
    for week in range(start_week, final_week + 1):
        snapshot = build_week_snapshot(schedules, pbp, season, week, ridge=ridge)
        if not snapshot.is_empty():
            snapshots.append(snapshot)

    if not snapshots:
        raise DataContractError("walk-forward construction produced no target games")
    return pl.concat(snapshots, how="vertical_relaxed").sort(["week", "game_id"])
