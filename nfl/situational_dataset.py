"""Chronological dataset for NFL situational PBP residual research."""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .ratings import fit_pregame_fair_score
from .situational import situational_matchup_signals, team_situational_features


def _regular_history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    require_columns(schedules, {"season", "week", "game_type", "game_id"}, "schedules")
    if week < 2:
        raise ValueError("week must be >= 2")
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


def build_situational_week_snapshot(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Build one target week using only completed same-season pregame game IDs."""

    history = _regular_history(schedules, season, week)
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if history.height < 2:
        raise DataContractError("at least two prior regular-season games are required")

    require_columns(pbp, {"game_id"}, "pbp")
    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = pbp.filter(pl.col("game_id").cast(pl.String).is_in(history_ids))
    if history_pbp.is_empty():
        raise DataContractError("no situational PBP rows matched eligible historical game IDs")

    features = team_situational_features(history_pbp)
    score_model = fit_pregame_fair_score(schedules, season, week, ridge=score_ridge)
    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        projection = score_model.project(home_team, away_team)
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
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - projection.home_margin,
            "total_residual": actual_total - projection.total,
        }
        row.update(situational_matchup_signals(features, home_team, away_team))
        rows.append(row)
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_situational_walkforward_dataset(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = 18,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct one season of point-in-time situational matchup signals."""

    if start_week < 3:
        raise ValueError("start_week must be >= 3")
    regular = completed_games(schedules).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if regular.is_empty():
        raise DataContractError(f"no completed regular-season games found for {season}")
    max_week = int(regular.get_column("week").max())
    final_week = max_week if end_week is None else min(int(end_week), max_week)
    if final_week < start_week:
        raise ValueError("end_week is before start_week")

    season_pbp = pbp.filter(pl.col("season") == season)
    frames: list[pl.DataFrame] = []
    for week in range(start_week, final_week + 1):
        snapshot = build_situational_week_snapshot(
            schedules,
            season_pbp,
            season,
            week,
            score_ridge=score_ridge,
        )
        if not snapshot.is_empty():
            frames.append(snapshot)
    if not frames:
        raise DataContractError("situational walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
