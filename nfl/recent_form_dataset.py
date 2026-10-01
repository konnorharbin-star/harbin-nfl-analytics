"""Chronological NFL recent-form PBP research dataset.

This dataset is separate from the canonical score path. Each target week is rebuilt
from exact completed regular-season game IDs so recent-form state cannot see the target
week or future PBP.
"""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError
from .data import completed_games
from .dataset import _pbp_for_games, _regular_season_history, _target_games
from .ratings import fit_pregame_fair_score
from .recent_form import recent_matchup_signals, team_recent_pbp_features


def _team_feature_map(features: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {str(row["team"]): row for row in features.iter_rows(named=True)}


def build_recent_form_week_snapshot(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
    *,
    recent_alpha: float = 0.35,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Build one pregame week of baseline scores plus recent-form PBP signals."""

    history = _regular_season_history(schedules, season, week)
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if history.height < 2:
        raise DataContractError("at least two prior regular-season games are required")

    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = _pbp_for_games(pbp, history_ids)
    feature_map = _team_feature_map(
        team_recent_pbp_features(history_pbp, alpha=recent_alpha)
    )
    score_model = fit_pregame_fair_score(
        schedules,
        season,
        week,
        ridge=score_ridge,
    )

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        if home_team not in feature_map or away_team not in feature_map:
            raise DataContractError(
                f"missing recent-form PBP state for {away_team} at {home_team}"
            )

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
            "recent_alpha": float(recent_alpha),
            "baseline_home_points": projection.home_points,
            "baseline_away_points": projection.away_points,
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - projection.home_margin,
            "total_residual": actual_total - projection.total,
            "home_recent_off_games": int(feature_map[home_team]["recent_off_games"]),
            "away_recent_off_games": int(feature_map[away_team]["recent_off_games"]),
            "home_recent_def_games": int(feature_map[home_team]["recent_def_games"]),
            "away_recent_def_games": int(feature_map[away_team]["recent_def_games"]),
        }
        row.update(recent_matchup_signals(feature_map[home_team], feature_map[away_team]))
        rows.append(row)

    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_recent_form_walkforward_dataset(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    *,
    recent_alpha: float = 0.35,
    start_week: int = 5,
    end_week: int | None = None,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct a full season of recent-form research rows week by week."""

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
    season_pbp = pbp.filter(pl.col("season") == season)
    for week in range(start_week, final_week + 1):
        frame = build_recent_form_week_snapshot(
            schedules,
            season_pbp,
            season,
            week,
            recent_alpha=recent_alpha,
            score_ridge=score_ridge,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("recent-form walk-forward produced no target games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])
