"""Chronological dataset for last-observed quarterback-state experiments."""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError
from .data import completed_games
from .qb_state import QB_PRIOR_DROPBACKS, qb_matchup_signals, team_qb_state
from .ratings import fit_pregame_fair_score


def _targets(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_qb_week_snapshot(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    season: int,
    week: int,
    *,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
) -> pl.DataFrame:
    """Build one completed target week using only earlier QB observations."""

    targets = _targets(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()

    state = team_qb_state(
        player_stats,
        season,
        week,
        prior_dropbacks=qb_prior_dropbacks,
    )
    state_map = {row["team"]: row for row in state.iter_rows(named=True)}
    score_model = fit_pregame_fair_score(schedules, season, week, ridge=score_ridge)

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        if home_team not in state_map or away_team not in state_map:
            raise DataContractError(
                f"missing last-observed QB state for {away_team} at {home_team}"
            )

        home_qb = state_map[home_team]
        away_qb = state_map[away_team]
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
            "home_qb_proxy_id": str(home_qb["qb_proxy_id"]),
            "away_qb_proxy_id": str(away_qb["qb_proxy_id"]),
            "home_qb_proxy_last_week": int(home_qb["qb_proxy_last_week"]),
            "away_qb_proxy_last_week": int(away_qb["qb_proxy_last_week"]),
            "baseline_home_points": projection.home_points,
            "baseline_away_points": projection.away_points,
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - projection.home_margin,
            "total_residual": actual_total - projection.total,
        }
        row.update(qb_matchup_signals(home_qb, away_qb))
        rows.append(row)

    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_qb_walkforward_dataset(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = None,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
) -> pl.DataFrame:
    """Reconstruct QB proxy signals independently for each target week."""

    if start_week < 3:
        raise ValueError("start_week must be >= 3 for QB continuity state")
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
        frame = build_qb_week_snapshot(
            schedules,
            player_stats,
            season,
            week,
            score_ridge=score_ridge,
            qb_prior_dropbacks=qb_prior_dropbacks,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError("QB walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["week", "game_id"])
