"""Leak-free historical non-QB personnel context for NFL residual research.

This module reconstructs pregame injury/depth-chart state only. It intentionally excludes
quarterback features because QB state has its own fixed-spec validation path. Sportsbook
prices are never read here.
"""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .injuries import normalize_injuries
from .personnel import normalize_depth_charts, summarize_team_personnel
from .ratings import fit_pregame_fair_score
from .weather import kickoff_utc

MARGIN_PERSONNEL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "starters": ("personnel_starter_diff",),
    "groups": (
        "personnel_ol_diff",
        "personnel_skill_diff",
        "personnel_defense_diff",
    ),
    "all": (
        "personnel_starter_diff",
        "personnel_ol_diff",
        "personnel_skill_diff",
        "personnel_defense_diff",
    ),
}

TOTAL_PERSONNEL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "starters": ("personnel_starter_sum",),
    "groups": (
        "personnel_ol_sum",
        "personnel_skill_sum",
        "personnel_defense_sum",
    ),
    "all": (
        "personnel_starter_sum",
        "personnel_ol_sum",
        "personnel_skill_sum",
        "personnel_defense_sum",
    ),
}

OL_POSITIONS = {
    "LT",
    "RT",
    "OT",
    "G",
    "LG",
    "RG",
    "OG",
    "C",
    "OL",
}
SKILL_POSITIONS = {"WR", "TE", "RB", "FB"}
DEFENSE_POSITIONS = {
    "CB",
    "S",
    "FS",
    "SS",
    "DB",
    "EDGE",
    "DE",
    "DT",
    "NT",
    "LB",
    "ILB",
    "OLB",
}


def _targets(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def _injury_lookup(injuries: pl.DataFrame) -> dict[tuple[str, str], float]:
    if injuries.is_empty():
        return {}
    require_columns(
        injuries,
        {"team", "player_key", "severity"},
        "historical_personnel_injuries",
    )
    return {
        (str(row["team"]), str(row["player_key"])): float(row["severity"])
        for row in injuries.iter_rows(named=True)
    }


def _non_qb_starter_risk(
    team: str,
    depth: pl.DataFrame,
    injuries: pl.DataFrame,
) -> float:
    if depth.is_empty():
        return 0.0
    team_depth = depth.filter(pl.col("team") == team)
    if team_depth.is_empty():
        return 0.0
    starters = team_depth.filter(
        (pl.col("position") != "QB")
        & (pl.col("depth_rank").is_null() | (pl.col("depth_rank") <= 1))
    )
    if starters.is_empty():
        return 0.0
    lookup = _injury_lookup(injuries)
    severity = sum(
        lookup.get((team, str(row["player_key"])), 0.0)
        for row in starters.iter_rows(named=True)
    )
    denominator = max(1.0, starters.height / 4.0)
    return min(1.0, float(severity) / denominator)


def _temporal_mode(team: str, depth: pl.DataFrame) -> str:
    if depth.is_empty():
        return "missing"
    rows = depth.filter(pl.col("team") == team)
    if rows.is_empty():
        return "missing"
    if (
        "depth_captured_at" in rows.columns
        and rows.get_column("depth_captured_at").drop_nulls().len() > 0
    ):
        return "timestamped"
    if "depth_week" in rows.columns:
        values = rows.get_column("depth_week").drop_nulls()
        if values.len() > 0:
            return "weekly"
    return "season_only"


def _latest_weekly_depth_snapshot(
    depth: pl.DataFrame,
) -> pl.DataFrame:
    """Keep one complete latest legacy weekly snapshot per team."""

    if depth.is_empty() or "depth_week" not in depth.columns:
        return depth
    weekly = depth.filter(pl.col("depth_week").is_not_null())
    timestamped = depth.filter(pl.col("depth_week").is_null())
    frames: list[pl.DataFrame] = []
    if not timestamped.is_empty():
        frames.append(timestamped)
    if not weekly.is_empty():
        for team_frame in weekly.partition_by(
            "team",
            maintain_order=True,
        ):
            latest_week = int(
                team_frame.get_column("depth_week").max()
            )
            frames.append(
                team_frame.filter(pl.col("depth_week") == latest_week)
            )
    if not frames:
        return depth
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["team", "position", "depth_rank", "player_name"]
    )


def _personnel_map(
    injuries: pl.DataFrame,
    depth: pl.DataFrame,
) -> dict[str, dict[str, object]]:
    summary = summarize_team_personnel(
        depth,
        pl.DataFrame(),
        injuries,
    )
    if summary.is_empty():
        return {}
    return {
        str(row["team"]): row
        for row in summary.iter_rows(named=True)
    }


def _signals(
    home: dict[str, object],
    away: dict[str, object],
    *,
    home_starter_risk: float,
    away_starter_risk: float,
) -> dict[str, float]:
    home_ol = float(home.get("ol_injury_risk", 0.0) or 0.0)
    away_ol = float(away.get("ol_injury_risk", 0.0) or 0.0)
    home_skill = float(home.get("skill_injury_risk", 0.0) or 0.0)
    away_skill = float(away.get("skill_injury_risk", 0.0) or 0.0)
    home_defense = float(home.get("defense_injury_risk", 0.0) or 0.0)
    away_defense = float(away.get("defense_injury_risk", 0.0) or 0.0)
    return {
        "personnel_starter_diff": home_starter_risk - away_starter_risk,
        "personnel_ol_diff": home_ol - away_ol,
        "personnel_skill_diff": home_skill - away_skill,
        "personnel_defense_diff": home_defense - away_defense,
        "personnel_starter_sum": home_starter_risk + away_starter_risk,
        "personnel_ol_sum": home_ol + away_ol,
        "personnel_skill_sum": home_skill + away_skill,
        "personnel_defense_sum": home_defense + away_defense,
    }


def build_personnel_week_snapshot(
    schedules: pl.DataFrame,
    injuries: pl.DataFrame,
    depth_charts: pl.DataFrame,
    season: int,
    week: int,
    *,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Build one completed week from only pre-kickoff non-QB personnel state."""

    targets = _targets(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    if injuries.is_empty() or depth_charts.is_empty():
        return pl.DataFrame()

    score_model = fit_pregame_fair_score(
        schedules,
        season,
        week,
        ridge=score_ridge,
    )
    state_cache: dict[
        str,
        tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, object]]],
    ] = {}
    rows: list[dict[str, object]] = []

    for game in targets.iter_rows(named=True):
        kickoff = kickoff_utc(game.get("gameday"), game.get("gametime"))
        if kickoff is None:
            continue
        cache_key = kickoff.isoformat()
        if cache_key not in state_cache:
            week_injuries = injuries.filter(
                (pl.col("season") == season)
                & (pl.col("week").cast(pl.Int64, strict=False) == week)
            )
            normalized_injuries = normalize_injuries(
                week_injuries,
                season=season,
                week=week,
                as_of=kickoff,
            )
            normalized_depth = _latest_weekly_depth_snapshot(
                normalize_depth_charts(
                    depth_charts,
                    season=season,
                    week=week,
                    as_of=kickoff,
                )
            )
            state_cache[cache_key] = (
                normalized_injuries,
                normalized_depth,
                _personnel_map(normalized_injuries, normalized_depth),
            )
        normalized_injuries, normalized_depth, personnel = state_cache[cache_key]

        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home = personnel.get(home_team)
        away = personnel.get(away_team)
        if home is None or away is None:
            continue
        if not bool(home.get("depth_source_available")):
            continue
        if not bool(away.get("depth_source_available")):
            continue

        home_starter = _non_qb_starter_risk(
            home_team,
            normalized_depth,
            normalized_injuries,
        )
        away_starter = _non_qb_starter_risk(
            away_team,
            normalized_depth,
            normalized_injuries,
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
            "kickoff": kickoff.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "home_depth_temporal_mode": _temporal_mode(
                home_team,
                normalized_depth,
            ),
            "away_depth_temporal_mode": _temporal_mode(
                away_team,
                normalized_depth,
            ),
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - projection.home_margin,
            "total_residual": actual_total - projection.total,
            "home_non_qb_starter_risk": home_starter,
            "away_non_qb_starter_risk": away_starter,
            "home_ol_injury_risk": float(
                home.get("ol_injury_risk", 0.0) or 0.0
            ),
            "away_ol_injury_risk": float(
                away.get("ol_injury_risk", 0.0) or 0.0
            ),
            "home_skill_injury_risk": float(
                home.get("skill_injury_risk", 0.0) or 0.0
            ),
            "away_skill_injury_risk": float(
                away.get("skill_injury_risk", 0.0) or 0.0
            ),
            "home_defense_injury_risk": float(
                home.get("defense_injury_risk", 0.0) or 0.0
            ),
            "away_defense_injury_risk": float(
                away.get("defense_injury_risk", 0.0) or 0.0
            ),
        }
        row.update(
            _signals(
                home,
                away,
                home_starter_risk=home_starter,
                away_starter_risk=away_starter,
            )
        )
        rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_personnel_walkforward_dataset(
    schedules: pl.DataFrame,
    injuries: pl.DataFrame,
    depth_charts: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = None,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct non-QB personnel features independently before each target game."""

    if start_week < 3:
        raise ValueError("start_week must be >= 3")
    regular = completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
    )
    if regular.is_empty():
        raise DataContractError(
            f"no completed regular-season games found for {season}"
        )
    max_week = int(regular.get_column("week").max())
    final_week = max_week if end_week is None else min(end_week, max_week)
    if final_week < start_week:
        raise ValueError("end_week is before start_week")

    frames: list[pl.DataFrame] = []
    for week in range(start_week, final_week + 1):
        frame = build_personnel_week_snapshot(
            schedules,
            injuries,
            depth_charts,
            season,
            week,
            score_ridge=score_ridge,
        )
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise DataContractError(
            f"personnel walk-forward produced no games for {season}"
        )
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
