"""Leakage-safe historical NFL schedule/venue context research.

This module deliberately excludes injuries, live weather, and sportsbook data. Every
feature is either present on the target game's schedule row before kickoff or derived
from deterministic team geography/time zones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from zoneinfo import ZoneInfo

import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .ratings import VALIDATED_PRIOR_SEASON_WEIGHT, fit_pregame_fair_score
from .weather import TEAM_HOME, haversine_miles

SCHEDULE_CONTEXT_REQUIRED = {
    "season",
    "week",
    "game_id",
    "game_type",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "away_rest",
    "home_rest",
    "location",
}

MARGIN_CONTEXT_FEATURES = (
    "ctx_rest_diff_days",
    "ctx_short_rest_edge",
    "ctx_away_travel_1000",
    "ctx_timezone_shift_hours",
    "ctx_neutral_site",
    "ctx_travel_available",
)

TOTAL_CONTEXT_FEATURES = (
    "ctx_combined_short_rest",
    "ctx_combined_travel_1000",
    "ctx_timezone_shift_hours",
    "ctx_neutral_site",
    "ctx_rest_sum_deviation",
    "ctx_travel_available",
)


def _finite_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"schedule context is missing numeric {name}") from exc
    if not isfinite(number):
        raise DataContractError(f"schedule context has non-finite {name}")
    return number


def _is_neutral(value: object) -> bool:
    return "neutral" in str(value or "").strip().lower()


def _timezone_shift_hours(
    away_team: str,
    home_team: str,
    gameday: object,
) -> tuple[float, bool]:
    away = TEAM_HOME.get(away_team)
    home = TEAM_HOME.get(home_team)
    if away is None or home is None:
        return 0.0, False
    try:
        stamp = datetime.fromisoformat(f"{str(gameday)[:10]}T12:00:00").replace(tzinfo=UTC)
        away_offset = stamp.astimezone(ZoneInfo(away[2])).utcoffset()
        home_offset = stamp.astimezone(ZoneInfo(home[2])).utcoffset()
    except (ValueError, KeyError):
        return 0.0, False
    if away_offset is None or home_offset is None:
        return 0.0, False
    shift = abs((home_offset - away_offset).total_seconds() / 3600.0)
    return float(shift), True


def _travel_values(
    away_team: str,
    home_team: str,
    *,
    neutral: bool,
) -> tuple[float, float, bool]:
    """Return away miles, home miles, availability.

    Neutral-site travel is intentionally not guessed from the nominal home team. It is
    represented as unavailable here and paired with an explicit neutral-site feature.
    """

    if neutral:
        return 0.0, 0.0, False
    away = TEAM_HOME.get(away_team)
    home = TEAM_HOME.get(home_team)
    if away is None or home is None:
        return 0.0, 0.0, False
    miles = haversine_miles(away[0], away[1], home[0], home[1])
    return float(miles), 0.0, True


def schedule_context_features(game: dict[str, object]) -> dict[str, float]:
    """Build deterministic pre-kickoff schedule/venue signals for one game."""

    away_team = str(game.get("away_team") or "")
    home_team = str(game.get("home_team") or "")
    if not away_team or not home_team:
        raise DataContractError("schedule context requires home_team and away_team")

    home_rest = _finite_number(game.get("home_rest"), "home_rest")
    away_rest = _finite_number(game.get("away_rest"), "away_rest")
    neutral = _is_neutral(game.get("location"))
    away_miles, home_miles, travel_available = _travel_values(
        away_team,
        home_team,
        neutral=neutral,
    )
    timezone_shift, timezone_available = _timezone_shift_hours(
        away_team,
        home_team,
        game.get("gameday"),
    )
    availability = travel_available and timezone_available

    home_short = float(home_rest <= 6.0)
    away_short = float(away_rest <= 6.0)
    return {
        "ctx_rest_diff_days": home_rest - away_rest,
        "ctx_short_rest_edge": away_short - home_short,
        "ctx_away_travel_1000": away_miles / 1000.0,
        "ctx_timezone_shift_hours": timezone_shift,
        "ctx_neutral_site": float(neutral),
        "ctx_travel_available": float(availability),
        "ctx_combined_short_rest": home_short + away_short,
        "ctx_combined_travel_1000": (away_miles + home_miles) / 1000.0,
        "ctx_rest_sum_deviation": (home_rest + away_rest - 14.0) / 7.0,
    }


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_schedule_context_week_snapshot(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    score_ridge: float = 8.0,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> pl.DataFrame:
    """Build one completed target week against the canonical pregame baseline."""

    require_columns(schedules, SCHEDULE_CONTEXT_REQUIRED, "schedules")
    if week < 2:
        raise ValueError("week must be >= 2")
    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()

    baseline = fit_pregame_fair_score(
        schedules,
        season,
        week,
        ridge=score_ridge,
        prior_season_weight=prior_season_weight,
    )
    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        projection = baseline.project(home_team, away_team)
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
        row.update(schedule_context_features(game))
        rows.append(row)
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_schedule_context_walkforward(
    schedules: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int | None = 18,
    score_ridge: float = 8.0,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> pl.DataFrame:
    """Reconstruct one season of pregame schedule-context residual rows."""

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
    for week in range(start_week, final_week + 1):
        snapshot = build_schedule_context_week_snapshot(
            schedules,
            season,
            week,
            score_ridge=score_ridge,
            prior_season_weight=prior_season_weight,
        )
        if not snapshot.is_empty():
            frames.append(snapshot)
    if not frames:
        raise DataContractError("schedule-context walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(["season", "week", "game_id"])
