"""Leakage-safe NFL player-stat prop challenger. No sportsbook prices or bet execution.

This is a *forecasting* experiment, not a validated sportsbook edge.
2024 development chooses parameters, 2025 measures diagnostic accuracy,
2026 current upcoming games are forecast from strictly previous weeks.
All games for which an eligible player fails to appear count as zero (including
DNP), avoiding postgame participant-selection leakage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import isfinite, sqrt
from typing import Any

import polars as pl

from .pregame import schedule_kickoff_utc

METRICS = {
    "receptions": frozenset(("WR", "TE", "RB", "FB")),
    "receiving_yards": frozenset(("WR", "TE", "RB", "FB")),
    "rushing_yards": frozenset(("RB", "QB", "WR", "FB")),
    "passing_yards": frozenset(("QB",)),
}
WINDOWS = (3, 5, 8)
SHRINKAGES = (0.0, 1.0, 3.0, 6.0)
PARAMETERS = tuple((w, k) for w in WINDOWS for k in SHRINKAGES)
MIN_PRIOR_PLAYER_GAMES = 3
MAX_LAST_ACTIVE_TEAM_GAMES = 2
MIN_POSITION_PRIOR_ROWS = 5
MIN_DIAGNOSTIC_ROWS = 70
BASELINE_WINDOW = 3


def _stat(raw: object) -> float | None:
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if isfinite(value) else None


def source_frames(
    schedules: pl.DataFrame, player_stats: pl.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate stat game/team identity against regular-season schedule.

    A final score validates historical target outcomes. We do not retroactively
    use later game results to construct prior-week features.
    """
    required_games = {
        "game_id", "season", "week", "game_type", "home_team", "away_team",
        "home_score", "away_score",
    }
    required_players = {
        "player_id", "player_name", "position", "season", "week",
        "season_type", "game_id", "team", *METRICS,
    }
    if not required_games.issubset(schedules.columns):
        raise ValueError("Schedule missing game/season/week/team/result provenance")
    if not required_players.issubset(player_stats.columns):
        raise ValueError("Player stats missing required metric or identity columns")

    games = []
    game_info = {}
    for game in schedules.iter_rows(named=True):
        if str(game.get("game_type")).upper() != "REG":
            continue
        try:
            season, week = int(game["season"]), int(game["week"])
        except (ValueError, TypeError, OverflowError):
            continue
        home = str(game.get("home_team") or "")
        away = str(game.get("away_team") or "")
        gid = str(game.get("game_id") or "")
        if not (gid and home and away and home != away and 1 <= week <= 18):
            continue
        completed = (
            _stat(game.get("home_score")) is not None
            and _stat(game.get("away_score")) is not None
            and _stat(game.get("home_score")) >= 0
            and _stat(game.get("away_score")) >= 0
        )
        item = {
            "game_id": gid, "season": season, "week": week,
            "home": home, "away": away,
            "completed": completed,
            "kickoff": schedule_kickoff_utc(game),
        }
        if gid in game_info:
            raise ValueError("Duplicate schedule game IDs")
        game_info[gid] = item
        games.append(item)

    rows = []
    seen = set()
    rejected = Counter()
    for row in player_stats.iter_rows(named=True):
        if str(row.get("season_type")).upper() != "REG":
            continue
        game = game_info.get(str(row.get("game_id") or ""))
        if game is None or not game["completed"]:
            rejected["not_a_completed_schedule_game"] += 1
            continue
        try:
            observed_season = int(row.get("season"))
            observed_week = int(row.get("week"))
        except (ValueError, TypeError, OverflowError):
            rejected["invalid_stat_season_week"] += 1
            continue
        if (observed_season, observed_week) != (game["season"], game["week"]):
            rejected["source_week_does_not_match_schedule"] += 1
            continue
        pid = str(row.get("player_id") or "")
        team = str(row.get("team") or "")
        position = str(row.get("position") or "").upper()
        if (not pid or team not in (game["home"], game["away"])
            or not any(position in valid for valid in METRICS.values())):
            rejected["invalid_player_team_or_position"] += 1
            continue
        key = (game["game_id"], pid)
        if key in seen:
            raise ValueError("Ambiguous duplicate player stats for one game")
        seen.add(key)
        values = {metric: _stat(row.get(metric)) for metric in METRICS}
        if not any(value is not None for value in values.values()):
            rejected["no_usable_stat_values"] += 1
            continue
        rows.append({
            "game_id": game["game_id"], "season": game["season"],
            "week": game["week"], "player_id": pid,
            "name": str(row.get("player_name") or pid),
            "team": team, "position": position, **values,
        })
    games.sort(key=lambda g:(g["season"],g["week"],g["game_id"]))
    rows.sort(key=lambda r:(r["season"],r["week"],r["game_id"],r["player_id"]))
    return games, rows, {"player_rows": len(rows), "excluded_stat_rows": dict(rejected)}


def _previous_games(games: list[dict[str, Any]], season: int, week: int
                    ) -> dict[str, list[str]]:
    by_team: dict[str, list[str]] = defaultdict(list)
    for game in games:
        if game["completed"] and game["season"] == season and game["week"] < week:
            for team in (game["home"], game["away"]):
                by_team[team].append(game["game_id"])
    return by_team


def forecast_week(
    games: list[dict[str, Any]], stats: list[dict[str, Any]],
    *, season: int, week: int, options: dict[str, tuple[int, float]] | None = None,
    future_only_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Forecast all known recent players BEFORE inspecting target week rows."""
    if future_only_at is not None and future_only_at.tzinfo is None:
        raise ValueError("as_of requires an aware timestamp")
    targets = [g for g in games if g["season"] == season and g["week"] == week]
    if future_only_at is not None:
        ts = future_only_at.astimezone(UTC)
        targets = [
            g for g in targets if g["kickoff"] is not None and g["kickoff"] > ts
        ]
    if not targets:
        return []
    options = options or {key:(5, 1.0) for key in METRICS}
    prev_games = _previous_games(games, season, week)
    history = [
        r for r in stats
        if r["season"] == season and r["week"] < week
    ]
    by_player: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_team_position: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in history:
        by_player[(r["team"],r["player_id"])].append(r)
        by_team_position[(r["team"],r["position"])].append(r)

    forecasts = []
    for game in targets:
        for team in (game["home"],game["away"]):
            team_games = prev_games.get(team, [])
            if len(team_games) < MIN_PRIOR_PLAYER_GAMES:
                continue
            recent_ids = set(team_games[-MAX_LAST_ACTIVE_TEAM_GAMES:])
            for (p_team, pid), played in by_player.items():
                if p_team != team or len(played) < MIN_PRIOR_PLAYER_GAMES:
                    continue
                # Entry selection is based on previous games, NOT target-game
                # participation; inactive players will score as zero.
                if not any(r["game_id"] in recent_ids for r in played):
                    continue
                last = played[-1]
                for metric, allowed in METRICS.items():
                    if last["position"] not in allowed:
                        continue
                    played_metric = [r[metric] for r in played if r[metric] is not None]
                    if len(played_metric) < MIN_PRIOR_PLAYER_GAMES:
                        continue
                    baseline = sum(played_metric[-BASELINE_WINDOW:]) / BASELINE_WINDOW
                    if baseline <= 0:
                        continue
                    window, shrink = options[metric]
                    recent = played_metric[-window:]
                    # Teammate/position prior uses only historical observations.
                    team_prior_rows = [
                        r[metric] for r in by_team_position[(team,last["position"])]
                        if r[metric] is not None
                        and r["game_id"] in set(team_games[-8:])
                    ]
                    prior = (
                        sum(team_prior_rows) / len(team_prior_rows)
                        if len(team_prior_rows) >= MIN_POSITION_PRIOR_ROWS
                        else sum(played_metric) / len(played_metric)
                    )
                    predicted = (
                        (sum(recent) + shrink * prior)/(len(recent)+shrink)
                    )
                    forecasts.append({
                        "game_id":game["game_id"],"season":season,"week":week,
                        "home_team":game["home"],"away_team":game["away"],
                        "kickoff_utc":(game["kickoff"].isoformat()
                                       if game["kickoff"] is not None else None),
                        "team":team,"player_id":pid,"player_name":last["name"],
                        "position":last["position"],"market":metric,
                        "prior_season_games":len(played_metric),
                        "most_recent_seen_week":last["week"],
                        "baseline_three_game_mean":round(baseline,6),
                        "challenger_mean":round(predicted,6),
                        "window":window,"shrinkage":shrink,
                        "line":None,"source_quote_at":None,"book":None,
                        "player_active_at_kickoff_verified":False,
                        "current_expected_starter_verified":False,
                        "calibrated_prop_probability":None,
                        "true_market_ev":None,
                        "recommendation":"UNPRICED_RESEARCH_ONLY",
                        "automatic_betting_enabled":False,
                    })
    return sorted(
        forecasts, key=lambda r:(r["game_id"],r["team"],r["player_id"],r["market"])
    )


def diagnostic(
    games: list[dict[str, Any]], stats: list[dict[str, Any]],
    *, years: tuple[int, ...]=(2024,2025),
) -> dict[str, Any]:
    """Tune on first year only; second historical year is diagnostics only.

    Actual target outcomes, including absence, are joined strictly AFTER
    forecasts have been generated from earlier weeks.
    """
    outcomes = {(r["game_id"],r["player_id"]):r for r in stats}
    cache: dict[tuple[int,int,int,float],list[dict[str,Any]]] = {}
    for season in years:
        weeks = sorted({g["week"] for g in games
                        if g["season"] == season and g["completed"] and g["week"] >= 4})
        for window, shrink in PARAMETERS:
            option={m:(window,shrink) for m in METRICS}
            for week in weeks:
                current=forecast_week(games,stats,season=season,week=week,
                                      options=option)
                # Exclude games without final scores (independent of player stats).
                confirmed={g["game_id"] for g in games
                           if g["season"]==season and g["week"]==week
                           and g["completed"]}
                cache[(season,week,window,shrink)] = [
                    r for r in current if r["game_id"] in confirmed
                ]
    chosen={}
    metrics={}
    for market in METRICS:
        dev=[]
        for w,k in PARAMETERS:
            squared=0.0
            abs_err=0.0
            n=0
            for (s,week,pw,pk),predictions in cache.items():
                if (s,pw,pk)!=(years[0],w,k):
                    continue
                for entry in predictions:
                    if entry["market"]!=market:
                        continue
                    actual=outcomes.get((entry["game_id"],entry["player_id"]))
                    target=actual.get(market) if actual else 0.0
                    if target is None:
                        continue
                    diff=entry["challenger_mean"]-target
                    squared+=diff*diff;abs_err+=abs(diff);n+=1
            if n:
                dev.append((abs_err/n,squared/n,w,k,n))
        if not dev:
            chosen[market]=(5,1.0)
            metrics[market]={"status":"NO_DEVELOPMENT_SAMPLE"}
            continue
        mae,mse,window,shrink,n=min(dev)
        chosen[market]=(window,shrink)
        samples=[]
        for (s,week,pw,pk),predictions in cache.items():
            if (s,pw,pk)!=(years[1],window,shrink):
                continue
            for entry in predictions:
                if entry["market"]!=market:
                    continue
                actual=outcomes.get((entry["game_id"],entry["player_id"]))
                target=actual.get(market) if actual else 0.0
                if target is not None:
                    samples.append((entry["challenger_mean"],
                                    entry["baseline_three_game_mean"], target))
        if len(samples)<MIN_DIAGNOSTIC_ROWS:
            metrics[market]={
                "status":"INSUFFICIENT_DIAGNOSTIC_SAMPLE",
                "development_rows":n,"diagnostic_rows":len(samples),
                "chosen_window":window,"chosen_shrinkage":shrink,
            }
            continue
        def score(idx: int) -> tuple[float,float]:
            return (sum(abs(row[idx]-row[2]) for row in samples)/len(samples),
                    sqrt(sum((row[idx]-row[2])**2 for row in samples)/len(samples)))
        chal_mae,chal_rmse=score(0)
        base_mae,base_rmse=score(1)
        metrics[market]={
            "status":"HISTORICAL_DIAGNOSTIC_ONLY",
            "development_rows":n,
            "development_challenger_mae":round(mae,5),
            "diagnostic_rows":len(samples),
            "chosen_window":window,"chosen_shrinkage":shrink,
            "baseline_three_game_mae":round(base_mae,5),
            "challenger_mae":round(chal_mae,5),
            "baseline_three_game_rmse":round(base_rmse,5),
            "challenger_rmse":round(chal_rmse,5),
            "beat_simple_baseline_on_mae":chal_mae<base_mae,
            "beat_market_no_vig_probability":False,
            "valid_sportsbook_edge_proven":False,
        }
    return {
        "development_season":years[0],"diagnostic_season":years[1],
        "years_are_pristine_unseen_holdout":False,
        "development_choices":chosen,"by_market":metrics,
        "no_market_price_data_used":True,"no_prop_probability_claims":True,
        "auto_wagering_enabled":False,
    }
