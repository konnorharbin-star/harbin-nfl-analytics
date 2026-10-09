"""All player-prop shadow forecasts must be generated strictly before target week."""
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl.prop_challenger import (
    METRICS,
    diagnostic,
    forecast_week,
    source_frames,
)
from run_player_prop_shadow import append_first_seen


def fixtures():
    games, stats = [], []
    for season in (2024, 2025, 2026):
        for week in range(1, 8 if season < 2026 else 6):
            finished = season < 2026 or week < 5
            gid = f"{season}_{week}_AAA_BBB"
            games.append({
                "game_id":gid, "season":season, "week":week,
                "game_type":"REG",
                "home_team":"AAA","away_team":"BBB",
                "home_score":20 if finished else None,
                "away_score":16 if finished else None,
                "gameday":f"{season}-10-{1+week:02d}",
                "gametime":"13:00",
            })
            if not finished:
                continue
            for team in ("AAA", "BBB"):
                for i,pos in enumerate(("WR","RB","QB")):
                    # A player who disappears from target week must still
                    # be eligible for the pregame prediction.
                    if season == 2025 and week == 5 and i == 0:
                        continue
                    stats.append({
                        "player_id":f"{team}_{i}", "player_name":f"{team} P{i}",
                        "position":pos,"season":season,"week":week,
                        "season_type":"REG","game_id":gid,"team":team,
                        "receptions":(4+week%3) if pos=="WR" else 1,
                        "receiving_yards":(50+week*2) if pos=="WR" else 4,
                        "rushing_yards":(65+week) if pos=="RB" else 4,
                        "passing_yards":(210+week*5) if pos=="QB" else 0,
                    })
    return pl.DataFrame(games),pl.DataFrame(stats)


def test_player_stat_source_only_accepts_completed_real_schedule_games():
    schedule,stats=fixtures()
    schedule_rows,stat_rows,diagnostics=source_frames(schedule,stats)
    assert len(schedule_rows) == len(schedule)
    assert len(stat_rows)==len(stats)
    assert diagnostics["player_rows"] == len(stats)
    assert any(not x["completed"] for x in schedule_rows)


def test_forecast_uses_only_previous_weeks_not_target_stats():
    schedule,players=fixtures()
    games,stats,_=source_frames(schedule,players)
    first=forecast_week(games,stats,season=2025,week=5)
    assert first
    for item in first:
        assert item["most_recent_seen_week"] < 5
        assert item["recommendation"] == "UNPRICED_RESEARCH_ONLY"
        assert item["true_market_ev"] is None
        assert item["calibrated_prop_probability"] is None
        assert item["automatic_betting_enabled"] is False
    # Mutating current/after target outcomes cannot change the prediction.
    second=[dict(row) for row in stats]
    for item in second:
        if item["week"] >=5 and item["season"]==2025:
            item["receiving_yards"]=100000
            item["receptions"]=10000
    assert forecast_week(games,stats,season=2025,week=5) == forecast_week(
        games,second,season=2025,week=5
    )


def test_player_missing_target_game_does_not_disappear_from_forecast():
    s,p=fixtures()
    games,stats,_=source_frames(s,p)
    names=forecast_week(games,stats,season=2025,week=5)
    assert any(r["player_id"]=="AAA_0" and r["market"]=="receptions"
               for r in names)
    assert all(r["week"]==5 for r in names)
    # Their postgame absence is not used when selecting whom to forecast.
    assert not any(r["season"]==2025 and r["week"]==5
                   and r["player_id"]=="AAA_0" for r in stats)


def test_upcoming_season_is_research_only_and_time_filtered():
    s,p=fixtures()
    games,stats,_=source_frames(s,p)
    at=datetime(2026,10,3,15,tzinfo=UTC)
    forecasts=forecast_week(
        games,stats,season=2026,week=5,future_only_at=at
    )
    assert forecasts
    assert {r["game_id"] for r in forecasts}=={"2026_5_AAA_BBB"}
    assert {r["market"] for r in forecasts}.issubset(METRICS)
    assert not forecast_week(
        games,stats,season=2026,week=5,
        future_only_at=datetime(2026,11,1,tzinfo=UTC),
    )


def test_not_enough_history_or_wrong_team_fails_closed():
    s,p=fixtures()
    games,stats,_=source_frames(s,p)
    assert forecast_week(games,stats,season=2025,week=3)==[]
    assert all(r["team"] in ("AAA","BBB")
               for r in forecast_week(games,stats,season=2025,week=5))


def test_historical_diagnostic_tunes_development_year_not_2025():
    s,p=fixtures()
    games,stats,_=source_frames(s,p)
    first=diagnostic(games,stats)
    assert first["development_season"]==2024
    assert first["diagnostic_season"]==2025
    assert not first["years_are_pristine_unseen_holdout"]
    second=[dict(r) for r in stats]
    for r in second:
        if r["season"]==2025:
            r["receptions"]=(r["receptions"] or 0)+4000
    changed=diagnostic(games,second)
    assert changed["development_choices"]==first["development_choices"]
    assert not first["auto_wagering_enabled"]
    assert all(len(v)==2 for v in first["development_choices"].values())


def test_first_seen_forward_ledger_is_immutable_and_no_backfill(tmp_path: Path):
    s,p=fixtures()
    games,stats,_=source_frames(s,p)
    asof=datetime(2026,10,3,15,tzinfo=UTC)
    candidates=forecast_week(
        games,stats,season=2026,week=5,future_only_at=asof
    )
    ledger=tmp_path/"forward.csv"
    a=append_first_seen(candidates,timestamp=asof,ledger=ledger)
    assert a["new_entries"] == len(candidates)
    first=ledger.read_text()
    b=append_first_seen(candidates,timestamp=asof,ledger=ledger)
    assert b["new_entries"]==0
    assert ledger.read_text()==first
    # Even a changed forecast must not overwrite first-seen pregame entry.
    mutation=[{**r,"challenger_mean":999999} for r in candidates]
    append_first_seen(mutation,timestamp=asof,ledger=ledger)
    assert ledger.read_text()==first
    future=append_first_seen(candidates,timestamp=datetime(2026,11,1,tzinfo=UTC),
                             ledger=tmp_path/"too_late.csv")
    assert future["new_entries"]==0


def test_rejects_stat_identity_mismatch_duplicate_players_and_bad_dates():
    s,p=fixtures()
    bad=p.with_columns(pl.lit("WRONG").alias("team"))
    g,stats,d=source_frames(s,bad)
    assert not stats
    assert d["excluded_stat_rows"]["invalid_player_team_or_position"]>0
    try:
        source_frames(s,pl.concat([p,p.head(1)],how="vertical"))
    except ValueError as e:
        assert "Duplicate player" in str(e)
    else:
        raise AssertionError("duplicated game stats accepted")
