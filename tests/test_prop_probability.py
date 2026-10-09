"""Synthetic-threshold player probabilities are prior-week research, never bets."""
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl.prop_challenger import forecast_week, source_frames
from nfl.prop_probability import (
    PROP_THRESHOLD_GRID,
    append_first_seen_curves,
    build_probability_experiment,
    evaluate_diagnostic,
    over_probability,
    probability_rows,
)


def inputs():
    games=[]
    stats=[]
    for season in (2024,2025,2026):
        for week in range(1,14 if season < 2026 else 6):
            completed=season<2026 or week<5
            gid=f"{season}_{week}_AAA_BBB"
            games.append({
                "season":season,"week":week,"game_type":"REG","game_id":gid,
                "home_team":"AAA","away_team":"BBB",
                "home_score":24 if completed else None,
                "away_score":17 if completed else None,
                "gameday":f"{season}-10-{week+1:02d}","gametime":"13:00",
            })
            if not completed:
                continue
            for team in ("AAA","BBB"):
                for i in range(23):
                    position="WR" if i < 9 else ("RB" if i < 13 else "QB")
                    if season==2025 and week==8 and i==0:
                        # Target-game absent player; pregame eligibility stays.
                        continue
                    stats.append({
                        "season":season,"week":week,"season_type":"REG",
                        "game_id":gid,"team":team,
                        "player_id":f"{team}_{i}","player_name":f"{team} Player {i}",
                        "position":position,
                        "receptions":(i+week)%9 if position=="WR" else 1,
                        "receiving_yards":(i*5 + week*3)%110
                            if position=="WR" else 5,
                        "rushing_yards":(i*3+week*9)%130
                            if position=="RB" else 1,
                        "passing_yards":(175+week*13+i*5)
                            if position=="QB" else 0,
                    })
    return source_frames(pl.DataFrame(games),pl.DataFrame(stats))[:2]


def experiment():
    games,stats=inputs()
    choices={market:(8,0.0) for market in PROP_THRESHOLD_GRID}
    current=forecast_week(
        games,stats,season=2026,week=5,options=choices,
        future_only_at=datetime(2026,10,4,12,tzinfo=UTC),
    )
    report,curves=build_probability_experiment(
        games,stats,options=choices,season=2026,week=5,
        current_forecasts=current,
    )
    return games,stats,choices,current,report,curves


def test_empirical_probabilities_monotone_and_complementary():
    _,_,_,_,report,curves=experiment()
    assert report["development_season"]==2024
    assert report["diagnostic_season"]==2025
    assert report["2026_forward_probability_rows"]==len(curves)>100
    assert report["calibrated_sportsbook_edge_proven"] is False
    assert report["automatic_betting_enabled"] is False
    grouped={}
    for entry in curves:
        assert 0.01 <= entry["probability_over"] <= .99
        assert abs(entry["probability_over"]+entry["probability_under"]-1)<1e-6
        assert entry["book_price"] is None
        assert entry["is_bet_recommendation"] is False
        key=(entry["game_id"],entry["player_id"],entry["market"])
        grouped.setdefault(key,[]).append(entry)
    assert grouped
    for entries in grouped.values():
        assert [x["synthetic_line"] for x in entries] == list(
            PROP_THRESHOLD_GRID[entries[0]["market"]]
        )
        assert all(
            entries[i]["probability_over"]>=entries[i+1]["probability_over"]
            for i in range(len(entries)-1)
        )


def test_one_year_development_and_separate_diagnostic_baseline():
    _,_,_,_,report,_=experiment()
    for _market,info in report["2025_synthetic_threshold_diagnostic"].items():
        assert info["status"]=="HISTORICAL_SYNTHETIC_LINES_DIAGNOSTIC_ONLY"
        assert info["evaluated_synthetic_thresholds"]>120
        assert info["challenger"]["n"]==info["rolling_three_baseline"]["n"]
        assert info["challenger"]["brier"]>=0
        assert info["rolling_three_baseline"]["brier"]>=0
        assert info["profitable_market_edge_proven"] is False
        assert info["tested_vs_actual_bookmaker_odds"] is False
        assert info["distinct_weeks"]>=8


def test_2025_results_never_fit_2024_residuals_or_alter_future_curve():
    games,stats,choices,forecast,report,curves=experiment()
    mutated=[dict(x) for x in stats]
    for row in mutated:
        if row["season"]==2025:
            row["receiving_yards"] = 2000
            row["receptions"] = 100
    changed_report,changed_curves=build_probability_experiment(
        games,mutated,options=choices,season=2026,week=5,
        current_forecasts=forecast,
    )
    assert curves==changed_curves
    assert changed_report["2025_synthetic_threshold_diagnostic"]!=(
        report["2025_synthetic_threshold_diagnostic"]
    )


def test_late_target_game_stats_do_not_rewrite_frozen_mean_or_thresholds():
    games,stats,choices,forecast,_,curves=experiment()
    current=[dict(r) for r in stats]
    current.append({
        **stats[0],"season":2026,"week":5,"game_id":"2026_5_AAA_BBB",
        "receptions":99,"receiving_yards":9000,
    })
    late=forecast_week(
        games,current,season=2026,week=5,options=choices,
        future_only_at=datetime(2026,10,4,12,tzinfo=UTC),
    )
    assert late==forecast
    _,late_curves=build_probability_experiment(
        games,current,options=choices,season=2026,week=5,
        current_forecasts=late,
    )
    assert late_curves==curves


def test_invalid_half_point_lines_and_missing_errors_fail_closed():
    for line in (-1.5, float("nan"), 3.0):
        with pytest.raises(ValueError):
            over_probability(5.0,line,"receptions",[1.0,-1.0])
    with pytest.raises(ValueError):
        over_probability(5.0,3.5,"receptions",[])
    assert over_probability(5.0,1.5,"receptions",[0.0]*300)<=.99


def test_low_data_market_has_no_spurious_confident_distribution():
    prediction={
        "game_id":"g","season":2026,"week":5,
        "kickoff_utc":"2026-10-12T20:00:00+00:00",
        "team":"AAA","player_id":"p","player_name":"P",
        "position":"WR","market":"receptions",
        "challenger_mean":5.0,"baseline_three_game_mean":5.0,
    }
    assert probability_rows(
        [prediction],residuals={("challenger","receptions","WR"):[0.1]*10}
    )==[]
    assert evaluate_diagnostic([],{})["receptions"]["status"]==(
        "INSUFFICIENT_2025_SYNTHETIC_THRESHOLDS"
    )


def test_probability_first_seen_never_backfills_after_game(tmp_path: Path):
    _,_,_,_,_,curves=experiment()
    seen_at=datetime(2026,10,4,12,tzinfo=UTC)
    path=tmp_path/"curves.csv"
    a=append_first_seen_curves(curves,observed_at=seen_at,destination=path)
    assert a["new_entries"]==len(curves)
    before=path.read_text()
    b=append_first_seen_curves(curves,observed_at=seen_at,destination=path)
    assert b["new_entries"]==0
    assert path.read_text()==before
    revised=[{**row,"probability_over":.99} for row in curves]
    append_first_seen_curves(revised,observed_at=seen_at,destination=path)
    assert path.read_text()==before
    after=datetime(2026,11,1,tzinfo=UTC)
    c=append_first_seen_curves(curves,observed_at=after,
                               destination=tmp_path/"late.csv")
    assert c["new_entries"]==0
    with pytest.raises(ValueError):
        append_first_seen_curves(
            curves,observed_at=datetime(2026,10,4,12),
            destination=tmp_path/"naive.csv",
        )


def test_source_missing_player_in_target_game_still_graded_zero():
    games,stats=inputs()
    choices={m:(8,0.0) for m in PROP_THRESHOLD_GRID}
    before=[x for x in forecast_week(
        games,stats,season=2025,week=8,options=choices
    ) if x["player_id"]=="AAA_0" and x["market"]=="receptions"]
    assert before
    # Player is not in target-week stats. Missingness cannot decide eligibility.
    assert not [r for r in stats if r["season"]==2025 and r["week"]==8
                and r["player_id"]=="AAA_0"]
