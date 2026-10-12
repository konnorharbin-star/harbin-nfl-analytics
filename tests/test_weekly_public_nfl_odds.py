"""Synthetic multiweek NFL capture chronology and exact game safeguards."""
import json
from datetime import datetime,timezone
from types import SimpleNamespace
import pytest

from scripts.weekly_public_nfl_odds import upcoming, collect

NOW="2026-10-12T02:00:00Z"
OBSERVED="2026-10-12T02:01:00Z"

def game(gid,day,time,week=5,status="REG",home_score=None):
    return {
        "game_id":gid,"season":2026,"week":week,
        "game_type":status,"gameday":day,"gametime":time,
        "home_team":"LA","away_team":"BUF",
        "home_score":home_score,"away_score":None,
    }

def quote(game_id):
    return SimpleNamespace(
        game_id=game_id,market_type="total",provider="action_network",
        book="BetMGM",source_event_id="source_event",
        source_quote_time_verified=False,source_quote_at=None,
        first_side="over",first_line=51.5,first_american_odds=-110,
        second_side="under",second_line=51.5,second_american_odds=-108,
    )

def targets():
    return [
        {"game_id":"2026_05_BUF_LA","season":2026,"week":5,
         "home_team":"LA","away_team":"BUF",
         "kickoff_utc":"2026-10-13T00:15:00Z"},
        {"game_id":"2026_06_BUF_LA","season":2026,"week":6,
         "home_team":"LA","away_team":"BUF",
         "kickoff_utc":"2026-10-16T00:15:00Z"},
    ]

def stub(season,week,games):
    return [quote(g["game_id"]) for g in games],OBSERVED,{"status":"MOCK"}

def test_select_both_future_weeks_without_backfill():
    games=[
        game("already_started","2026-10-11","20:20"),
        game("2026_05_BUF_LA","2026-10-12","20:15",week=5),
        game("2026_06_BUF_LA","2026-10-15","20:15",week=6),
        game("later","2026-10-24","20:15",week=7),
        game("preseason","2026-10-15","20:15",status="PRE"),
        game("finished","2026-10-15","20:15",home_score=21),
    ]
    x=upcoming(games,now=NOW,hours=192)
    assert [g["game_id"] for g in x]==["2026_05_BUF_LA","2026_06_BUF_LA"]

def test_multi_week_capture_keeps_account_unverified(tmp_path):
    journal=tmp_path/"history.jsonl"
    report=collect(targets(),stub,observed=NOW,journal=journal)
    assert report["target_games"]==2
    assert report["games_with_capture"]==2
    assert report["captured_paired_rows"]==2
    assert report["origin_verified_paired_rows"]==0
    assert report["approved_bets"]==0
    records=[json.loads(v) for v in journal.read_text().splitlines()]
    assert {v["event_id"] for v in records}=={
        "2026_05_BUF_LA","2026_06_BUF_LA"}
    assert all(v["betting_decision"]=="PASS" for v in records)
    assert all(v["collector_observed_utc"]=="2026-10-12T02:01:00+00:00" for v in records)

def test_no_future_games_is_empty_and_nonbet():
    result=collect([],stub,observed=NOW)
    assert result["target_games"]==0
    assert result["decision"]=="PASS"
    assert result["captures"]==[]

def test_duplicate_games_fail_closed():
    g=game("x","2026-10-12","20:15")
    with pytest.raises(ValueError,match="duplicated"):
        upcoming([g,g],now=NOW)

def test_missing_kickoff_never_generated():
    x=upcoming([game("a","2026-10-12",None)],now=NOW)
    assert x==[]

def test_source_failure_records_error_not_fake_price():
    def broken(*args):
        raise ValueError("source offline")
    r=collect(targets(),broken,observed=NOW)
    assert r["captured_paired_rows"]==0
    assert r["games_with_capture"]==0
    assert len(r["rejections"])==2

def test_source_with_no_markets_not_saved(tmp_path):
    def empty(*args):
        return [],OBSERVED,{"status":"NONE"}
    p=tmp_path/"j.jsonl"
    x=collect(targets(),empty,observed=NOW,journal=p)
    assert not p.exists()
    assert x["games_with_capture"]==0

def test_started_game_cannot_enter():
    t=targets()
    t[0]["kickoff_utc"]="2026-10-11T23:00:00Z"
    with pytest.raises(ValueError,match="postkickoff"):
        collect(t,stub,observed=NOW)

def test_unconfirmed_source_time_not_imputed():
    def source(*args):
        data=stub(*args)
        return data[0],"2026-10-11T22:00:00Z",data[2]
    r=collect(targets(),source,observed=NOW)
    assert not r["captures"]
    assert r["rejections"]

def test_capture_not_an_account_price():
    result=collect(targets()[:1],stub,observed=NOW)
    assert not result["validated_edge"]
    assert result["captures"][0]["account_accepted_quotes"]==0

def test_snapshot_does_not_overwrite_same_observed_time(tmp_path):
    p=tmp_path/"journal.jsonl"
    collect(targets()[:1],stub,observed=NOW,journal=p)
    later=collect(targets()[:1],stub,observed=NOW,journal=p)
    assert later["captures"]==[]
    assert later["rejections"][0]["reason"]=="SOURCE_WEEK_FAILED"

def test_horizon_limit_rejects_overlong():
    with pytest.raises(ValueError,match="horizon"):
        upcoming([],now=NOW,hours=1000)
