"""Participation-conditioned v2: exact cohort, leakage and immutable-forward tests."""
from datetime import datetime, timezone, timedelta
import csv
import pytest

from scripts import participation_conditional_prop_v2 as v2

def pred(game="2024_04_X_Y",player="P1",market="receptions"):
    return {
        "game_id":game,"season":2024,"week":4,"team":"X",
        "player_id":player,"position":"WR","market":market,
        "challenger_mean":4.0,"baseline_three_game_mean":3.0,
    }

def game(gid="2024_04_X_Y",week=4):
    return {"game_id":gid,"season":2024,"week":week,
            "completed":True,"home":"X","away":"Y"}

def stat(game_id="2024_04_X_Y",pid="P1",**values):
    return {"game_id":game_id,"season":2024,"week":4,
            "player_id":pid,"team":"X","receptions":values.get("receptions",2)}

def test_training_missing_player_is_unknown_not_observed_zero(monkeypatch):
    players=[pred(player="P1"),pred(player="P2")]
    monkeypatch.setattr(v2,"forecast_week",lambda *a,**k:players)
    result=v2.samples([game()], [stat()],2024,v2.DEFAULT_OPTIONS)
    assert len(result["participation_proxy"])==1
    assert result["participation_proxy"][0][1]==2.0
    assert len(result["legacy_zero_imputation"])==2
    assert result["legacy_zero_imputation"][1][1]==0.0
    assert len(result["unresolved_without_matching_stat"])==1


def test_explicit_zero_stat_is_observed_not_dnp(monkeypatch):
    monkeypatch.setattr(v2,"forecast_week",lambda *a,**k:[pred()])
    out=v2.samples([game()],[stat(receptions=0)],2024,v2.DEFAULT_OPTIONS)
    assert out["participation_proxy"][0][1]==0.0
    assert not out["unresolved_without_matching_stat"]


def test_invalid_missing_stat_not_imputed_zero(monkeypatch):
    monkeypatch.setattr(v2,"forecast_week",lambda *a,**k:[pred()])
    out=v2.samples([game()],[stat(receptions=None)],2024,v2.DEFAULT_OPTIONS)
    assert out["participation_proxy"]==[]
    assert len(out["unresolved_without_matching_stat"])==1
    assert out["legacy_zero_imputation"]==[]


def test_player_team_mismatch_is_a_hard_error(monkeypatch):
    monkeypatch.setattr(v2,"forecast_week",lambda *a,**k:[pred()])
    row=stat();row["team"]="Y"
    with pytest.raises(ValueError,match="player/team"):
        v2.samples([game()],[row],2024,v2.DEFAULT_OPTIONS)


def test_only_completed_schedule_games_scored(monkeypatch):
    monkeypatch.setattr(v2,"forecast_week",lambda *a,**k:[pred()])
    target=game();target["completed"]=False
    out=v2.samples([target],[stat()],2024,v2.DEFAULT_OPTIONS)
    assert out["participation_proxy"]==[]


def fixture_row(time="2026-10-16T00:15:00Z",line=4.5):
    return {
        "game_id":"2026_06_X_Y","season":2026,"week":6,
        "kickoff_utc":time,
        "team":"X","player_id":"P1","player_name":"Synthetic Player",
        "position":"WR","market":"receptions",
        "synthetic_line":line,"p_over":.6,"p_under":.4,
        "calibration_basis":"POSITION_2024_EMPIRICAL",
        "train_residual_count":130,
        "player_participation_verified":False,"bookmaker":None,
        "american_odds":None,"model_ev":None,
        "recommended_bet":False,
    }


def test_freeze_first_seen_before_future_game_never_rewrites(tmp_path):
    p=tmp_path/"first_seen.csv"
    observed="2026-10-12T03:20:00Z"
    count=v2.freeze_first_seen([fixture_row()],observed_utc=observed,path=p)
    assert count=={"existing":0,"new":1,"total":1}
    with p.open(newline="") as source:
        lines=list(csv.DictReader(source))
    assert lines[0]["version"]==v2.VERSION
    assert lines[0]["first_observed_at_utc"]=="2026-10-12T03:20:00+00:00"
    assert lines[0]["recommended_bet"]=="False"
    assert lines[0]["bookmaker"]==""
    second=v2.freeze_first_seen([fixture_row()],observed_utc="2026-10-13T00:00:00Z",path=p)
    assert second=={"existing":1,"new":0,"total":1}
    assert len(p.read_text().splitlines())==2


def test_never_backfills_after_kickoff(tmp_path):
    p=tmp_path/"history.csv"
    row=fixture_row(time="2026-10-11T00:00:00Z")
    out=v2.freeze_first_seen([row],observed_utc="2026-10-12T03:20:00Z",path=p)
    assert out["new"]==0 and not p.exists()


def test_duplicate_new_threshold_in_same_capture_fails(tmp_path):
    with pytest.raises(ValueError,match="duplicate"):
        v2.freeze_first_seen([fixture_row(),fixture_row()],
                              observed_utc="2026-10-12T03:20:00Z",
                              path=tmp_path/"x.csv")


def test_timezone_naive_collection_rejected(tmp_path):
    with pytest.raises(ValueError,match="UTC-aware"):
        v2.freeze_first_seen([fixture_row()],observed_utc="2026-10-12T03:20:00",
                              path=tmp_path/"x.csv")


def test_empty_historical_evaluation_returns_no_claims():
    report=v2.evaluate({"participation_proxy":[]},{})
    assert report=={}


def test_forward_curves_require_explicit_fitted_residual_pool():
    # Without empirically fitted 2024 player residuals, cannot make forecast
    # or invent sportsbook probability, and v2 returns zero candidate curves.
    assert v2.forward_curves([{
        "game_id":"2026_06_X_Y","market":"receptions",
        "position":"WR",
    }],{})==[]


def test_no_current_2026_retraining_opt_in():
    assert v2.DEFAULT_OPTIONS=={k:(5,1.0) for k in v2.METRICS}
    assert v2.VERSION=="PROP_PARTICIPATION_CONDITIONAL_V2_RESEARCH"
