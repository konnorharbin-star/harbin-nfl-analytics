"""Prospective stat-row forward grading is not betting settlement."""
import hashlib
import json
from datetime import datetime,timezone,timedelta
import pytest
from scripts.nfl_prop_stat_row_forward_grader import (
    FROZEN_MODEL,VERSION,observations,write_immutable_grades,score_grades,
    sha256_beta,read_jsonl
)

def pinned():
    beta=[.23,.48,.9,.55,.5]
    return {"model":FROZEN_MODEL,
        "development_year":2024,
        "full_beta":beta,
        "full_beta_sha256":sha256_beta(beta),
        "development_stat_row_presence_rate":.8234786361674579}

def game():
    return {"game_id":"2026_05_BUF_LA","game_type":"REG","season":2026,
        "week":5,"gameday":"2026-10-12","gametime":"20:15",
        "home_team":"LA","away_team":"BUF",
        "home_score":24,"away_score":14}

def original():
    return {"version":FROZEN_MODEL,"game_id":"2026_05_BUF_LA",
        "player_id":"00-TEST","wager":False,
        "first_observed_at_utc":"2026-10-12T11:00:00Z",
        "kickoff_utc":"2026-10-13T00:15:00+00:00",
        "p_official_stat_row_present_NOT_DNP":.75,
        "fit_beta_sha256":pinned()["full_beta_sha256"],
        "p_player_active":None,"p_prop_void":None,
        "estimated_true_EV":None}

def stats(include_player=True):
    rows=[]
    for tm in ("BUF","LA"):
        for i in range(7):
            rows.append({"game_id":"2026_05_BUF_LA","team":tm,
                "player_id":f"{tm}{i}"})
    if include_player:
        rows.append({"game_id":"2026_05_BUF_LA","team":"BUF","player_id":"00-TEST"})
    return rows

def test_maturity_requires_full_72hours_after_game():
    before="2026-10-13T01:00:00Z"
    y,reason=observations([original()],[game()],stats(),pinned(),captured_utc=before)
    assert not y and reason["SOURCE_MATURITY_72H_NOT_REACHED"]==1

def test_grades_recorded_stat_row_not_void():
    now="2026-10-16T09:00:00Z"
    r,reason=observations([original()],[game()],stats(),pinned(),captured_utc=now)
    assert not reason and len(r)==1
    assert r[0]["observed_official_player_stat_row_presence"]==1
    assert not r[0]["outcome_means_confirmed_DNP_or_void"]
    assert r[0]["money_ev"] is None and r[0]["betting_decision"]=="PASS"
    alt,_=observations([original()],[game()],stats(False),pinned(),captured_utc=now)
    assert alt[0]["observed_official_player_stat_row_presence"]==0
    assert not alt[0]["outcome_means_confirmed_DNP_or_void"]

def test_both_team_source_coverage_gate():
    truncated=stats()[:3]
    out,reason=observations([original()],[game()],truncated,pinned(),
        captured_utc="2026-10-16T09:00:00Z")
    assert not out
    assert reason["INSUFFICIENT_BOTH_TEAM_OFFICIAL_STAT_SOURCE_COVERAGE"]==1

def test_no_grade_after_kickoff_revision_without_verified_original_timeline():
    revised=dict(game(),gameday="2026-10-13")
    out,reason=observations([original()],[revised],stats(),pinned(),
                             captured_utc="2026-10-17T09:00:00Z")
    assert not out
    assert reason["KICKOFF_CHANGED_REQUIRES_MANUAL_ORIGINAL_TIMELINE_AUDIT"]==1

def test_source_cannot_change_original_pregame_probability():
    corrupted=dict(original(),p_official_stat_row_present_NOT_DNP=1.1)
    with pytest.raises(ValueError,match="probability"):
        observations([corrupted],[game()],stats(),pinned(),
                     captured_utc="2026-10-16T09:00:00Z")
    corrupted=dict(original(),first_observed_at_utc="2026-10-13T00:16:00Z")
    with pytest.raises(ValueError,match="not pregame"):
        observations([corrupted],[game()],stats(),pinned(),
                     captured_utc="2026-10-16T09:00:00Z")

def test_pinned_hash_is_mandatory():
    p=pinned();p["full_beta"][0]+=1
    with pytest.raises(ValueError,match="calibration invalid"):
        observations([original()],[game()],stats(),p,
                     captured_utc="2026-10-16T09:00:00Z")

def test_immutable_grade_then_append_data_correction(tmp_path):
    path=tmp_path/"grade.jsonl"
    at="2026-10-16T09:00:00Z"
    r,_=observations([original()],[game()],stats(),pinned(),captured_utc=at)
    w=write_immutable_grades(r,path)
    assert w["new_records"]==1 and w["source_corrections_appended"]==0
    assert write_immutable_grades(r,path)["new_records"]==0
    corrected,_=observations([original()],[game()],stats(False),pinned(),
         captured_utc="2026-10-17T09:00:00Z")
    w=write_immutable_grades(corrected,path)
    assert w["source_corrections_appended"]==1
    entries=read_jsonl(path)
    assert len(entries)==2
    assert entries[0]["revision"]==1 and entries[1]["revision"]==2
    assert entries[1]["record_type"]=="APPENDED_SOURCE_CORRECTION"
    assert entries[1]["previous_observed_row_presence"]==1

def test_no_results_if_none_are_matured():
    x=score_grades([])
    assert x["n_games"]==0 and x["calibration_sample_insufficient"]
    assert x["profitable_edge_proven"] is False

def test_forward_brier_is_calculated_from_latest_corrections():
    at="2026-10-16T09:00:00Z"
    yes,_=observations([original()],[game()],stats(),pinned(),captured_utc=at)
    no,_=observations([original()],[game()],stats(False),pinned(),captured_utc=at)
    for i,r in enumerate((yes[0],no[0])):
        r["game_id"]=f"game{i}"
    data=score_grades([yes[0],no[0]])
    assert data["n_player_games"]==2
    assert data["n_games"]==2
    assert data["calibration_sample_insufficient"]
    assert data["label_is_actually_bet_settlement"] is False

def test_duplicate_original_player_forecast_blocks_grading():
    with pytest.raises(ValueError,match="duplicate frozen"):
        observations([original(),original()],[game()],stats(),pinned(),
                     captured_utc="2026-10-16T09:00:00Z")
