"""All numbers synthetic; engineering invariants NOT forecast validation."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import pytest
from nfl.joint_score_shadow import score_shadow

AT=datetime(2026,10,12,18,tzinfo=UTC)
FORECAST_CAPTURE=AT-timedelta(hours=1)
KICK=AT+timedelta(days=6)

def forecast():
    return {"league":"NFL","record_kind":"FROZEN_FIRST_SEEN_FORECAST",
      "event_id":"TEST_NEXT_GAME","model_version":"SYNTHETIC_INDEPENDENT_SCORE",
      "content_sha256":"a"*64,"captured_utc":FORECAST_CAPTURE.isoformat(),
      "kickoff_utc":KICK.isoformat(),"independent_home_points":26.2,
      "independent_away_points":21.1,"independent_projected_total":47.3,
      "independent_projected_home_margin":5.1}

def corpus():
    start=datetime(2025,1,5,tzinfo=UTC)
    rows=[]
    for i in range(64):
        kickoff=start+timedelta(weeks=i//8)
        rows.append({"game_id":f"HIST_TEST_{i}","season":2025,"week":i//8+1,
          "forecast_captured_utc":(kickoff-timedelta(days=2)).isoformat(),
          "kickoff_utc":kickoff.isoformat(),
          "final_first_verified_utc":(kickoff+timedelta(days=4)).isoformat(),
          "independent_final_source_verified":True,
          "original_forecast_sha256":"b"*64,"final_source_sha256":"c"*64,
          "forecast_home_points":23.1,"forecast_away_points":22.3,
          "actual_home_points":21+i%10,"actual_away_points":17+i%12})
    return rows

def test_bivariate_simulation_coheres_repeats_and_never_claims_bets():
    f,h=forecast(),corpus()
    before_f,before_h=deepcopy(f),deepcopy(h)
    a=score_shadow(f,h,at=AT,seed=2,draws=4000,total_line=48,home_spread=-3)
    b=score_shadow(f,h,at=AT,seed=2,draws=4000,total_line=48,home_spread=-3)
    assert a==b
    assert f==before_f and h==before_h
    assert a["training_games"]==64 and a["training_weeks"]==8
    assert a["score_distributions"]["home"]["mean"]+a["score_distributions"]["away"]["mean"]==pytest.approx(a["score_distributions"]["total"]["mean"])
    assert a["score_distributions"]["home"]["mean"]-a["score_distributions"]["away"]["mean"]==pytest.approx(a["score_distributions"]["home_margin"]["mean"])
    assert a["home_win"]+a["away_win"]+a["tie"]==pytest.approx(1)
    for v in a["thresholds"].values():
        assert sum(v[k] for k in v if k!="line")==pytest.approx(1)
    assert a["is_drive_or_player_simulator"] is False
    assert a["real_book_quote"] is None and a["estimated_ev"] is None
    assert a["wager_authorized"] is False and a["action"]=="PASS_RESEARCH_ONLY"

def test_rejects_missing_grade_and_post_cutoff_grade():
    for change in (
        {"independent_final_source_verified":False},
        {"final_source_sha256":None},
        {"final_first_verified_utc":(AT+timedelta(hours=1)).isoformat()},
        {"final_first_verified_utc":(datetime(2025,1,5,tzinfo=UTC)+timedelta(hours=20)).isoformat()},
        {"forecast_captured_utc":(AT+timedelta(days=1)).isoformat()},
        {"game_id":"TEST_NEXT_GAME"},
    ):
        rows=corpus();rows[0].update(change)
        with pytest.raises(ValueError):
            score_shadow(forecast(),rows,at=AT,draws=1000)

def test_weak_history_and_duplicate_game_fail_closed():
    with pytest.raises(ValueError,match="INSUFFICIENT"):
        score_shadow(forecast(),corpus()[:63],at=AT)
    rows=corpus();rows[-1]["game_id"]=rows[0]["game_id"]
    with pytest.raises(ValueError,match="DUPLICATE"):
        score_shadow(forecast(),rows,at=AT)

def test_original_score_forecast_must_be_available_before_kickoff():
    f=forecast();f["captured_utc"]=(KICK+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="FORECAST_NOT"):
        score_shadow(f,corpus(),at=AT)
    f=forecast();f["independent_projected_total"]=46
    with pytest.raises(ValueError,match="FORECAST_SCORE"):
        score_shadow(f,corpus(),at=AT)
    with pytest.raises(ValueError,match="FORECAST_NOT"):
        score_shadow(forecast(),corpus(),at=KICK+timedelta(hours=1))

def test_rejects_invalid_threshold_drawcount_and_unzoned_clock():
    with pytest.raises(ValueError,match="INVALID_THRESHOLD"):
        score_shadow(forecast(),corpus(),at=AT,total_line=float("nan"))
    with pytest.raises(ValueError,match="INVALID_DRAWS"):
        score_shadow(forecast(),corpus(),at=AT,draws=999)
    with pytest.raises(ValueError,match="UNZONED"):
        score_shadow(forecast(),corpus(),at=AT.replace(tzinfo=None))

def test_every_sample_is_integer_and_nonnegative_even_under_large_negative_error():
    rows=corpus()
    for r in rows:
        r["actual_home_points"]=0
        r["actual_away_points"]=0
        r["forecast_home_points"]=60
        r["forecast_away_points"]=60
    x=score_shadow(forecast(),rows,at=AT,draws=1000)
    assert x["score_distributions"]["home"]["mean"]==0
    assert x["score_distributions"]["away"]["mean"]==0
    assert x["score_distributions"]["total"]["mean"]==0
    assert x["tie"]==1
