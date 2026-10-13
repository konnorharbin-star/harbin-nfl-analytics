"""Synthetic-only conservation, chronology and side-effect tests for joint shadow."""
from datetime import UTC, datetime, timedelta
from copy import deepcopy
import pytest
from nfl.coupled_drive_player_shadow import simulate_joint

NOW=datetime(2026,10,13,11,tzinfo=UTC)
CAPTURE=NOW-timedelta(minutes=10)
KICK=NOW+timedelta(days=5)

def forecast():
    return {"record_kind":"FROZEN_FIRST_SEEN_FORECAST","event_id":"TEST_WEEK_6_BUF_LA",
            "home_team":"LA","away_team":"BUF","model_version":"SYNTH_TEST_RIDGE",
            "content_sha256":"a"*64,"captured_utc":CAPTURE.isoformat(),
            "kickoff_utc":KICK.isoformat(),
            "independent_home_points":26.0,"independent_away_points":24.0,
            "independent_projected_total":50.0,
            "independent_projected_home_margin":2.0}

def history():
    first=datetime(2025,9,7,tzinfo=UTC)
    records=[]
    for g in range(40):
        ko=first+timedelta(weeks=g//8)
        for t in ("X","Y"):
            for j in range(11):
                pts=(0,3,7)[j%3]
                records.append({"game_id":f"2025_GAME_{g}","drive_id":f"{t}_{j}",
                  "team":t,"kickoff_utc":ko.isoformat(),
                  "verified_at_utc":(ko+timedelta(days=4)).isoformat(),
                  "completed_nflverse_schedule_game":True,"source_sha256":"c"*64,
                  "season":2025,"week":g//8+1,
                  "points":pts,"pass_attempts":3+(j%3),"completions":2,
                  "passing_yards":15+(j%4)*4,"rush_attempts":2+(j%2),
                  "rushing_yards":8+(j%5)})
    # 40 games, but needs six distinct weeks. Add five more eligible games.
    for g in range(40,48):
        ko=first+timedelta(weeks=5)
        for t in ("X","Y"):
            for j in range(11):
                records.append({"game_id":f"2025_GAME_{g}","drive_id":f"{t}_{j}",
                  "team":t,"kickoff_utc":ko.isoformat(),
                  "verified_at_utc":(ko+timedelta(days=4)).isoformat(),
                  "completed_nflverse_schedule_game":True,"source_sha256":"c"*64,
                  "season":2025,"week":6,
                  "points":(0,3,7)[j%3],"pass_attempts":3,
                  "completions":2,"passing_yards":21,
                  "rush_attempts":2,"rushing_yards":8})
    return records

def roles():
    rows=[]
    for team in ("LA","BUF"):
        for pid,role,tar,car in (("qb","QB",0.0,.12),
                                 ("rb","RB",.12,.78),
                                 ("wr","WR",.52,.0),
                                 ("te","TE",.19,.0)):
            rows.append({"team":team,"player_id":team+"_"+pid,
                         "role":role,"target_share":tar,"carry_share":car,
                         "qb_pass_share":.92 if role=="QB" else 0.0,
                         "observed_at_utc":(CAPTURE-timedelta(hours=3)).isoformat(),
                         "activity":"UNKNOWN"})
    return rows

def run(**kwargs):
    return simulate_joint(forecast(),history(),roles(),asof=NOW,draws=1000,seed=32,**kwargs)

def test_determinism_score_and_team_player_accounting():
    a=run(queries=[{"metric":"TOTAL","line":52.5},
                   {"metric":"LA:LA_wr:receiving_yards","line":30.5},
                   {"metric":"LA:LA_rb:rushing_yards","line":20.5}])
    b=run(queries=[{"metric":"TOTAL","line":52.5},
                   {"metric":"LA:LA_wr:receiving_yards","line":30.5},
                   {"metric":"LA:LA_rb:rushing_yards","line":20.5}])
    assert a==b
    dist=a["distribution"]
    assert dist["LA:points"]["mean"]+dist["BUF:points"]["mean"]==pytest.approx(dist["TOTAL"]["mean"])
    assert dist["LA:points"]["mean"]-dist["BUF:points"]["mean"]==pytest.approx(dist["HOME_MARGIN"]["mean"])
    for team in ("LA","BUF"):
        assert sum(dist[f"{team}:{team}_{p}:receptions"]["mean"] for p in ("rb","wr","te"))<=dist[f"{team}:completions"]["mean"]
        assert sum(dist[f"{team}:{team}_{p}:carries"]["mean"] for p in ("qb","rb"))<=dist[f"{team}:rush_attempts"]["mean"]
    for row in a["query_thresholds"]:
        assert row["over"]+row["under"]+row["push"]==pytest.approx(1)
        assert row["betting_ev"] is None
    assert a["wager_authorized"] is False and a["estimated_ev"] is None
    assert a["accuracy_improvement_validated"] is False
    assert a["training_games"]==48 and a["training_weeks"]==6
    assert a["home_win"]+a["away_win"]+a["tie"]==pytest.approx(1)

def test_inputs_not_mutated_and_solely_pregame_snapshot():
    f,h,p=forecast(),history(),roles()
    snap=deepcopy((f,h,p))
    simulate_joint(f,h,p,asof=NOW,draws=1000)
    assert (f,h,p)==snap

def test_bad_history_rejected_instead_of_giving_confident_results():
    for change in (
      {"completed_nflverse_schedule_game":False},
      {"verified_at_utc":(NOW+timedelta(days=1)).isoformat()},
      {"source_sha256":None},
      {"completions":9},
      {"passing_yards":-8},
      {"points":8},
    ):
        h=history();h[0].update(change)
        with pytest.raises(ValueError):
            simulate_joint(forecast(),h,roles(),asof=NOW,draws=1000)
    h=history();h[-1]["drive_id"]=h[0]["drive_id"];h[-1]["game_id"]=h[0]["game_id"]
    with pytest.raises(ValueError,match="DUPLICATE"):
        simulate_joint(forecast(),h,roles(),asof=NOW,draws=1000)
    with pytest.raises(ValueError,match="INSUFFICIENT"):
        simulate_joint(forecast(),history()[:50],roles(),asof=NOW,draws=1000)

def test_forecast_time_frozen_and_no_past_game_prediction():
    f=forecast()
    f["captured_utc"]=(NOW+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError,match="FORECAST_NOT"):
        simulate_joint(f,history(),roles(),asof=NOW,draws=1000)
    f=forecast();f["kickoff_utc"]=(NOW-timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="FORECAST_NOT"):
        simulate_joint(f,history(),roles(),asof=NOW,draws=1000)
    f=forecast();f["independent_projected_total"]=21
    with pytest.raises(ValueError,match="INCOHERENT"):
        simulate_joint(f,history(),roles(),asof=NOW,draws=1000)

def test_roster_sums_missing_qb_and_availability_block_unsupported_usage():
    p=roles();p[0]["observed_at_utc"]=(NOW+timedelta(days=1)).isoformat()
    with pytest.raises(ValueError,match="LOOKAHEAD"):
        simulate_joint(forecast(),history(),p,asof=NOW,draws=1000)
    p=roles();p[0]["carry_share"]=.6
    with pytest.raises(ValueError,match="PLAYER_SHARES"):
        simulate_joint(forecast(),history(),p,asof=NOW,draws=1000)
    p=roles();p[0]["activity"]="OUT"
    with pytest.raises(ValueError,match="ACTIVITY"):
        simulate_joint(forecast(),history(),p,asof=NOW,draws=1000)
    with pytest.raises(ValueError,match="MISSING_TEAM"):
        simulate_joint(forecast(),history(),[],asof=NOW,draws=1000)
    with pytest.raises(ValueError,match="UNKNOWN"):
        run(queries=[{"metric":"UNKNOWN","line":1.5}])
