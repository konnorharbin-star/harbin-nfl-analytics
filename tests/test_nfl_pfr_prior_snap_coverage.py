"""PFR snap context does not provide future activation or void certainty."""
import json
import pytest
from scripts.nfl_pfr_prior_snap_coverage import (
    unique_crosswalk,preceding_games,coverage,append_first_seen
)

def players():
    return [{"gsis_id":"00-PLAYERA","pfr_id":"PFR-A"},
            {"gsis_id":"00-PLAYERB","pfr_id":"PFR-B"}]

def schedules():
    return [{"game_id":f"2026_0{w}_BUF_LA","season":2026,"week":w,
         "game_type":"REG","home_team":"BUF","away_team":"LA",
         "home_score":24,"away_score":17} for w in (1,2,3,4)]

def snap(gid,team,pid,n,week):
    return {"game_id":gid,"game_type":"REG","season":2026,"week":week,
            "team":team,"pfr_player_id":pid,"offense_snaps":n}

def full():
    return {"game_id":"2026_05_BUF_LA","season":2026,"week":5,
       "first_observed_at_utc":"2026-10-12T04:36:00Z",
       "kickoff_utc":"2026-10-13T00:15:00Z",
       "team":"LA","player_id":"00-PLAYERA","market":"rushing_yards"}

def fixture_snaps():
    return [snap(f"2026_0{w}_BUF_LA","LAR","PFR-A",n,w)
            for w,n in [(1,20),(2,30),(3,40),(4,50)]] + [
          snap(f"2026_0{w}_BUF_LA","LAR","PFR-B",4,w) for w in (1,2,3,4)]

def test_unique_player_id_crosswalk_no_name_fallback():
    m=unique_crosswalk(players())
    assert m["00-PLAYERA"]=="PFR-A"
    p=players()+[{"gsis_id":"00-PLAYERX","pfr_id":"PFR-A"}]
    m=unique_crosswalk(p)
    assert "00-PLAYERA" not in m and "00-PLAYERX" not in m

def test_exact_gsis_to_pfr_prior_game_snap_coverage():
    out=coverage([full()],players(),fixture_snaps(),schedules(),
       "2026-10-12T11:30:00Z")
    assert len(out)==1
    a=out[0]
    assert a["pfr_id"]=="PFR-A"
    assert a["observed_offense_snaps"]==[20,30,40,50]
    assert a["observed_mean"]==35.0
    assert a["status"]=="HISTORICAL_OFFENSIVE_SNAPS_OBSERVED"
    assert a["forecast_active_probability"] is None
    assert not a["future_player_availability_verified"]
    assert not a["missing_as_zero"]

def test_original_player_missing_sourced_row_is_unknown_not_zero():
    x=dict(full(),player_id="00-PLAYERX")
    rows=coverage([x],players(),fixture_snaps(),schedules(),
       "2026-10-12T11:30:00Z")
    assert rows[0]["status"]=="MISSING_UNIQUE_PFR_GSIS_CROSSWALK"
    assert rows[0]["observed_mean"] is None

def test_only_strictly_previous_games_eligible():
    c=schedules()
    assert len(preceding_games(c,2026,3,"LA"))==2
    assert len(preceding_games(c,2026,5,"LA"))==4

def test_same_week_target_game_snaps_ignored():
    snap_rows=fixture_snaps()+[snap("2026_05_BUF_LA","LAR","PFR-A",145,5)]
    r=coverage([full()],players(),snap_rows,schedules(),
       "2026-10-12T11:30:00Z")[0]
    assert r["observed_mean"]==35.0
    assert 145 not in r["observed_offense_snaps"]

def test_missing_player_in_some_prior_games_no_zero_imputation():
    snap_rows=[s for s in fixture_snaps() if not (
         s["pfr_player_id"]=="PFR-A" and s["week"]==2)]
    r=coverage([full()],players(),snap_rows,schedules(),
       "2026-10-12T11:30:00Z")[0]
    assert r["observed_offense_snaps"]==[20,40,50]
    assert r["observed_mean"]==pytest.approx(110/3)
    assert not r["missing_as_zero"]

def test_no_postkickoff_coverage_new_row():
    r=coverage([full()],players(),fixture_snaps(),schedules(),
       "2026-10-13T00:16:00Z")
    assert r==[]

def test_snap_source_observation_must_follow_original_model():
    with pytest.raises(ValueError,match="cannot predate"):
        coverage([full()],players(),fixture_snaps(),schedules(),
                 "2026-10-11T11:30:00Z")

def test_duplicate_source_game_player_rejected():
    v=fixture_snaps()
    with pytest.raises(ValueError,match="ambiguous"):
        coverage([full()],players(),v+[v[0]],schedules(),
                 "2026-10-12T11:30:00Z")

def test_duplicate_original_prop_group_rejected():
    with pytest.raises(ValueError,match="duplicate original"):
        coverage([full(),full()],players(),fixture_snaps(),schedules(),
                 "2026-10-12T11:30:00Z")

def test_snap_counts_not_infinite():
    v=fixture_snaps()
    v[0]["offense_snaps"]=float("nan")
    with pytest.raises(ValueError,match="invalid historical"):
        coverage([full()],players(),v,schedules(),
                 "2026-10-12T11:30:00Z")

def test_original_coverage_never_overwritten(tmp_path):
    r=coverage([full()],players(),fixture_snaps(),schedules(),
       "2026-10-12T11:30:00Z")
    path=tmp_path/"snaps.jsonl"
    assert append_first_seen(r,path)["created"]==1
    other=[dict(r[0],observed_mean=0)]
    assert append_first_seen(other,path)["created"]==0
    assert json.loads(path.read_text())["observed_mean"]==35.0
