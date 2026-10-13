"""Synthetic nflreadpy-schema integration: score -> prior PBP drives -> player shares."""
from datetime import UTC, datetime, timedelta
import polars as pl
import pytest

from nfl.coupled_nflverse_asof import kickoff_utc, prepare_asof_source
from nfl.coupled_drive_player_shadow import simulate_joint

AT=datetime(2026,10,13,12,tzinfo=UTC)

def fixture_data():
    schedules=[]
    pbp=[]
    base=datetime(2025,9,7,tzinfo=UTC)
    for i in range(48):
        ko=base+timedelta(weeks=i//8)
        game=f"2025_{i:03d}_BUF_LA"
        schedules.append({"season":2025,"week":i//8+1,"game_id":game,
              "game_type":"REG","gameday":ko.strftime("%Y-%m-%d"),
              "gametime":"13:00","home_team":"LA","away_team":"BUF",
              "home_score":24+i%5,"away_score":20+i%8})
        for team in ("LA","BUF"):
            for j in range(11):
                drive=f"{team}_{j}"
                res=("Punt","Field goal","Touchdown")[j%3]
                pbp.append({"game_id":game,"fixed_drive":drive,
                     "fixed_drive_result":res,"posteam":team,
                     "play_type":"pass","pass_attempt":1,
                     "complete_pass":0 if j%3==0 else 1,
                     "passing_yards":0 if j%3==0 else 25+j,
                     "rush_attempt":0,"rushing_yards":0,
                     "pass_touchdown":1 if j%3==2 else 0,
                     "rush_touchdown":0,
                     "interception":1 if j%3==0 else 0,
                     "two_point_attempt":0})
                pbp.append({"game_id":game,"fixed_drive":drive,
                     "fixed_drive_result":res,"posteam":team,
                     "play_type":"run","pass_attempt":0,"complete_pass":0,
                     "passing_yards":0,"rush_attempt":1,"rushing_yards":5+j,
                     "pass_touchdown":0,"rush_touchdown":0,
                     "interception":0,"two_point_attempt":0})
    schedules.append({"season":2026,"week":6,"game_id":"2026_06_BUF_LA",
        "game_type":"REG","gameday":"2026-10-18","gametime":"16:05",
        "home_team":"LA","away_team":"BUF","home_score":None,"away_score":None})
    stats=[]
    for team in ("LA","BUF"):
        for wk in range(1,6):
            for player,pos,targets,carries in [
                ("wr","WR",9,0),("rb","RB",2,18),("qb","QB",0,3),
                ("te","TE",3,0)]:
                stats.append({"team":team,"player_id":team+"_"+player,
                              "position":pos,"season":2026,"week":wk,
                              "targets":targets,"carries":carries,
                              "attempts":28 if pos=="QB" else 0})
    return pl.DataFrame(schedules),pl.DataFrame(pbp),pl.DataFrame(stats)

def test_source_adapter_generates_genuinely_runnable_football_input():
    schedule,plays,stats=fixture_data()
    f,d,p,meta=prepare_asof_source(
        schedule,plays,stats,season=2026,week=6,game_id="2026_06_BUF_LA",at=AT)
    assert meta["eligible_schedule_games"]==48
    assert meta["derived_drives"]==48*22
    assert f["independent_projected_total"]==pytest.approx(
        f["independent_home_points"]+f["independent_away_points"])
    assert f["captured_utc"]==AT.isoformat()
    assert all(x["verified_at_utc"]==AT.isoformat() for x in d)
    assert all(x["activity"]=="UNKNOWN" for x in p)
    result=simulate_joint(f,d,p,asof=AT,draws=1000)
    assert result["status"]=="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH"
    assert result["training_games"]==48
    assert "LA" in result["qb_passing_stat_projections_available_for_teams"]
    assert "BUF" in result["qb_passing_stat_projections_available_for_teams"]
    assert result["distribution"]["LA:LA_qb:passing_yards"]["mean"]>0
    assert result["distribution"]["BUF:BUF_qb:passing_attempts"]["mean"]>0
    assert result["qb_passing_player_ids_not_confirmed_starters"] is True
    assert result["source_verified_td_int_drive_classification"] is True
    assert result["distribution"]["LA:passing_touchdowns"]["mean"]>0
    assert result["distribution"]["LA:passing_interceptions"]["mean"]>0
    assert result["distribution"]["LA:LA_qb:passing_touchdowns"]["mean"]>0
    assert result["distribution"]["LA:LA_qb:passing_interceptions"]["mean"]>0
    assert result["distribution"]["LA:LA_wr:receiving_touchdowns"]["mean"]>0
    assert result["td_int_betting_authorized"] is False
    assert result["wager_authorized"] is False

def test_malformed_source_kickoff_or_pregame_capture_fails_closed():
    s,p,stats=fixture_data()
    assert kickoff_utc({"gameday":"2026-10-18","gametime":"16:05"}).isoformat()=="2026-10-18T20:05:00+00:00"
    with pytest.raises(ValueError,match="MISSING_SOURCE_KICKOFF"):
        kickoff_utc({})
    with pytest.raises(ValueError,match="SOURCE_NOT_PROSPECTIVE"):
        prepare_asof_source(s,p,stats,season=2026,week=6,
                            game_id="2026_06_BUF_LA",
                            at=datetime(2026,10,19,12,tzinfo=UTC))
    with pytest.raises(ValueError,match="MISSING_OR_DUPLICATE"):
        prepare_asof_source(s,p,stats,season=2026,week=5,
                            game_id="DOES_NOT_EXIST",at=AT)
