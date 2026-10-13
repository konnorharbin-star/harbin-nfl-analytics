"""Synthetic source chronology checks for NFL weekly injury companion."""
from datetime import UTC,datetime,timedelta
from copy import deepcopy
import pytest
from nfl.nflverse_injury_evidence import add_injury_reports
from nfl.web_conditioned_model import status_candidates

AT=datetime(2026,10,13,20,tzinfo=UTC)
def source():
    return {"status":"SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY",
      "games":[{"game_id":"2026_06_BUF_LA",
         "teams":{"LA":{"players":[
             {"player_id":"00-0034211","espn_reported_status":None,
              "availability_evidence_status":"NOT_LISTED_IN_SNAPSHOT_NOT_CONFIRMED_HEALTHY"}]},
           "BUF":{"players":[]}}}]}

def entry(status="Questionable",offset=-2,*,week=6):
    return {"season":2026,"week":week,"game_type":"REG",
       "team":"LA","gsis_id":"00-0034211","report_status":status,
       "date_modified":(AT+timedelta(hours=offset)).isoformat()}

def test_weekly_gsis_match_becomes_hypothetical_candidate_not_confirmed_out():
    r=source();rows=[entry()]
    copy=deepcopy((r,rows))
    out=add_injury_reports(r,rows,season=2026,week=6,captured=AT)
    assert (r,rows)==copy
    p=out["games"][0]["teams"]["LA"]["players"][0]
    assert p["nflverse_reported_status"]=="questionable"
    assert p["source_proves_final_inactive"] is False
    assert out["nflverse_injury_source"]["matched_player_statuses"]==1
    candidates=status_candidates(out["games"][0])
    assert len(candidates)==1
    assert candidates[0][-1]=="NFLVERSE_WEEKLY_INJURY_REPORT"

def test_future_or_different_week_and_wrong_team_are_not_authorized():
    for record in (entry(offset=1),entry(week=5),dict(entry(),team="BUF")):
        out=add_injury_reports(source(),[record],season=2026,week=6,captured=AT)
        assert out["games"][0]["teams"]["LA"]["players"][0]["nflverse_reported_status"] is None
        assert status_candidates(out["games"][0])==[]

def test_not_listed_never_means_healthy_and_conflicts_block():
    out=add_injury_reports(source(),[],season=2026,week=6,captured=AT)
    p=out["games"][0]["teams"]["LA"]["players"][0]
    assert p["nflverse_report_evidence"]=="NOT_LISTED_DOES_NOT_PROVE_ACTIVE"
    assert p["source_proves_final_inactive"] is False
    x=entry("Out")
    y=entry("Questionable")
    report=add_injury_reports(source(),[x,y],season=2026,week=6,captured=AT)
    assert report["games"][0]["teams"]["LA"]["players"][0]["nflverse_report_evidence"]=="AMBIGUOUS_DATE_STATUS"
    assert status_candidates(report["games"][0])==[]

def test_latest_eligible_status_wins_without_asof_leak():
    older=entry("Doubtful",offset=-5)
    newer=entry("Questionable",offset=-1)
    future=entry("Out",offset=3)
    report=add_injury_reports(source(),[older,newer,future],season=2026,week=6,captured=AT)
    p=report["games"][0]["teams"]["LA"]["players"][0]
    assert p["nflverse_reported_status"]=="questionable"
    assert p["nflverse_report_modified_utc"]==(AT-timedelta(hours=1)).isoformat()
    assert p["source_proves_final_inactive"] is False
