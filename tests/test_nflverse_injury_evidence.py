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
         "kickoff_utc": "2026-10-18T20:25:00+00:00",
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


def test_kickoff_missing_fail_closed_without_guessing_report_vintage():
    review=source()
    del review["games"][0]["kickoff_utc"]
    report=add_injury_reports(review,[entry()],season=2026,week=6,captured=AT)
    p=report["games"][0]["teams"]["LA"]["players"][0]
    assert p["nflverse_report_evidence"]=="MISSING_KICKOFF_FAIL_CLOSED"
    assert p["nflverse_reported_status"] is None
    assert report["nflverse_injury_source"]["source_rejections"]["missing_game_kickoff"]==1


def test_postkickoff_capture_cannot_be_backfilled_into_pregame_approval():
    late=AT+timedelta(days=6)
    report=add_injury_reports(source(),[entry()],season=2026,week=6,captured=late)
    p=report["games"][0]["teams"]["LA"]["players"][0]
    assert p["nflverse_report_evidence"]=="POSTKICKOFF_CAPTURE_NOT_PREGAME"
    assert p["nflverse_reported_status"] is None
    assert report["nflverse_injury_source"]["source_rejections"]["postkickoff_capture"]==1
    assert report["nflverse_injury_source"]["starting_qbs_verified"] is False


def test_explicit_unknown_injuries_never_reclassify_qb_as_starting():
    report=add_injury_reports(source(),[entry("Out")],season=2026,week=6,captured=AT)
    p=report["games"][0]["teams"]["LA"]["players"][0]
    assert p["source_proves_final_inactive"] is False
    assert p["source_proves_starting_qb"] is False
    assert report["nflverse_injury_source"]["date_modified_not_original_publication_proof"]
