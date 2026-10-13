"""Exact athlete ID / GSIS injury evidence, no premature starter confirmation."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import pytest

from nfl.player_evidence_bridge import review_player_evidence
from nfl.web_context_research import parse_espn_injuries

AT=datetime(2026,10,13,12,tzinfo=UTC)
KICK=AT+timedelta(days=4)

def frozen():
    return {"spec":"NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1",
      "capture_utc":(AT-timedelta(minutes=3)).isoformat(),
      "games":[{"game_id":"2026_06_BUF_LA",
        "status":"UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH",
        "frozen_football_score_input":{
          "home_team":"LA","away_team":"BUF","kickoff_utc":KICK.isoformat(),
          "captured_utc":(AT-timedelta(minutes=3)).isoformat()},
        "distribution":{
          "LA:00-0000001:passing_attempts":{"mean":29.5},
          "LA:00-0000001:passing_yards":{"mean":235},
          "LA:00-0000002:receptions":{"mean":4.2},
          "BUF:00-0000003:passing_attempts":{"mean":33.1},
          "BUF:00-0000003:passing_yards":{"mean":248},
          "LA:OTHER:passing_yards":{"mean":12},
          "LA:points":{"mean":25},
        }}]}

def report():
    return {"version":"NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1",
      "captured_utc":AT.isoformat(),
      "games":[{"game_id":"2026_06_BUF_LA","home_team":"LA","away_team":"BUF",
        "kickoff_utc":KICK.isoformat(),
        "structured_injury_snapshot":{
            "captured_utc":AT.isoformat(),
            "LA":[{"player":"Sample QB","espn_athlete_id":"40401","reported_status":"Questionable"},
                  {"player":"Sample WR","espn_athlete_id":"40402","reported_status":"Limited"}],
            "BUF":None,
        }}]}

def players():
    return [{"gsis_id":"00-0000001","espn_id":"40401","display_name":"Sample QB","position":"QB"},
            {"gsis_id":"00-0000002","espn_id":"40402","display_name":"Sample WR","position":"WR"},
            {"gsis_id":"00-0000003","espn_id":40403,"display_name":"Other QB","position":"QB"}]

def test_exact_id_crosswalk_and_injury_source_gap_research_only():
    a,b,c=frozen(),report(),players()
    before=deepcopy((a,b,c))
    x=review_player_evidence(a,b,c,at=AT)
    assert (a,b,c)==before
    assert x["model_fair_score_or_prop_adjusted"] is False
    assert x["wager_authorized"] is False
    assert x["estimated_ev"] is None
    game=x["games"][0]
    h=game["teams"]["LA"]
    assert len(h["players"])==2
    assert len(h["historical_qb_volume_candidates"])==1
    qb=next(v for v in h["players"] if v["player_id"]=="00-0000001")
    assert qb["name_from_nflverse"]=="Sample QB"
    assert qb["availability_evidence_status"]=="ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS"
    assert qb["espn_reported_status"]=="Questionable"
    assert qb["verified_active"] is False
    assert qb["verified_starting_qb"] is False
    aw=game["teams"]["BUF"]["players"][0]
    assert aw["availability_evidence_status"]=="INJURY_SOURCE_UNAVAILABLE"
    assert game["starter_verified"] is False

def test_unknown_player_not_listed_does_not_mean_active():
    p=report()
    p["games"][0]["structured_injury_snapshot"]["LA"]=[]
    x=review_player_evidence(frozen(),p,players(),at=AT)
    statuses=[r["availability_evidence_status"] for r in x["games"][0]["teams"]["LA"]["players"]]
    assert statuses==["NOT_LISTED_IN_SNAPSHOT_NOT_CONFIRMED_HEALTHY"]*2
    assert not x["official_week_specific_inactives_verified"]

def test_ambiguous_espn_id_and_name_spoofing_do_not_cause_wrong_match():
    p=report()
    p["games"][0]["structured_injury_snapshot"]["LA"][0].update(
        {"player":"Completely Different Person","espn_athlete_id":"40401"})
    out=review_player_evidence(frozen(),p,players(),at=AT)
    assert out["games"][0]["teams"]["LA"]["players"][0]["availability_evidence_status"]=="ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS"
    directory=players()+[{"gsis_id":"00-0999999","espn_id":"40401","display_name":"Other Person"}]
    out=review_player_evidence(frozen(),report(),directory,at=AT)
    assert out["games"][0]["teams"]["LA"]["players"][0]["availability_evidence_status"]=="AMBIGUOUS_ESPN_ID_CROSSWALK"

def test_duplicate_injury_item_blocks_false_status_corroboration():
    a=report()
    a["games"][0]["structured_injury_snapshot"]["LA"].append(
        dict(a["games"][0]["structured_injury_snapshot"]["LA"][0]))
    out=review_player_evidence(frozen(),a,players(),at=AT)
    assert out["games"][0]["teams"]["LA"]["players"][0]["availability_evidence_status"]=="AMBIGUOUS_INJURY_RECORDS"

def test_postkickoff_and_source_capture_futures_rejected():
    p=report()
    p["captured_utc"]=(KICK+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="CHRONOLOGY|POSTKICKOFF"):
        review_player_evidence(frozen(),p,players(),at=KICK+timedelta(hours=3))
    p=report()
    p["games"][0]["home_team"]="BUF"
    with pytest.raises(ValueError,match="CONFLICTING"):
        review_player_evidence(frozen(),p,players(),at=AT)
    p=report()
    p["games"][0]["structured_injury_snapshot"]["captured_utc"]=(AT-timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError,match="INJURY_CAPTURE"):
        review_player_evidence(frozen(),p,players(),at=AT)

def test_espn_parser_retains_exact_athlete_id_and_never_calls_it_final():
    feed={"injuries":[{"team":{"abbreviation":"LA"},
      "injuries":[{"athlete":{"id":"40401","fullName":"Sample QB"},
                   "status":{"name":"Questionable"}}]}]}
    data=parse_espn_injuries(feed,"LA")
    assert data[0]["espn_athlete_id"]=="40401"
    assert data[0]["status_is_official_week_specific_final_game_status"] is False
