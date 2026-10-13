"""Synthetic provenance, no-wager and conservation tests, not real forecasts."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import pytest
from nfl.web_conditioned_model import conditional_roles,status_candidates,generate

NOW=datetime(2026,10,13,20,tzinfo=timezone.utc)
FIRST=NOW-timedelta(hours=2)
WEB=NOW-timedelta(minutes=10)
KICK=NOW+timedelta(days=3)

def roles():
    z=[]
    for team in ("BUF","LA"):
        for name,kind,tar,car,qb in (
            ("QB","QB",0,.1,1),
            ("RB","RB",.1,.7,0),
            ("WR","WR",.6,0,0),
            ("TE","TE",.2,.1,0),
        ):
            z.append({"team":team,"player_id":team+"_"+name,"role":kind,
                      "target_share":tar,"carry_share":car,"qb_pass_share":qb})
    return z

def fixture():
    frozen={"record_kind":"FROZEN_FIRST_SEEN_FORECAST",
            "event_id":"G1","content_sha256":"a"*64,
            "captured_utc":FIRST.isoformat(),"kickoff_utc":KICK.isoformat(),
            "home_team":"BUF","away_team":"LA"}
    baseline={"spec":"NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1",
        "capture_utc":FIRST.isoformat(),"season":2026,"week":6,
        "games":[{"game_id":"G1","status":"UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH",
                  "frozen_football_score_input":frozen,"draws":1000,"seed":32,
                  "distribution":{"TOTAL":{"mean":50},"HOME_MARGIN":{"mean":2},
                    "BUF:points":{"mean":26},"LA:points":{"mean":24},
                    "BUF:BUF_WR:receptions":{"mean":6},
                    "BUF:BUF_TE:receptions":{"mean":2}}}]}
    replay={"spec":"NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1",
            "capture_utc":FIRST.isoformat(),"season":2026,"week":6,
            "games":[{"game_id":"G1","forecast":deepcopy(frozen),
                      "drives":[],"roles":roles()}]}
    inj={"status":"SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY",
         "games":[{"game_id":"G1","kickoff_utc":KICK.isoformat(),
           "teams":{"BUF":{"players":[
             {"player_id":"BUF_WR","espn_athlete_id":"12",
              "espn_reported_status":"Questionable",
              "availability_evidence_status":"ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS"}]},
             "LA":{"players":[]}}}]}
    web={"version":"NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1",
         "captured_utc":WEB.isoformat(),
         "paired_original_model_capture_utc":FIRST.isoformat(),
         "player_evidence_review":inj}
    return baseline,replay,web

def mock_model(forecast,drives,adjusted,*,asof,draws,seed):
    assert asof==WEB and draws==1000 and seed==32
    src={r["player_id"]:r for r in adjusted}
    assert src["BUF_WR"]["target_share"]==0
    return {"distribution":{"TOTAL":{"mean":50},"HOME_MARGIN":{"mean":2},
           "BUF:points":{"mean":26},"LA:points":{"mean":24},
           "BUF:BUF_WR:receptions":{"mean":0},
           "BUF:BUF_TE:receptions":{"mean":8}}}

def test_web_evidence_scenario_keeps_original_frozen_and_no_wagers():
    a,b,c=fixture()
    before=deepcopy((a,b,c))
    x=generate(a,b,c,now=NOW,simulate=mock_model)
    assert (a,b,c)==before
    assert x["count"]==1
    item=x["scenarios"][0]
    assert item["scenario"]=="IF_PLAYER_UNAVAILABLE_NOT_CONFIRMED"
    assert item["baseline_total_mean"]==item["scenario_total_mean"]==50
    assert item["baseline_player_means"]["BUF:BUF_WR:receptions"]==6
    assert item["scenario_player_means"]["BUF:BUF_WR:receptions"]==0
    assert item["probability_player_unavailable"] is None
    assert x["original_forecast_modified"] is False
    assert x["estimated_ev"] is None and x["wager_authorized"] is False

def test_share_mass_conservation_and_qb_exclusion_fails_closed():
    initial=roles()
    new=conditional_roles(initial,team="BUF",missing_id="BUF_WR")
    assert sum(r["target_share"] for r in new if r["team"]=="BUF")==pytest.approx(.9)
    assert sum(r["carry_share"] for r in new if r["team"]=="BUF")==pytest.approx(.9)
    assert next(r for r in new if r["player_id"]=="BUF_WR")["target_share"]==0
    assert next(r for r in initial if r["player_id"]=="BUF_WR")["target_share"]==.6
    with pytest.raises(ValueError,match="QB_REPLACEMENT"):
        conditional_roles(initial,team="BUF",missing_id="BUF_QB")

def test_web_report_uncertain_and_unmatched_names_do_not_force_predictions():
    a,b,c=fixture()
    c["player_evidence_review"]["games"][0]["teams"]["BUF"]["players"][0]["espn_reported_status"]="Limited"
    assert generate(a,b,c,now=NOW,simulate=mock_model)["count"]==0
    a,b,c=fixture()
    c["player_evidence_review"]["games"][0]["teams"]["BUF"]["players"][0]["player_id"]="WR_NAME_ONLY"
    x=generate(a,b,c,now=NOW,simulate=mock_model)
    assert x["count"]==0 and x["blocked"][0]["reason"]=="ORIGINAL_PLAYER_ID_NOT_UNIQUE"

def test_provenance_missing_and_future_sources_fail_closed():
    a,b,c=fixture()
    c["captured_utc"]=(KICK+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="NOT_PREKICKOFF"):
        generate(a,b,c,now=KICK+timedelta(hours=2),simulate=mock_model)
    a,b,c=fixture()
    b["games"][0]["forecast"]["content_sha256"]="b"*64
    with pytest.raises(ValueError,match="FROZEN_BASELINE"):
        generate(a,b,c,now=NOW,simulate=mock_model)
    a,b,c=fixture()
    c["paired_original_model_capture_utc"]=(FIRST+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="MISMATCHED_OR_FUTURE"):
        generate(a,b,c,now=NOW,simulate=mock_model)
