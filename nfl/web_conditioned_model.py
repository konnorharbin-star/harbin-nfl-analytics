"""Source-conditioned NFL research player-availability scenarios."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import math

SPEC='NFL_WEB_CONDITIONED_PLAYER_RESEARCH_V1'
PRIORITY={'out':0,'injured reserve':0,'doubtful':1,'questionable':2}

def utc(value):
    if not isinstance(value,str):
        raise ValueError("MISSING_TIMESTAMP")
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    if dt.tzinfo is None:
        raise ValueError("UNZONED_TIMESTAMP")
    return dt.astimezone(timezone.utc)

def conditional_roles(original_roles, *, team, missing_id):
    """Hypothetical absence transfers historic opportunities, NOT score effects."""
    roles=deepcopy(original_roles)
    teammates=[r for r in roles if r.get("team")==team]
    targets=[r for r in teammates if r.get("player_id")==missing_id]
    if len(targets)!=1:raise ValueError("ORIGINAL_PLAYER_ID_NOT_UNIQUE")
    absent=targets[0]
    if absent.get("role")=="QB":
        raise ValueError("QB_REPLACEMENT_MODEL_UNVERIFIED")
    adjusted=False
    for key in ("target_share","carry_share"):
        for row in teammates:
            weight=row.get(key,0)
            if (isinstance(weight,bool) or not isinstance(weight,(float,int))
                or not math.isfinite(weight) or weight<0):
                raise ValueError("INVALID_HISTORICAL_ROLE_WEIGHT")
        total=sum(float(r.get(key,0)) for r in teammates)
        if total>1.000001:raise ValueError("ROLE_SHARES_EXCEED_ONE")
        removed=float(absent.get(key,0))
        if removed==0:continue
        remaining=total-removed
        if remaining<=0:raise ValueError("NO_HISTORICAL_REPLACEMENT")
        ratio=total/remaining
        for row in teammates:
            row[key]=0.0 if row is absent else float(row.get(key,0))*ratio
        adjusted=True
    if not adjusted:raise ValueError("NO_EXISTING_PLAYER_OPPORTUNITY")
    return roles

def status_candidates(evidence):
    """Reports are conditional leads only, not confirmed game inactives."""
    result=[]
    for team,data in evidence.get("teams",{}).items():
        for p in data.get("players",[]):
            status=str(p.get("espn_reported_status") or "").strip().lower()
            if (p.get("availability_evidence_status")=="ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS"
                and p.get("espn_athlete_id") and isinstance(p.get("player_id"),str)
                and status in PRIORITY):
                result.append((PRIORITY[status],team,p["player_id"],status))
    return sorted(result)

def generate(original, frozen_inputs, web, *, now, simulate, max_total=28):
    """Replay SAME frozen training inputs under a reported absence hypothetical."""
    if now.tzinfo is None:raise ValueError("UNZONED_CLOCK")
    if original.get("spec")!="NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1":
        raise ValueError("INVALID_ORIGINAL_FORECAST")
    if frozen_inputs.get("spec")!="NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1":
        raise ValueError("INVALID_FROZEN_INPUTS")
    if web.get("version")!="NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1":
        raise ValueError("INVALID_WEB_CONTEXT")
    observation=web.get("player_evidence_review",{})
    if observation.get("status")!="SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY":
        raise ValueError("NO_ID_VERIFIED_PLAYER_EVIDENCE")
    original_at=utc(original["capture_utc"])
    web_at=utc(web["captured_utc"])
    if (utc(frozen_inputs["capture_utc"])!=original_at
        or utc(web["paired_original_model_capture_utc"])!=original_at
        or not original_at<=web_at<=now.astimezone(timezone.utc)):
        raise ValueError("MISMATCHED_OR_FUTURE_SOURCE_CAPTURE")
    if (frozen_inputs.get("season"),frozen_inputs.get("week"))!=(original.get("season"),original.get("week")):
        raise ValueError("MISMATCHED_SEASON_WEEK")
    if isinstance(max_total,bool) or not isinstance(max_total,int) or not 1<=max_total<=64:
        raise ValueError("INVALID_SCENARIO_LIMIT")
    games=[g for g in original.get("games",[]) if g.get("status")=="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH"]
    if not games:raise ValueError("NO_ORIGINAL_RESEARCH_FORECASTS")
    if len({g["game_id"] for g in games})!=len(games):
        raise ValueError("DUPLICATED_ORIGINAL_GAME")
    base={g["game_id"]:g for g in games}
    replay={g["game_id"]:g for g in frozen_inputs["games"]}
    source={g["game_id"]:g for g in observation["games"]}
    if len(replay)!=len(frozen_inputs["games"]) or len(source)!=len(observation["games"]):
        raise ValueError("DUPLICATE_REPLAY_OR_WEB_GAME")
    if set(base)!=set(replay) or set(base)!=set(source):
        raise ValueError("SOURCE_GAME_COVERAGE_MISMATCH")
    scenarios,blocked=[],[]
    for gid in sorted(base):
        baseline=base[gid]
        data=replay[gid]
        forecast=data["forecast"]
        freeze=baseline["frozen_football_score_input"]
        if (forecast.get("content_sha256")!=freeze.get("content_sha256")
            or forecast.get("event_id")!=gid or freeze.get("event_id")!=gid
            or forecast.get("captured_utc")!=freeze.get("captured_utc")
            or forecast.get("kickoff_utc")!=freeze.get("kickoff_utc")):
            raise ValueError("FROZEN_BASELINE_MISMATCH")
        if not utc(forecast["captured_utc"])<=original_at<=web_at<utc(forecast["kickoff_utc"]):
            raise ValueError("NOT_PREKICKOFF_ORIGINAL_EVIDENCE")
        reviewed=source[gid]
        if utc(reviewed["kickoff_utc"])!=utc(forecast["kickoff_utc"]):
            raise ValueError("SOURCE_MATCHUP_MISMATCH")
        used=set()
        for _,team,pid,status in status_candidates(reviewed):
            if team in used:continue
            if team not in (forecast["home_team"],forecast["away_team"]):
                raise ValueError("EVIDENCE_WRONG_TEAM")
            if len(scenarios)>=max_total:
                blocked.append({"game_id":gid,"team":team,"reason":"SCENARIO_LIMIT"})
                continue
            try:
                roles=conditional_roles(data["roles"],team=team,missing_id=pid)
                candidate=simulate(forecast,data["drives"],roles,asof=web_at,
                                   draws=baseline["draws"],seed=baseline["seed"])
                expected=baseline["distribution"]
                output=candidate["distribution"]
                for metric in ("TOTAL","HOME_MARGIN",team+":points"):
                    if not math.isclose(expected[metric]["mean"],output[metric]["mean"],
                                        abs_tol=1e-9,rel_tol=0):
                        raise ValueError("SCENARIO_CHANGED_UNCALIBRATED_TEAM_SCORE")
            except (KeyError,ValueError,TypeError) as exc:
                blocked.append({"game_id":gid,"team":team,
                                "player_id":pid,"reason":str(exc)})
                continue
            def relevant(d):
                return {k:v["mean"] for k,v in d.items()
                        if k.startswith(team+":") and len(k.split(":"))==3}
            used.add(team)
            scenarios.append({
                "game_id":gid,"team":team,"player_id":pid,
                "reported_status":status,"source_timestamp_utc":web_at.isoformat(),
                "scenario":"IF_PLAYER_UNAVAILABLE_NOT_CONFIRMED",
                "baseline_total_mean":expected["TOTAL"]["mean"],
                "scenario_total_mean":output["TOTAL"]["mean"],
                "baseline_player_means":relevant(expected),
                "scenario_player_means":relevant(output),
                "probability_player_unavailable":None,
                "outcome_accuracy_validated":False,"wager_authorized":False})
    return {"spec":SPEC,"status":"WEB_CONDITIONAL_RESEARCH_ONLY",
            "original_capture_utc":original_at.isoformat(),
            "evidence_capture_utc":web_at.isoformat(),
            "scenarios":scenarios,"blocked":blocked,"count":len(scenarios),
            "score_coefficients_changed":False,"original_forecast_modified":False,
            "unvalidated_weather_effect_applied":False,
            "injury_absence_probability":None,"estimated_ev":None,
            "wager_authorized":False,"action":"PASS_RESEARCH_ONLY"}
