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
