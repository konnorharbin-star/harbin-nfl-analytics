"""Time-bounded NFL weekly injury report evidence by GSIS player ID."""

from copy import deepcopy
from datetime import datetime, UTC

VERSION = 'NFL_DATED_GSIS_INJURY_REPORT_RESEARCH_V1'

ALLOWED={"out","doubtful","questionable"}

def _utc(value):
    if isinstance(value,str):
        try:value=datetime.fromisoformat(value.replace("Z","+00:00"))
        except ValueError as err:raise ValueError("INVALID_DATE_MODIFIED") from err
    if not isinstance(value,datetime) or value.tzinfo is None:
        raise ValueError("DATE_MODIFIED_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)

def add_injury_reports(review,rows,*,season,week,captured):
    """Secondary source evidence, not official confirmed game-day inactive."""
    at=_utc(captured)
    if review.get("status")!="SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY":
        raise ValueError("ID_LINKED_RESEARCH_REQUIRED")
    if not isinstance(rows,list) or not isinstance(season,int) or not isinstance(week,int):
        raise ValueError("INVALID_INJURY_REPORT_INPUT")
    found={}
    conflicts=set()
    stats={"not_matching_week":0,"invalid_timestamp":0,"no_game_status":0}
    for row in rows:
        if not isinstance(row,dict):continue
        if (str(row.get("season"))!=str(season) or
            str(row.get("week"))!=str(week) or
            str(row.get("game_type") or row.get("season_type") or "").upper()!="REG"):
            stats["not_matching_week"]+=1
            continue
        key=(row.get("team"),row.get("gsis_id"))
        if not all(isinstance(v,str) and v for v in key):continue
        try:modified=_utc(row.get("date_modified"))
        except ValueError:
            stats["invalid_timestamp"]+=1
            continue
        if modified>at:
            stats["invalid_timestamp"]+=1
            continue
        status=str(row.get("report_status") or "").strip().lower()
        if status not in ALLOWED:
            stats["no_game_status"]+=1
            continue
        prior=found.get(key)
        if prior and prior["modified"]==modified and prior["status"]!=status:
            conflicts.add(key)
        if prior is None or modified>prior["modified"]:
            found[key]={"modified":modified,"status":status}
