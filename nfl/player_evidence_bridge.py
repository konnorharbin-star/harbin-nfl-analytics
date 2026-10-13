"""Exact-ID linked player availability context; no starter or injury effect inference.

Join original *frozen* game player prop IDs (nflverse GSIS) to the nflverse
players crosswalk, then join ESPN injury entries by ESPN's athlete ID and exact
team. Never join merely by player name or treat not-listed as healthy. This
report is a later supplemental web observation, NOT a retroactively changed
first-seen forecast.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import UTC,datetime
from math import isfinite
import re

VERSION="NFL_EXACT_ID_PLAYER_AVAILABILITY_REVIEW_V1"
DIGITS=re.compile(r"^[0-9]+$")

def _clock(s):
    if not isinstance(s,str):raise ValueError("MISSING_UTC_TIME")
    try:d=datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError as e:raise ValueError("BAD_UTC_TIME") from e
    if d.tzinfo is None:raise ValueError("UNZONED_TIMESTAMP")
    return d.astimezone(UTC)

def _espn_id(value):
    if isinstance(value,bool) or value is None:return None
    if isinstance(value,int):return str(value) if value>0 else None
    if isinstance(value,str) and DIGITS.fullmatch(value.strip()):
        return str(int(value)) if int(value)>0 else None
    return None

def _crosswalk(rows):
    if not isinstance(rows,list):raise ValueError("PLAYER_DIRECTORY_MISSING")
    by_gsis={}
    duplicates=set()
    espn_to_gsis=defaultdict(set)
    for row in rows:
        if not isinstance(row,dict):continue
        gsis=row.get("gsis_id")
        if not isinstance(gsis,str) or not gsis:continue
        eid=_espn_id(row.get("espn_id"))
        if gsis in by_gsis:
            duplicates.add(gsis)
        by_gsis[gsis]={
            "gsis_id":gsis,"espn_id":eid,
            "display_name":str(row.get("display_name") or row.get("full_name") or "")[:100],
            "position":str(row.get("position") or "")[:10],
        }
        if eid:espn_to_gsis[eid].add(gsis)
    for gsis in duplicates:
        by_gsis.pop(gsis,None)
    ambiguous_espn={eid for eid,ids in espn_to_gsis.items() if len(ids)>1}
    return by_gsis,ambiguous_espn,duplicates

def review_player_evidence(scan, context, players, *, at):
    if at.tzinfo is None:raise ValueError("UNZONED_REVIEW_TIME")
    now=at.astimezone(UTC)
    if scan.get("spec")!="NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1":
        raise ValueError("ORIGINAL_FROZEN_SCAN_REQUIRED")
    if context.get("version")!="NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1":
        raise ValueError("INVALID_PUBLIC_CONTEXT_SOURCE")
    original_captured=_clock(scan.get("capture_utc"))
    context_captured=_clock(context.get("captured_utc"))
    if not original_captured<=context_captured<=now:
        raise ValueError("SOURCE_CAPTURE_CHRONOLOGY_INVALID")
    crosswalk,ambiguous_espn,duplicate_ids=_crosswalk(players)
    context_map={}
    for x in context.get("games",[]):
        gid=x.get("game_id")
        if not gid or gid in context_map:
            raise ValueError("DUPLICATE_OR_MISSING_CONTEXT_GAME")
        context_map[gid]=x
    output=[]
    for game in scan.get("games",[]):
        if game.get("status")!="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH":
            continue
        gid=game.get("game_id")
        source=context_map.get(gid)
        if source is None:
            raise ValueError("MISSING_CONTEXT_FOR_RESEARCH_GAME")
        home,away=game.get("frozen_football_score_input",{}).get("home_team"),game.get("frozen_football_score_input",{}).get("away_team")
        frozen=game.get("frozen_football_score_input",{})
        kick=_clock(frozen["kickoff_utc"])
        first=_clock(frozen["captured_utc"])
        if not (first<=original_captured<=context_captured<kick
                and source.get("home_team")==home and source.get("away_team")==away
                and _clock(source["kickoff_utc"])==kick):
            raise ValueError("CONFLICTING_OR_POSTKICKOFF_CONTEXT")
        snap=source.get("structured_injury_snapshot") or {}
        source_at=_clock(snap.get("captured_utc"))
        if source_at!=context_captured:raise ValueError("INJURY_CAPTURE_MISMATCH")
        by_team={}
        for team in (home,away):
            injuries=snap.get(team)
            if injuries is not None and not isinstance(injuries,list):
                raise ValueError("INVALID_INJURY_REPORT")
            idx=defaultdict(list)
            if injuries is not None:
                for injury in injuries:
                    if not isinstance(injury,dict):continue
                    eid=_espn_id(injury.get("espn_athlete_id"))
                    if eid:idx[eid].append(injury)
            candidates={}
            for metric,desc in game.get("distribution",{}).items():
                parts=metric.split(":")
                if len(parts)!=3 or parts[0]!=team or parts[1]=="OTHER":
                    continue
                gsis=parts[1]
                entry=candidates.setdefault(gsis,{"stats":set(),"qb_attempts_mean":None})
                entry["stats"].add(parts[2])
                if parts[2]=="passing_attempts" and isinstance(desc,dict):
                    mean=desc.get("mean")
                    if isinstance(mean,(float,int)) and not isinstance(mean,bool) and isfinite(mean) and mean>=0:
                        entry["qb_attempts_mean"]=float(mean)
            rows=[]
            for gsis,meta in sorted(candidates.items()):
                mapped=crosswalk.get(gsis)
                status="NO_UNIQUE_GSIS_CROSSWALK"
                reported=None
                if gsis in duplicate_ids:
                    status="DUPLICATED_GSIS_DIRECTORY_ID"
                elif mapped is not None:
                    eid=mapped["espn_id"]
                    if injuries is None:
                        status="INJURY_SOURCE_UNAVAILABLE"
                    elif not eid:
                        status="NO_ESPN_ID_CROSSWALK"
                    elif eid in ambiguous_espn:
                        status="AMBIGUOUS_ESPN_ID_CROSSWALK"
                    elif len(idx.get(eid,[]))>1:
                        status="AMBIGUOUS_INJURY_RECORDS"
                    elif len(idx.get(eid,[]))==1:
                        status="ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS"
                        reported=idx[eid][0].get("reported_status")
                    else:
                        status="NOT_LISTED_IN_SNAPSHOT_NOT_CONFIRMED_HEALTHY"
                rows.append({
                    "player_id":gsis,"team":team,
                    "name_from_nflverse":mapped["display_name"] if mapped else None,
                    "position_from_nflverse":mapped["position"] if mapped else None,
                    "espn_athlete_id":mapped["espn_id"] if mapped else None,
                    "research_stats":sorted(meta["stats"]),
                    "historical_qb_attempts_proxy_mean":meta["qb_attempts_mean"],
                    "availability_evidence_status":status,
                    "espn_reported_status":reported,
                    "verified_active":False,
                    "verified_starting_qb":False,
                    "player_probabilities_adjusted":False,
                })
            qb_candidates=sorted(
                (row for row in rows if row["historical_qb_attempts_proxy_mean"] is not None),
                key=lambda r:-r["historical_qb_attempts_proxy_mean"],
            )
            by_team[team]={
                "players":rows,
                "historical_qb_volume_candidates":[
                    {"player_id":r["player_id"],"name":r["name_from_nflverse"],
                     "passing_attempts_shadow_mean":r["historical_qb_attempts_proxy_mean"],
                     "source_is_not_confirmation_of_starter":True}
                    for r in qb_candidates],
                "reported_injury_count":len(injuries) if injuries is not None else None,
                "none_listed_does_not_prove_availability":True,
            }
        output.append({
            "game_id":gid,"kickoff_utc":kick.isoformat(),
            "original_forecast_capture_utc":first.isoformat(),
            "teams":by_team,"context_is_supplemental":True,
            "starter_verified":False,"final_inactive_list_verified":False,
            "betting_authorized":False,
        })
    return {"version":VERSION,"status":"SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY",
            "captured_utc":now.isoformat(),
            "context_source_captured_utc":context_captured.isoformat(),
            "original_football_forecast_capture_utc":original_captured.isoformat(),
            "games":output,"model_fair_score_or_prop_adjusted":False,
            "official_week_specific_inactives_verified":False,
            "injury_feed_player_match_mode":"EXACT_ESPN_ID_CROSSWALK_ONLY",
            "actual_book_price":None,"estimated_ev":None,
            "wager_authorized":False}
