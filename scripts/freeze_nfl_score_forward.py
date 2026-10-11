"""Immutable first-seen NFL margin and total forecasts (research only).

Reads independent current-projection output, an nflverse schedule with original
kickoff fields and an append-only JSONL history. Uses no sportsbook odds.
A forecast already present for an event is never silently replaced. The
season/target source timestamp is NOT a documented historically frozen vintage.
Betting decision: ALWAYS PASS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from nfl.pregame import schedule_kickoff_utc

VERSION = "FIRST_SEEN_NFL_SCORE_V1"
MODEL_VERSION = "HARBIN_CANONICAL_SCORE_V4.9.1"
REQUIRED_FIELDS = ("game_id", "season", "week", "home_team", "away_team",
                   "baseline_home_margin", "baseline_total")


def _utc(s):
    if not isinstance(s,str):raise ValueError("timestamp must be ISO8601")
    dt=datetime.fromisoformat(s.replace("Z","+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("UTC offset required")
    return dt.astimezone(timezone.utc)


def _finite(value):
    if isinstance(value,bool):
        raise ValueError("boolean score")
    x=float(value)
    if not math.isfinite(x):
        raise ValueError("nonfinite score")
    return x


def _load_history(path):
    if not path.exists():return []
    out=[];seen=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get("event_id") in seen or not r.get("event_id"):
            raise ValueError("duplicate/invalid persisted event_id")
        if r.get("record_kind") != "FROZEN_FIRST_SEEN_FORECAST":
            raise ValueError("history contains incompatible record")
        seen.add(r["event_id"])
        out.append(r)
    return out


def freeze(projection, schedule_rows, existing, *, captured_utc, source_sha):
    now=_utc(captured_utc)
    if not source_sha or len(str(source_sha))<7:
        raise ValueError("real source Git SHA required")
    if not isinstance(existing,list):
        raise ValueError("history must be a list")
    keys=[r.get("event_id") for r in existing]
    if any(not x for x in keys) or len(keys)!=len(set(keys)):
        raise ValueError("existing forward history duplicated or malformed")
    for r in existing:
        if r.get("record_kind")!="FROZEN_FIRST_SEEN_FORECAST":
            raise ValueError("existing history invalid")
    mapped={}
    for game in schedule_rows:
        gid=str(game.get("game_id",""))
        if not gid or gid in mapped:
            raise ValueError("duplicate/missing schedule event ID")
        mapped[gid]=game
    report={
        "version":VERSION,
        "status":"BLOCKED" if projection.get("status")!="READY" else "RESEARCH_ONLY",
        "captured_utc":now.isoformat(),
        "decision":"PASS","edge_validated":False,
        "bookmaker_quotes_verified":False,"positive_ev_proven":False,
        "new_records":[],"skipped_existing":[],"skipped_not_pregame":[],"blockers":[],
    }
    if projection.get("status")!="READY":
        report["blockers"].append(str(projection.get("reason","projection not READY")))
        return existing,report
    games=projection.get("games")
    if not isinstance(games,list) or not games:
        raise ValueError("projection READY but no game records")
    game_ids=[str(g.get("game_id","")) for g in games]
    if not all(game_ids) or len(game_ids)!=len(set(game_ids)):
        raise ValueError("duplicate or blank projected event ID")
    existing_ids=set(keys)
    added=[]
    for p in games:
        if any(f not in p or p[f] is None for f in REQUIRED_FIELDS):
            raise ValueError("missing canonical score / identity")
        gid=str(p["game_id"])
        if gid in existing_ids:
            report["skipped_existing"].append(gid)
            continue
        sg=mapped.get(gid)
        if sg is None:
            raise ValueError("schedule missing target event "+gid)
        for field in ("season","week","home_team","away_team"):
            if str(p[field])!=str(sg.get(field)):
                raise ValueError("schedule/projection identity mismatch "+gid)
        if sg.get("game_type")!="REG" or (sg.get("home_score") is not None
                                           and sg.get("away_score") is not None):
            report["skipped_not_pregame"].append(gid)
            continue
        kickoff=schedule_kickoff_utc(sg)
        if kickoff is None:
            raise ValueError("kickoff unavailable for "+gid)
        if kickoff<=now:
            report["skipped_not_pregame"].append(gid)
            continue
        margin=_finite(p["baseline_home_margin"])
        total=_finite(p["baseline_total"])
        if total<=0 or abs(margin)>total:
            raise ValueError("physically impossible score prediction")
        row={
            "record_kind":"FROZEN_FIRST_SEEN_FORECAST",
            "schema_version":VERSION,
            "model_version":MODEL_VERSION,
            "event_id":gid,"league":"NFL","season":int(p["season"]),
            "week":int(p["week"]),
            "home_team":str(p["home_team"]),"away_team":str(p["away_team"]),
            "kickoff_utc":kickoff.isoformat(),
            "captured_utc":now.isoformat(),
            "source_repository_sha":source_sha,
            "independent_projected_home_margin":margin,
            "independent_projected_total":total,
            "independent_home_points":(total+margin)/2,
            "independent_away_points":(total-margin)/2,
            "home_qb_proxy":p.get("home_qb_proxy_name"),
            "away_qb_proxy":p.get("away_qb_proxy_name"),
            "qb_confirmed_at_capture":False,
            "roster_weather_confirmed_at_capture":False,
            "odds_verified":False,
            "p_win":None,"p_push":None,"p_loss":None,
            "estimated_ev":None,"decision":"PASS","bet_executed":False,
            "upstream_schedule_vintage": "LATEST_AVAILABLE_AT_CAPTURE_NOT_PROVEN_ORIGINAL_SOURCE_VERSION",
            "note":"First-seen football projection, not validated betting edge or verified market quote",
        }
        row["content_sha256"]=hashlib.sha256(
            json.dumps(row,sort_keys=True,separators=(",",":")).encode()
        ).hexdigest()
        added.append(row)
    report["new_records"]=[r["event_id"] for r in added]
    return existing+added,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection",required=True,type=Path)
    p.add_argument("--season",required=True,type=int)
    p.add_argument("--history",default="history/score_forward_2026.jsonl",type=Path)
    p.add_argument("--summary",default="reports/score_forward_snapshot.json",type=Path)
    p.add_argument("--source-sha",default=os.environ.get("GITHUB_SHA",""))
    args=p.parse_args()
    from nfl.data import NFLDataClient
    source=json.loads(args.projection.read_text(encoding="utf8"))
    historical=_load_history(args.history)
    season=NFLDataClient().load_schedules([args.season])
    updated,report=freeze(source,season.iter_rows(named=True),historical,
                          captured_utc=datetime.now(timezone.utc).isoformat(),
                          source_sha=args.source_sha)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    if len(updated)>len(historical):
        args.history.parent.mkdir(parents=True,exist_ok=True)
        args.history.write_text(
            "".join(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n"
                    for record in updated),encoding="utf8"
        )
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
