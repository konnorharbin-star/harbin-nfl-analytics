"""Free nflverse -> football-only simultaneous score and props shadow scan.

No bookmaker, no paid API, no wager, no historical rewriting. Needs internet.
All outputs are separately captured research artifacts. Mature previously played
games provide training; current/future games receive first-seen forecasts.
"""
from __future__ import annotations
import argparse
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import nflreadpy as nfl
from nfl.coupled_nflverse_asof import kickoff_utc,prepare_asof_source
from nfl.coupled_drive_player_shadow import simulate_joint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--week",type=int)
    p.add_argument("--game-id",default="ALL")
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--seed",type=int,default=20261013)
    p.add_argument("--out",type=Path,default=Path("reports/coupled_drive_player_shadow.json"))
    a=p.parse_args()
    seasons=[a.season-1,a.season]
    # Source captured once and dated; no inference from sportsbook prices.
    schedules=nfl.load_schedules(seasons)
    pbp=nfl.load_pbp(seasons)
    stats=nfl.load_player_stats(seasons)
    captured=datetime.now(UTC)
    all_rows=schedules.to_dicts()
    if a.week is None:
        future=[]
        for r in all_rows:
            if r.get("season")!=a.season or r.get("game_type")!="REG":continue
            try: k=kickoff_utc(r)
            except ValueError:continue
            if k>captured:future.append((k,int(r["week"])))
        a.week=min(future)[1] if future else -1
    targets=[]
    for r in all_rows:
        if r.get("season")!=a.season or r.get("week")!=a.week or r.get("game_type")!="REG":
            continue
        if a.game_id!="ALL" and r.get("game_id")!=a.game_id:continue
        try: kickoff=kickoff_utc(r)
        except ValueError:continue
        if captured<kickoff:
            targets.append((kickoff,str(r["game_id"])))
    results=[]
    for kickoff,gid in sorted(targets):
        try:
            forecast,drives,roles,provenance=prepare_asof_source(
                schedules,pbp,stats,season=a.season,week=a.week,game_id=gid,at=captured)
            # All proportions/models are conditional, uncalibrated, football-only.
            result=simulate_joint(forecast,drives,roles,asof=captured,
                                  draws=a.draws,seed=a.seed)
            result["input_provenance"]=provenance
            result["frozen_football_score_input"]=forecast
            result["source_capture"]="SNAPSHOT_READ_ONLY_NOT_COMMITTED_AS_IMMUTABLE_FORWARDED_PROOF"
            results.append(result)
        except (ValueError,RuntimeError) as e:
            results.append({"game_id":gid,"status":"SOURCE_OR_MODEL_BLOCKED",
                            "reason":str(e),"estimated_ev":None,
                            "wager_authorized":False,"action":"PASS_RESEARCH_ONLY"})
    output={"spec":"NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1",
            "capture_utc":captured.isoformat(),
            "source":"free_nflreadpy_nflverse",
            "season":a.season,"week":a.week,"requested_game":a.game_id,
            "status":"RESEARCH_ONLY_NO_VALIDATED_EDGE",
            "games":results,"approved_wagers":0,
            "original_preexisting_forward_archive_unchanged":True,
            "first_seen_evidence_not_persisted_in_immutable_ledger":True}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(output,sort_keys=True,indent=2)+"\n"
    a.out.write_text(data,encoding="utf-8")
    print(json.dumps({"status":output["status"],"asof":output["capture_utc"],
                      "games_seen":len(results),
                      "research_generated":sum(r.get("status")=="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH" for r in results),
                      "blocked":sum(r.get("status")=="SOURCE_OR_MODEL_BLOCKED" for r in results),
                      "artifact":str(a.out),
                      "artifact_sha256":hashlib.sha256(data.encode()).hexdigest()}))
if __name__=="__main__":
    main()
