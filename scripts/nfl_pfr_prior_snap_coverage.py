"""Pregame NFL player snap-history coverage from free PFR/nflverse records.

PFR snaps are updated during the season. The detailed FTN participation
file is not. Join only via unique official PFR-ID <-> GSIS-ID crosswalk,
and use games strictly before the target week. Missing player row is
UNKNOWN, never a proven zero, DNP or current-game participation.
Existing model probability history remains unchanged.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import nflreadpy as nfl
from nfl.data import NFLDataClient

VERSION="PFR_PRIOR_OFFENSIVE_SNAP_EVIDENCE_V1"

def as_utc(value):
    dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if dt.tzinfo is None or dt.utcoffset()!=timedelta(0):
        raise ValueError("UTC timestamp required")
    return dt.astimezone(timezone.utc)

def unique_crosswalk(rows):
    gsis=defaultdict(set);pfr=defaultdict(set)
    for x in rows:
        a=x.get("gsis_id");b=x.get("pfr_id")
        if not a or not b:continue
        gsis[str(a)].add(str(b));pfr[str(b)].add(str(a))
    return {a:next(iter(ids)) for a,ids in gsis.items()
            if len(ids)==1 and len(pfr[next(iter(ids))])==1}

def preceding_games(schedule,season,week,team):
    games=[g for g in schedule
        if g.get("game_type")=="REG" and int(g["season"])==season
        and int(g["week"])<week
        and team in (g["home_team"],g["away_team"])
        and g.get("home_score") is not None and g.get("away_score") is not None]
    return sorted(games,key=lambda g:(int(g["week"]),str(g["game_id"])))[-4:]

def coverage(full,players,snaps,schedule,at):
    at=as_utc(at)
    mapping=unique_crosswalk(players)
    index={}
    by_team=defaultdict(int)
    for x in snaps:
        if x.get("game_type")!="REG":continue
        gid=str(x.get("game_id") or "")
        pf=str(x.get("pfr_player_id") or "")
        tm=str(x.get("team") or "")
        if not gid or not pf or not tm:continue
        key=(gid,tm,pf)
        if key in index:raise ValueError("ambiguous PFR player-game snap record")
        n=x.get("offense_snaps")
        value=float(n) if n is not None else None
        if value is not None and (value<0 or value>150):
            raise ValueError("invalid historical offensive snap count")
        index[key]=value
        by_team[(gid,tm)]+=1
    result=[]
    seen=set()
    for r in full:
        key=(r["game_id"],r["player_id"],r["market"])
        if key in seen:raise ValueError("duplicate original full player-stat group")
        seen.add(key)
        if at>=as_utc(r["kickoff_utc"]):continue
        if at<as_utc(r["first_observed_at_utc"]):
            raise ValueError("source capture cannot predate full-distribution freeze")
        pfr_id=mapping.get(str(r["player_id"]))
        games=preceding_games(schedule,int(r["season"]),int(r["week"]),r["team"])
        available=[g for g in games if by_team[(str(g["game_id"]),str(r["team"]))]>0]
        observed=[index[(str(g["game_id"]),str(r["team"]),pfr_id)]
            for g in available if pfr_id is not None and
            index.get((str(g["game_id"]),str(r["team"]),pfr_id)) is not None]
        status=("MISSING_UNIQUE_PFR_GSIS_CROSSWALK" if pfr_id is None
            else "INSUFFICIENT_OFFICIAL_TEAM_SNAP_COVERAGE" if len(available)<2
            else "PLAYER_SNAP_HISTORY_INSUFFICIENT" if len(observed)<2
            else "HISTORICAL_OFFENSIVE_SNAPS_OBSERVED")
        result.append({
            "version":VERSION,"observed_at_utc":at.isoformat(),
            "source":"NFLVERSE_PFR_SNAP_COUNTS","game_id":r["game_id"],
            "season":r["season"],"week":r["week"],"kickoff_utc":r["kickoff_utc"],
            "player_id":r["player_id"],"pfr_id":pfr_id,"market":r["market"],
            "prior_team_games":len(games),"source_covered_games":len(available),
            "player_snap_rows_observed":len(observed),
            "observed_offense_snaps":observed,
            "observed_mean":round(sum(observed)/len(observed),3) if observed else None,
            "status":status,"missing_as_zero":False,
            "future_player_availability_verified":False,
            "future_player_offensive_snap_share_verified":False,
            "forecast_active_probability":None,
            "decision":"RESEARCH_ONLY",
        })
    return result

def append_first_seen(rows,path):
    path=Path(path)
    old=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    def key(r):return (r["game_id"],r["player_id"],r["market"])
    keys={key(x) for x in old}
    if len(keys)!=len(old):raise ValueError("duplicate original coverage")
    created=[x for x in rows if key(x) not in keys]
    if created:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a") as stream:
            for x in created:stream.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
    return {"existing":len(old),"created":len(created),"total":len(old)+len(created)}

def main():
    at=datetime.now(timezone.utc).isoformat()
    full_path=Path("history/player_prop_full_distributions_2026.jsonl")
    full=[json.loads(x) for x in full_path.read_text().splitlines() if x.strip()]
    schedule=NFLDataClient().load_schedules([2026],refresh=True).to_dicts()
    snaps=nfl.load_snap_counts([2026]).to_dicts()
    players=nfl.load_players().to_dicts()
    entries=coverage(full,players,snaps,schedule,at)
    written=append_first_seen(entries,Path("history/player_prop_prior_snap_coverage_2026.jsonl"))
    count=defaultdict(int)
    for x in entries:count[x["status"]]+=1
    report={"version":VERSION,"source_observed_utc":at,"decision":"RESEARCH_ONLY",
        "prior_snap_source_rows":len(snaps),
        "original_player_stats":len(full),"eligible_future_context_rows":len(entries),
        "status_counts":dict(count),"first_seen":written,
        "original_probability_sources_edited":False,
        "current_activity_confirmations":0,"market_quote_records":0}
    p=Path("reports/player_prop_prior_snap_coverage.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
