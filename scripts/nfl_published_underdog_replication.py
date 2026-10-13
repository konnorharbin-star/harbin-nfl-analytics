"""External-published NFL short-underdog moneyline hypothesis replication.

Primary published hypothesis (Journal of Prediction Markets, 2026):
  * underdog moneyline when game spread <= 3
  * underdog moneyline when 3 < spread <= 7

Only predeclared categories are reported. This uses undated archived nflverse
schedule prices for BOTH sides with actual results, not original timestamped
accepted sportsbook offers; it is a RETROSPECTIVE replication only.
"""
from __future__ import annotations
from collections import defaultdict
from math import isfinite
from pathlib import Path
from urllib.request import Request, urlopen
import csv, io, json, random

SPEC="NFL_PUBLISHED_UNDERDOG_MARKET_REPLICATION_V1"
SOURCE_COMMIT="fc78300c9d7af11de25e8cd2c224485ccfe34f95"
SOURCE_URL="https://raw.githubusercontent.com/nflverse/nfldata/"+SOURCE_COMMIT+"/data/games.csv"
PAPER="https://doi.org/10.5750/jpm.v19i2.2325"
YEARS=(2022,2023,2024,2025)
GROUPS=("spread_0_to_3","spread_gt_3_to_7","spread_gt_7")
MIN_WEEKS=8

def _num(value):
    if value in (None,"") or isinstance(value,bool):
        return None
    try:x=float(value)
    except (ValueError,TypeError):return None
    return x if isfinite(x) else None

def _group(spread):
    if spread<=3:return "spread_0_to_3"
    if spread<=7:return "spread_gt_3_to_7"
    return "spread_gt_7"

def _unit_return(american,won,push):
    if push:return 0.
    if not won:return -1.
    return american/100 if american>0 else 100/(-american)

def _wilson(wins,decided,z=1.95996398454):
    if decided==0:return None
    p=wins/decided
    d=1+z*z/decided
    mid=(p+z*z/(2*decided))/d
    radius=z*((p*(1-p)/decided+z*z/(4*decided*decided))**.5)/d
    return [max(0.,mid-radius),min(1.,mid+radius)]

def _summary(group):
    n=len(group)
    if n==0:return {"games":0,"roi":None,"net_units":0,"win_rate":None}
    win=sum(a["won"] for a in group)
    push=sum(a["push"] for a in group)
    units=sum(a["units"] for a in group)
    return {
        "games":n,"wins":win,"losses":n-win-push,"pushes":push,
        "net_hypothetical_units":units,
        "hypothetical_unit_roi":units/n,
        "win_rate_decided":win/(n-push) if n>push else None,
        "nominal_wilson_win_interval_95":_wilson(win,n-push),
        "accepted_bookmaker_bets":0,
    }

def analyze(rows):
    if not isinstance(rows,list):raise ValueError("ROWS_MUST_BE_LIST")
    used=set();records=[];rejects=defaultdict(int)
    for r in rows:
        if r.get("game_type")!="REG" or str(r.get("season")) not in map(str,YEARS):
            continue
        gid=r.get("game_id")
        if not isinstance(gid,str) or not gid:raise ValueError("GAME_ID_REQUIRED")
        if gid in used:raise ValueError("DUPLICATE_NFL_GAME")
        used.add(gid)
        hscore=_num(r.get("home_score"))
        ascore=_num(r.get("away_score"))
        spread=_num(r.get("spread_line"))
        hm=_num(r.get("home_moneyline"))
        am=_num(r.get("away_moneyline"))
        try:week=int(r["week"])
        except (TypeError,ValueError) as e:raise ValueError("INVALID_WEEK") from e
        if (hscore is None or ascore is None or spread is None
            or hm is None or am is None or spread==0
            or not 1<=week<=18):
            rejects["missing_or_pickem"]+=1
            continue
        if (abs(hm)<100 or abs(am)<100 or abs(hm)>10000 or abs(am)>10000
            or not hm.is_integer() or not am.is_integer()):
            rejects["invalid_moneyline"]+=1
            continue
        # nflverse spread_line sign: positive means HOME favorite.
        # Safeguard: if underdog price is less favorable than favorite,
        # flag market-side inconsistency rather than count a synthetic win.
        dog="away" if spread>0 else "home"
        dog_odds=am if dog=="away" else hm
        favorite_odds=hm if dog=="away" else am
        if dog_odds<=favorite_odds:
            rejects["spread_vs_moneyline_orientation_conflict"]+=1
            continue
        won=(ascore>hscore) if dog=="away" else (hscore>ascore)
        push=hscore==ascore
        if push:won=False
        records.append({
            "year":int(r["season"]),"week":week,"game_id":gid,
            "underdog_side":dog,"closing_spread_magnitude":abs(spread),
            "moneyline":int(dog_odds),"won":int(won),"push":int(push),
            "units":_unit_return(int(dog_odds),won,push),
        })
    results={}
    for year in YEARS:
        group=[a for a in records if a["year"]==year]
        results[str(year)]={
            "all_underdogs":_summary(group),
            **{k:_summary([a for a in group if _group(a["closing_spread_magnitude"])==k])
               for k in GROUPS},
        }
    # An independent publication predates the 2025 grade, but this source's
    # single updatable snapshot is NOT a preserved as-of bookmaker archive.
    return {
        "spec":SPEC,"status":"PUBLISHED_HYPOTHESIS_NOT_REPLICATED_AS_PROVEN_EDGE",
        "source_data":SOURCE_URL,
        "source_commit":SOURCE_COMMIT,
        "external_paper":PAPER,
        "source_price_stage":"NFLVERSE_GAMES_SCHEDULE_UNDATED_PRICE",
        "source_quote_origin_verified":False,
        "books_and_actual_fills_verified":False,
        "all_underdog_sides_included_not_model_selected":True,
        "study_publication_date":"2026-02-25",
        "years":results,
        "exclusions":dict(rejects),
        "distinct_games_included":len(records),
        "expected_return_or_profit_in_future":None,
        "independent_future_validation":False,
        "betting_authorized":False,
        "action":"PASS_RESEARCH_ONLY",
    }

def run(source=SOURCE_URL,out="reports/published_underdog_replication.json"):
    if source!=SOURCE_URL:
        raise ValueError("SOURCE_MUST_BE_PINNED_NFLVERSE_COMMIT")
    request=Request(source,headers={"User-Agent":"HarbinSportsResearch/1.0"})
    with urlopen(request,timeout=30) as handle:
        content=handle.read()
    records=list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    results=analyze(records)
    results["rows_loaded"]=len(records)
    dest=Path(out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(results,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return results

if __name__=="__main__":
    result=run()
    print(json.dumps({
        "status":result["status"],
        "close_spread_underdog_roi":{
            year:round(row["spread_0_to_3"]["hypothetical_unit_roi"],5)
            for year,row in result["years"].items()},
        "no_verified_bets":True,
    },sort_keys=True))
