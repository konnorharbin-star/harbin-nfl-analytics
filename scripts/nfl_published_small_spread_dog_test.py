"""Predeclared small-spread NFL underdog moneyline hypothesis, research only.

Krieger & Shank (2026): underdog ML when spread <= 3 showed historical gains.
This local replication uses one named-book *aggregator-labeled* source pair per
game, with both spread and moneyline in same snapshot, no line/price shopping.

Collector capture time != actual bookmaker offer timestamp. No betting claims.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
import json, random

PAPER="https://doi.org/10.5750/jpm.v19i2.2325"
SPEC="NFL_SMALL_SPREAD_DOG_PUBLISHED_HYPOTHESIS_V1"
BOOK="BetMGM"
PROVIDER="action_network"
SPREAD_MAX=3.0
MINUTES_BEFORE_KICKOFF=60
MATURE_HOURS=76
BOOT_SEED=20261013

def utc(v):
    if not isinstance(v,str) or not v:raise ValueError("MISSING_SOURCE_TIME")
    x=datetime.fromisoformat(v.replace("Z","+00:00"))
    if x.tzinfo is None:raise ValueError("NAIVE_SOURCE_TIME")
    return x.astimezone(timezone.utc)

def numeric(value):
    if isinstance(value,bool):raise ValueError("NUMERIC_BOOL")
    try:x=float(value)
    except (TypeError,ValueError) as e:raise ValueError("INVALID_NUMERIC") from e
    if not isfinite(x):raise ValueError("NONFINITE_NUMERIC")
    return x

def sides(row,field):
    if {row.get("first_side"),row.get("second_side")}!={"home","away"}:
        raise ValueError("INVALID_MARKET_SIDE_IDENTITIES")
    return {row["first_side"]:numeric(row["first_"+field]),
            row["second_side"]:numeric(row["second_"+field])}

def paired_quotes(raw, *, season=2026, week=4, at):
    at=utc(at) if isinstance(at,str) else at.astimezone(timezone.utc)
    bypair=defaultdict(dict)
    games=set()
    skipped=defaultdict(int)
    for row in raw:
        if (row.get("season")!=str(season) or row.get("week")!=str(week)
            or row.get("book")!=BOOK or row.get("provider")!=PROVIDER):
            continue
        if row.get("market_type") not in ("moneyline","spread"):continue
        gid=row.get("game_id");games.add(gid)
        if not gid or not row.get("kickoff"):
            skipped["missing_identity"]+=1
            continue
        kickoff=utc(row["kickoff"]);capture=utc(row["captured_at"])
        if capture>at or capture>kickoff-timedelta(minutes=MINUTES_BEFORE_KICKOFF):
            skipped["after_research_or_kickoff_cutoff"]+=1
            continue
        key=(gid,row["book"],row["provider"],row["captured_at"])
        market=row["market_type"]
        if market in bypair[key]:
            skipped["duplicate_same_snapshot_market"]+=1
            bypair[key]["DUPLICATE"]=True
        bypair[key][market]=row
    bygame=defaultdict(list)
    for (gid,book,provider,when),m in bypair.items():
        if "DUPLICATE" in m or "moneyline" not in m or "spread" not in m:continue
        bygame[gid].append((utc(when),m))
    observations=[]
    for gid,entries in sorted(bygame.items()):
        capture,m=max(entries,key=lambda x:x[0])
        ml,sp=m["moneyline"],m["spread"]
        if (ml["kickoff"]!=sp["kickoff"] or
            ml["home_team"]!=sp["home_team"] or
            ml["away_team"]!=sp["away_team"]):
            skipped["mismatched_game_identity"]+=1
            continue
        odds=sides(ml,"american_odds")
        points=sides(sp,"line")
        # Require a clear two-sided +dog/-favorite at the exact same book.
        dogs=[side for side in ("home","away") if odds[side]>0 and odds[
            "away" if side=="home" else "home"]<0]
        if len(dogs)!=1:
            skipped["no_clear_positive_moneyline_underdog"]+=1
            continue
        dog=dogs[0]
        line=points[dog]
        if line<=0 or abs(line+points["away" if dog=="home" else "home"])>1e-9:
            skipped["inconsistent_spread_vs_moneyline"]+=1
            continue
        observations.append({
            "game_id":gid,"season":season,"week":week,
            "away_team":ml["away_team"],"home_team":ml["home_team"],
            "kickoff_utc":utc(ml["kickoff"]).isoformat(),
            "collector_capture_utc":capture.isoformat(),
            "book_label":book,"provider":provider,
            "underdog_side":dog,
            "underdog_team":ml["home_team"] if dog=="home" else ml["away_team"],
            "dog_american_moneyline":int(odds[dog]),
            "dog_point_spread":line,
            "qualifies_paper_le_3":line<=SPREAD_MAX,
            "qualifies_paper_gt_3_le_7":3<line<=7,
            "price_origin_and_account_execution_verified":False,
            "actual_game_result":None,
            "hypothetical_net_units":None,
            "decision":"RESEARCH_ONLY",
        })
    return observations,{"games_with_source_rows":len(games),
                         "games_with_valid_paired_quotes":len(observations),
                         "source_exclusions":dict(skipped)}

def wilson(wins,n):
    if n==0:return None
    z=1.959963984540054
    p=wins/n
    half=z*((p*(1-p)+z*z/(4*n))/n)**.5/(1+z*z/n)
    mid=(p+z*z/(2*n))/(1+z*z/n)
    return [max(0,mid-half),min(1,mid+half)]

def summarize(grades):
    n=len(grades)
    if not n:return {"graded_games":0,"win_rate":None,"roi":None,
                     "bootstrap_95_roi":None,"win_rate_wilson95":None}
    wins=sum(x["actual_game_result"]=="win" for x in grades)
    roi=sum(x["hypothetical_net_units"] for x in grades)/n
    rng=random.Random(BOOT_SEED)
    vals=[x["hypothetical_net_units"] for x in grades]
    boot=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n
                for unused in range(5000))
    return {"graded_games":n,"wins":wins,"losses":n-wins,
            "win_rate":wins/n,"win_rate_wilson95":wilson(wins,n),
            "hypothetical_flat_stake_net_units":sum(vals),
            "hypothetical_flat_stake_roi":roi,
            "bootstrap_95_roi":[boot[125],boot[4874]],
            "independent_accepted_bookmaker_fills":0,
            "sample_sufficient_for_proven_edge":False}

def run(source,completed,*,season=2026,week=4,at):
    now=utc(at) if isinstance(at,str) else at.astimezone(timezone.utc)
    quotes,diag=paired_quotes(source,season=season,week=week,at=now)
    final_map={r["game_id"]:r for r in completed
               if str(r.get("season"))==str(season) and
               str(r.get("week"))==str(week) and r.get("game_type")=="REG"}
    graded=[];pending=[]
    for obs in quotes:
        f=final_map.get(obs["game_id"])
        # Do not backfill a known result into what was originally a
        # pregame quote; a researcher running now labels this historical.
        if now<utc(obs["kickoff_utc"])+timedelta(hours=MATURE_HOURS):
            pending.append(obs)
            continue
        if (not f or f.get("home_score") is None or f.get("away_score") is None or
            f.get("home_team")!=obs["home_team"] or
            f.get("away_team")!=obs["away_team"]):
            pending.append(obs)
            continue
        home,away=int(f["home_score"]),int(f["away_score"])
        if home==away: # moneyline tie settlement varies; no assumed loss.
            pending.append(obs)
            continue
        won=(home>away)==(obs["underdog_side"]=="home")
        done={**obs,"actual_game_result":"win" if won else "loss",
              "hypothetical_net_units":obs["dog_american_moneyline"]/100 if won else -1}
        graded.append(done)
    small=[g for g in graded if g["qualifies_paper_le_3"]]
    medium=[g for g in graded if g["qualifies_paper_gt_3_le_7"]]
    return {
        "spec":SPEC,"hypothesis_source":PAPER,
        "date_research_ran_utc":now.isoformat(),
        "source_price_data":"ACTION_NETWORK_AGGREGATED_BETMGM_LABEL_NOT_OFFER",
        "betting_contract":"NFL_REGULAR_SEASON_FULL_GAME_MONEYLINE_INCL_OT",
        "market_venue_and_account_executable_verified":False,
        "selection_rule":"fixed_book_last_matched_pair_at_least_60min_pre_kickoff",
        "prespecified_primary_spread":"0 < underdog points <= 3",
        "retrospective_analysis_not_original_pregame_strategy_decision":True,
        "proven_profitable_edge":False,"betting_authorized":False,
        "input_diagnostic":diag,
        "primary_small_spread":summarize(small),
        "secondary_three_to_seven":summarize(medium),
        "all_mature_valid_pairs":summarize(graded),
        "individual_matured_observations":graded,
        "pending_source_maturity":pending,
        "action":"PASS_RESEARCH_ONLY",
    }

def main():
    import nflreadpy as nfl
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=Path("history/market_snapshots.csv"))
    p.add_argument("--out",type=Path,default=Path("reports/nfl_published_small_dog_2026w4_diagnostic.json"))
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--week",type=int,default=4)
    opt=p.parse_args()
    with opt.source.open(newline="",encoding="utf8") as fh:source=list(csv.DictReader(fh))
    schedule=nfl.load_schedules(opt.season).to_dicts()
    report=run(source,schedule,season=opt.season,week=opt.week,
               at=datetime.now(timezone.utc))
    opt.out.parent.mkdir(parents=True,exist_ok=True)
    opt.out.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n",encoding="utf8")
    print(json.dumps({"source_games":report["input_diagnostic"]["games_with_source_rows"],
                      "valid_pairs":report["input_diagnostic"]["games_with_valid_paired_quotes"],
                      "small":report["primary_small_spread"],
                      "medium":report["secondary_three_to_seven"],
                      "betting_authorized":False},sort_keys=True))
if __name__=="__main__":main()
