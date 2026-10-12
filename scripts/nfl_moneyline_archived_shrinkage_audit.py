"""Chronologically fit a conservative moneyline-vs-market *research benchmark*.

NFLVERSE's historical moneyline is an archival snapshot of UNKNOWN offer-origin
time, UNKNOWN sportsbook, and NOT a tradable U.S. BetMGM quote. Consequently this
script NEVER creates a live wager prediction or reports verified profit.

This only measures whether pregame outcome-only Elo adds probability signal
to archived de-vigged market estimates. Forecast source must come from original
strictly week-pregame Elo; market information enters exclusively during the
retrospective comparison after Elo probabilities have been generated.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from nfl.data import NFLDataClient
from scripts.nfl_moneyline_elo_research import (
    VERSION as ELO_VERSION, WARMUP_START, market_home_no_vig, pregame_elo,
    prior_tie_rate,
)

VERSION="NFL_MONEYLINE_ARCHIVED_MARKET_SHRINKAGE_DIAGNOSTIC_V1"
DEVELOPMENT_SEASONS=(2019,2020,2021,2022)
VALIDATION_SEASON=2023
INSPECTED_DIAGNOSTICS=(2024,2025)
WEIGHTS=(0.0,.05,.1,.2,.35,.5,.75,1.0)


def _pair(rows):
    paired=[]
    seen=set()
    for g in rows:
        if int(g["season"]) < min(DEVELOPMENT_SEASONS):
            continue
        if g["home_score"] is None or g["away_score"] is None:
            continue
        if float(g["home_score"]) == float(g["away_score"]):
            continue  # two-way refunded tie excluded from conditional ML log loss
        market=market_home_no_vig(g.get("archive_home_moneyline"),
                                  g.get("archive_away_moneyline"))
        if market is None:
            continue
        if g["game_id"] in seen:
            raise ValueError("duplicate paired NFL game")
        seen.add(g["game_id"])
        elo=float(g["conditional_p_home_win_given_no_tie"])
        if not .0<elo<1. or not .0<market<1.:
            raise ValueError("invalid input probabilities")
        paired.append({"game_id":g["game_id"],
                       "season":int(g["season"]),"week":int(g["week"]),
                       "y":1 if float(g["home_score"])>float(g["away_score"]) else 0,
                       "elo":elo,"market":market})
    return paired


def _loss(y,p):
    q=max(1e-10,min(1-1e-10,float(p)))
    return -(y*math.log(q)+(1-y)*math.log1p(-q))


def blend(row,weight):
    if not 0<=weight<=1:
        raise ValueError("weight must be 0..1")
    return (1-weight)*row["market"]+weight*row["elo"]


def _average(rows,weight):
    if not rows: return None
    return sum(_loss(x["y"],blend(x,weight)) for x in rows)/len(rows)


def choose_weight(rows,*,seasons=DEVELOPMENT_SEASONS,weights=WEIGHTS):
    if not weights or any(w<0 or w>1 for w in weights):
        raise ValueError("invalid preregistered blend grid")
    dev=[r for r in rows if r["season"] in seasons]
    if not dev:
        raise ValueError("no historical development observations")
    # The market-only candidate is explicitly on the grid. Smaller weights
    # always win ties: do not falsely claim incremental Elo prediction value.
    selected=min(weights,key=lambda w:(_average(dev,w),w))
    return selected,{"train_games":len(dev),"train_seasons":list(seasons),
        "selected_elo_weight":selected,
        "dev_market_only_logloss":round(_average(dev,0),8),
        "dev_selected_logloss":round(_average(dev,selected),8),
        "dev_elo_only_logloss":round(_average(dev,1),8)}


def _metrics(rows,weight):
    if not rows:return {"n":0,"model_brier":None,"market_brier":None}
    model=[blend(x,weight) for x in rows]
    y=[x["y"] for x in rows]
    diff=[(p-z)**2-(q-z)**2
          for p,q,z in zip(model,[x["market"] for x in rows],y)]
    return {
        "n":len(rows),"elo_weight":weight,
        "model_brier":round(sum((p-z)**2 for p,z in zip(model,y))/len(y),8),
        "market_brier":round(sum((x["market"]-x["y"])**2 for x in rows)/len(rows),8),
        "model_logloss":round(sum(_loss(x["y"],p) for x,p in zip(rows,model))/len(rows),8),
        "market_logloss":round(sum(_loss(x["y"],x["market"]) for x in rows)/len(rows),8),
        "model_minus_market_brier":round(sum(diff)/len(diff),8),
        "unique_games":len({x["game_id"] for x in rows}),
        "unverified_book_offer_timestamps":len(rows),
        "accepted_BetMGM_US_account_quotes":0,
        "validated_betting_edge":False,
    }


def report(rows,*,min_dev_games=500):
    paired=_pair(rows)
    w,train=choose_weight(paired)
    if train["train_games"]<min_dev_games:
        raise ValueError("insufficient chronological development games")
    validation=[r for r in paired if r["season"]==VALIDATION_SEASON]
    if len(validation)<100:
        raise ValueError("insufficient 2023 validation observations")
    val=_metrics(validation,w)
    by_year={str(y):_metrics(
        [r for r in paired if r["season"]==y],w)
        for y in INSPECTED_DIAGNOSTICS}
    return {
        "version":VERSION,"elo_source":ELO_VERSION,
        "decision":"PASS","research_only":True,
        "winner_probability_priced_live":False,
        "market_column_provenance":"NFLVERSE_ARCHIVED_MONEYLINE_NO_ORIGINAL_OFFER_TIMESTAMP",
        "archive_market_not_an_executable_BetMGM_quote":True,
        "candidate_was_fit_on_2024_2025":False,
        "year_2023_is_retrospective_validation":True,
        "2024_2025_are_already_inspected":True,
        "train":train,"2023_validation":val,"previously_inspected_diagnostics":by_year,
        "improved_against_market_2023":(
            val["model_minus_market_brier"]<0 and
            val["model_logloss"]<val["market_logloss"]),
        "profitable_BetMGM_edge_proven":False,
        "model_wager_recommendations":[],
        "estimated_real_EV":None,"actual_wagers":0,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,default=Path("reports/moneyline_archived_shrinkage_audit.json"))
    x=p.parse_args()
    years=list(range(WARMUP_START,2026))
    schedule=NFLDataClient().load_schedules(years).to_dicts()
    prior,n,t=prior_tie_rate(schedule)
    pregame=pregame_elo(schedule,tie_rate=prior)
    out=report(pregame)
    out["tie_prior_training_games"]=n
    out["tie_prior_training_ties"]=t
    out["source_capture_utc"]=datetime.now(timezone.utc).isoformat()
    x.report.parent.mkdir(parents=True,exist_ok=True)
    x.report.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({k:out[k] for k in ("decision","train","2023_validation",
        "previously_inspected_diagnostics","profitable_BetMGM_edge_proven")},
        indent=2,sort_keys=True))


if __name__=="__main__":
    main()
