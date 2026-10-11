"""Diagnostic discrete win/push/loss probabilities for NFL margin and total.
Compares fixed 2023-24 football-only normal and kernel-empirical distributions
on PREVIOUSLY INSPECTED 2025, using archived-final line thresholds only.
Never claims calibrated BetMGM edge or alters canonical probability pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from math import erf, exp, floor, ceil, sqrt, log, isfinite
import json
from pathlib import Path
import random

from scripts.score_market_accuracy import (
    football_predictions, attach_archive, MARKET_STAGE)
from nfl.data import NFLDataClient

VERSION="discrete_nfl_score_research_v1"
DEV_YEARS=(2023,2024)
TEST_YEAR=2025
KERNEL_BANDWIDTH={"margin":7.0,"total":10.0}
NORMAL_MIX=.25

def _cdf(z):
    return .5*(1+erf(z/sqrt(2.)))

def _moments(dev,target):
    errors=[x["actual_"+target]-x["model_"+target] for x in dev]
    if len(errors)<80:
        raise ValueError("insufficient football residuals")
    mean=sum(errors)/len(errors)
    sigma=sqrt(sum((e-mean)**2 for e in errors)/(len(errors)-1))
    if sigma<1:
        raise ValueError("degenerate residual distribution")
    return mean,sigma

def normal_probs(projection,line,mean,sigma):
    """NFL points and margin are discrete INTEGERS. Normal is integrated
    over integer bins, retaining pushes at whole-number thresholds."""
    if not all(isfinite(float(x)) for x in (projection,line,mean,sigma)) or sigma<=0:
        raise ValueError("nonfinite inputs or invalid sigma")
    mu=projection+mean
    f=lambda k:_cdf((k+.5-mu)/sigma)
    lose=f(ceil(line)-1)
    win=1-f(floor(line))
    push=max(0.,1-win-lose)
    return win,push,lose

def empirical_probs(train,target,projection,line,base):
    """Fixed conditional actual-score kernel with normal prior blending."""
    bw=KERNEL_BANDWIDTH[target]
    scores=[r["actual_"+target] for r in train]
    expected=[r["model_"+target] for r in train]
    weights=[.05+exp(-.5*((projection-x)/bw)**2) for x in expected]
    denom=sum(weights)
    win=sum(w for w,y in zip(weights,scores) if y>line)/denom
    push=sum(w for w,y in zip(weights,scores) if y==line)/denom
    loss=sum(w for w,y in zip(weights,scores) if y<line)/denom
    mix=NORMAL_MIX
    result=[(1-mix)*p+mix*b for p,b in zip((win,push,loss),base)]
    return tuple(max(0.,x) for x in result)

def _scores(records):
    n=len(records)
    if not n:return {"n":0,"brier":None,"log_loss":None}
    brier=sum(sum((p-(1 if i==real else 0))**2
                  for i,p in enumerate(probs)) for real,probs,_ in records)/n
    logloss=sum(-log(max(probs[real],1e-12)) for real,probs,_ in records)/n
    pushes=sum(real==1 for real,_,_ in records)
    return {"n":n,"multiclass_brier":round(brier,7),
            "multiclass_log_loss":round(logloss,7),"actual_pushes":pushes}

def audit(rows):
    dev=[r for r in rows if r["season"] in DEV_YEARS]
    test=[r for r in rows if r["season"]==TEST_YEAR]
    result={"version":VERSION,"status":"DIAGNOSTIC_RESEARCH_ONLY",
            "training_seasons":list(DEV_YEARS),"diagnostic_season":TEST_YEAR,
            "diagnostic_pristine":False,"decision":"PASS",
            "market_source_stage":MARKET_STAGE,
            "has_verifiable_pregame_book_prices":False,
            "positive_ev_verified":False,"by_market":{}}
    for target in ("margin","total"):
        mean,sigma=_moments(dev,target)
        base,challenger=[],[]
        deltas=defaultdict(list)
        for r in test:
            if r.get("market_"+target) is None:continue
            line=r["market_"+target]
            y=r["actual_"+target]
            outcome=0 if y>line else 1 if y==line else 2
            gaussian=normal_probs(r["model_"+target],line,mean,sigma)
            empirical=empirical_probs(dev,target,r["model_"+target],line,gaussian)
            base.append((outcome,gaussian,r["week"]))
            challenger.append((outcome,empirical,r["week"]))
            deltas[r["week"]].append(
                -log(max(empirical[outcome],1e-12))+
                 log(max(gaussian[outcome],1e-12)))
        if len(base)<80:raise ValueError("missing archived line thresholds")
        rng=random.Random(20261011)
        weekmeans=[sum(v)/len(v) for v in deltas.values()]
        draws=sorted(sum(weekmeans[rng.randrange(len(weekmeans))]
                         for __ in weekmeans)/len(weekmeans)
                     for _ in range(1000))
        n=_scores(base); c=_scores(challenger)
        result["by_market"][target]={
            "historical_gaussian":n,"fixed_empirical_conditional":c,
            "challenger_minus_gaussian_log_loss":
                round(c["multiclass_log_loss"]-n["multiclass_log_loss"],7),
            "descriptive_week_cluster_ci95":[round(draws[24],7),round(draws[974],7)],
            "fixed_kernel_bandwidth":KERNEL_BANDWIDTH[target],
            "fixed_normal_prior_weight":NORMAL_MIX,
            "promotion_eligible":False}
    result["limitations"]=[
        "Market thresholds from archive-final schedules have no decision-time BetMGM quote timestamp.",
        "Historical football source revisions may differ from original pregame versions.",
        "2025 already inspected: descriptive diagnostic only, not fresh untouched holdout.",
        "No live QB/injury/weather, exact executable odds, or independently proven profitability."]
    return result

def main():
    data=NFLDataClient().load_schedules([2022,2023,2024,2025])
    rows=attach_archive(football_predictions(data),data)
    result=audit(rows)
    out=Path("reports/discrete_score_diagnostic.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__":main()
