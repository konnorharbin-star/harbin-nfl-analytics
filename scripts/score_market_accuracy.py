"""Research-only spread/total accuracy relative to *archive-final* NFL market."""
import json, math, random, hashlib
from pathlib import Path
from collections import defaultdict
import polars as pl
from nfl.data import NFLDataClient
from nfl.priors import build_prior_walkforward

VERSION="score_market_accuracy_v1"
MARKET_STAGE="ARCHIVE_FINAL_NOT_TIME_VERIFIED"

def _number(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):
        return None

def football_predictions(schedules):
    """Canonical ratings fit using prior seasons and strictly previous weeks."""
    rows=[]
    for year in (2023,2024,2025):
        forecasts=build_prior_walkforward(schedules,year,prior_weight=.10,
                                          start_week=5,ridge=8.)
        for p in forecasts.iter_rows(named=True):
            rows.append(dict(game_id=str(p["game_id"]),season=year,
                week=int(p["week"]),model_margin=float(p["projected_home_margin"]),
                model_total=float(p["projected_total"]),
                actual_margin=float(p["actual_home_margin"]),
                actual_total=float(p["actual_total"])))
    return rows

def attach_archive(rows,schedules):
    needed={"game_id","spread_line","total_line"}
    if not needed.issubset(schedules.columns):
        raise ValueError("Schedule lacks archive market fields")
    indexed={}
    for p in schedules.select(sorted(needed)).iter_rows(named=True):
        if p["game_id"] in indexed:
            raise ValueError("duplicate game_id in archive")
        indexed[p["game_id"]]=p
    result=[]
    for r in rows:
        g=indexed.get(r["game_id"],{})
        spread=_number(g.get("spread_line"))
        total=_number(g.get("total_line"))
        result.append({**r,"market_margin":spread if spread is not None and abs(spread)<=35 else None,
                       "market_total":total if total is not None and 20<=total<=90 else None})
    return result

def metrics(rows,target,forecast):
    e=[float(r[forecast])-float(r["actual_"+target]) for r in rows
       if r.get(forecast) is not None]
    if not e:
        return {"n":0,"mae":None,"rmse":None}
    return {"n":len(e),"mae":round(sum(abs(x) for x in e)/len(e),6),
            "rmse":round(math.sqrt(sum(x*x for x in e)/len(e)),6)}

def fit_market_weight(rows,target):
    """Fixed penalty, MSE-optimal alpha on independent-model *difference*."""
    subset=[r for r in rows if r.get("market_"+target) is not None]
    if len(subset)<80:
        raise ValueError("insufficient archive development pairs")
    den=sum((r["model_"+target]-r["market_"+target])**2 for r in subset)
    num=sum((r["actual_"+target]-r["market_"+target])*
            (r["model_"+target]-r["market_"+target]) for r in subset)
    return max(0.,min(1.,num/(den+1250.)))

def week_interval(rows,target):
    groups=defaultdict(list)
    for r in rows:
        if r["market_"+target] is None:
            continue
        y=r["actual_"+target]
        groups[(r["season"],r["week"])].append(
            abs(r["blend_"+target]-y)-abs(r["market_"+target]-y))
    if len(groups)<8: return None
    means=[sum(g)/len(g) for g in groups.values()]
    randomizer=random.Random(20261011)
    samples=sorted(sum(means[randomizer.randrange(len(means))] for _ in means)/len(means)
                   for __ in range(1000))
    return [round(samples[24],6),round(samples[974],6)]

def evaluate(rows):
    if not rows or len({r["game_id"] for r in rows})!=len(rows):
        raise ValueError("empty/duplicate independent game forecasts")
    dev=[r for r in rows if r["season"] in (2023,2024)]
    test=[r for r in rows if r["season"]==2025]
    if len(test)<80: raise ValueError("insufficient 2025 diagnostic games")
    weights={target:fit_market_weight(dev,target) for target in ("margin","total")}
    for r in rows:
        for t in ("margin","total"):
            m=r["market_"+t]
            r["blend_"+t]=m+weights[t]*(r["model_"+t]-m) if m is not None else None
    report={"status":"DEVELOPMENT_DIAGNOSTIC_ONLY",
            "version":VERSION,"market_stage":MARKET_STAGE,
            "independent_model_consumes_market":False,
            "market_quotes_time_verified":False,"bookmaker_prices_executable":False,
            "development_seasons":[2023,2024],"diagnostic_season":2025,
            "2025_is_untouched_holdout":False,
            "market_blend_weight":{k:round(v,7) for k,v in weights.items()},
            "decision":"PASS","positive_ev_validated":False,"by_target":{}}
    for target in ("margin","total"):
        val={}
        for label,subset in (("development",dev),("diagnostic_2025",test)):
            paired=[r for r in subset if r["market_"+target] is not None]
            val[label]={"independent_all":metrics(subset,target,"model_"+target),
                        "paired":len(paired),
                        "independent_paired":metrics(paired,target,"model_"+target),
                        "archive_final":metrics(paired,target,"market_"+target),
                        "blended_archive":metrics(paired,target,"blend_"+target),
                        "descriptive_week_ci_blend_minus_archive":
                            week_interval(paired,target),
                        "observed_exact_line_pushes":
                            sum(r["actual_"+target]==r["market_"+target] for r in paired)}
        report["by_target"][target]=val
    snap=json.dumps(sorted(rows,key=lambda r:r["game_id"]),sort_keys=True,
                    separators=(",",":"))
    report["normalized_score_and_archive_sha256"]=hashlib.sha256(snap.encode()).hexdigest()
    report["limitations"]=[
        "Archived final lines have NO verified pregame timestamp and may reflect late information.",
        "2023-2025 were previously inspected; 2025 is descriptive, not pristine validation.",
        "Market-blended forecast is NOT an independent bet-pricing model.",
        "No independent accepted BetMGM quote, weather, or roster adjustment tested."]
    return report

def main():
    schedules=NFLDataClient().load_schedules([2022,2023,2024,2025])
    report=evaluate(attach_archive(football_predictions(schedules),schedules))
    path=Path("reports/score_market_accuracy.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"weights":report["market_blend_weight"],
       "diagnostic":{k:v["diagnostic_2025"] for k,v in report["by_target"].items()}},indent=2))
if __name__=="__main__":main()
