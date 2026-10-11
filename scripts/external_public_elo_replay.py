"""Independent reimplementation of *public* FiveThirtyEight-style NFL Elo.

Research-only: source documented K=20, MOV factor, 65 Elo HFA and 1/3
offseason regression. Not the proprietary QB-adjusted or contemporary 538
model. Total score uses separately pre-specified, shrunk trailing scoring
rates; 538 Elo itself does NOT forecast totals.

Chronological weekly snapshots: update Elo and team scoring histories only
AFTER producing every prediction for that week. 2019-22 calibrates Elo-to-
point scaling and the fixed total offset; 2023-25 were previously inspected
and are purely HISTORICAL diagnostics, not untouched test data.

All source market lines join after football forecasts exist. Never bet.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import polars as pl

from nfl.data import NFLDataClient
from nfl.priors import build_prior_walkforward

VERSION = "external_public_elo_replication_v1"
BASE_ELO = 1505.0
HFA_ELO = 65.0
K_FACTOR = 20.0
REVERSION = 1 / 3
CALIBRATION_YEARS = (2019, 2020, 2021, 2022)
DIAGNOSTIC_YEARS = (2023, 2024, 2025)
FIRST_SEASON = 2014
FIRST_WEEK = 5
MIN_CALIBRATION_GAMES = 500
ALIASES = {"OAK": "LV", "LV": "LV", "STL": "LA", "LAR": "LA",
           "LA": "LA", "SD": "LAC", "LAC": "LAC"}


def franchise(code):
    return ALIASES.get(str(code), str(code))


def elo_probability(delta):
    return 1 / (1 + 10 ** (-delta / 400))


def mov_multiplier(margin, elo_delta, outcome):
    pd = max(abs(float(margin)), 1)
    direction = 0 if outcome == 0.5 else (1 if outcome == 1 else -1)
    denominator = (elo_delta * direction * .001 + 2.2)
    if denominator <= .5:
        raise ValueError("unreasonable rating difference")
    return math.log(pd + 1) * (2.2 / (1 if outcome == .5 else denominator))


def _scoring_average(history, season, baseline):
    """Shrunk offense and defense, never current-week final scores."""
    if not history:
        return baseline, baseline
    recent = history[-8:]
    weights = [1.0 if y == season else .25 for y,_,_ in recent]
    n = sum(weights)
    offense = sum(w * pf for w,(_,pf,pa) in zip(weights,recent)) / n
    defense = sum(w * pa for w,(_,pf,pa) in zip(weights,recent)) / n
    factor = n / (n + 8.0)
    return (factor * offense + (1-factor)*baseline,
            factor * defense + (1-factor)*baseline)


def _schedules(schedules):
    required = {"game_id","season","week","game_type","home_score",
                "away_score","home_team","away_team","location"}
    if not required.issubset(set(schedules.columns)):
        raise ValueError("missing required schedule fields")
    games = schedules.filter((pl.col("game_type")=="REG")&
             pl.col("home_score").is_not_null() &
             pl.col("away_score").is_not_null() &
             (pl.col("season")>=FIRST_SEASON)&
             (pl.col("season")<=max(DIAGNOSTIC_YEARS)))
    return [dict(r) for r in games.sort(["season","week","game_id"]).iter_rows(named=True)]


def pregame_elo_and_scoring(schedules):
    """Every target-week game's projection occurs before that week's updates."""
    rows = _schedules(schedules)
    grouped = defaultdict(list)
    for g in rows:
        grouped[(int(g["season"]),int(g["week"]))].append(g)
    ratings={}
    score_hist=defaultdict(list)
    league_points=[]
    prev_season=None
    output=[]
    for (season,week) in sorted(grouped):
        if season != prev_season:
            if prev_season is not None:
                ratings={k: BASE_ELO*REVERSION+v*(1-REVERSION)
                         for k,v in ratings.items()}
            prev_season=season
        batch=grouped[(season,week)]
        participating=[]
        seen=set()
        for g in batch:
            home=franchise(g["home_team"])
            away=franchise(g["away_team"])
            if home==away or home in seen or away in seen:
                raise ValueError("invalid simultaneous NFL team-week")
            seen.update((home,away))
            h=ratings.get(home,BASE_ELO)
            a=ratings.get(away,BASE_ELO)
            is_neutral = str(g["location"]).lower() == "neutral"
            elo_diff=h-a+(0 if is_neutral else HFA_ELO)
            baseline=(sum(league_points[-256:]) /
                     (2*len(league_points[-256:]))) if league_points else 22.
            off_h,def_h=_scoring_average(score_hist[home],season,baseline)
            off_a,def_a=_scoring_average(score_hist[away],season,baseline)
            # Fixed, football-only 1-point home venue uplift in each
            # team's scoring mean. Neutral stadium receives no uplift.
            home_pts=(off_h+def_a)/2 + (0 if is_neutral else 1.)
            away_pts=(off_a+def_h)/2
            hp=float(g["home_score"])
            ap=float(g["away_score"])
            record=dict(game_id=str(g["game_id"]),season=season,week=week,
                        home_team=home,away_team=away,
                        elo_delta=elo_diff,
                        elo_home_win_probability=elo_probability(elo_diff),
                        raw_trailing_total=home_pts+away_pts,
                        actual_margin=hp-ap,actual_total=hp+ap)
            if season >= min(CALIBRATION_YEARS):
                output.append(record)
            participating.append((home,away,hp,ap,elo_diff))
        # CRITICAL: no forecast reads the current week's actual scores.
        for home,away,hp,ap,elo_diff in participating:
            actual=hp-ap
            outcome = 1. if actual>0 else 0. if actual<0 else .5
            shift=K_FACTOR*mov_multiplier(actual,elo_diff,outcome)*(
                outcome-elo_probability(elo_diff))
            ratings[home]=ratings.get(home,BASE_ELO)+shift
            ratings[away]=ratings.get(away,BASE_ELO)-shift
            score_hist[home].append((season,hp,ap))
            score_hist[away].append((season,ap,hp))
            league_points.append(hp+ap)
    return output


def _metrics(rows,pred,actual):
    errors=[float(r[pred])-float(r[actual]) for r in rows if r.get(pred) is not None]
    if not errors:
        return {"n":0,"mae":None,"rmse":None,"bias":None}
    n=len(errors)
    return {"n":n,"mae":round(sum(abs(v) for v in errors)/n,6),
            "rmse":round(math.sqrt(sum(v*v for v in errors)/n),6),
            "bias":round(sum(errors)/n,6)}


def _week_ci(rows,model,baseline,target):
    groups=defaultdict(list)
    for r in rows:
        if r.get(model) is not None and r.get(baseline) is not None:
            y=r["actual_"+target]
            groups[(r["season"],r["week"])].append(abs(r[model]-y)-abs(r[baseline]-y))
    if len(groups)<8:return None
    means=[sum(v)/len(v) for v in groups.values()]
    rng=random.Random(20261011)
    draws=sorted(sum(means[rng.randrange(len(means))] for _ in means)/len(means)
                 for __ in range(1000))
    return [round(draws[24],6),round(draws[974],6)]


def fit_calibration(rows):
    dev=[r for r in rows if r["season"] in CALIBRATION_YEARS]
    if len(dev)<MIN_CALIBRATION_GAMES:
        raise ValueError("insufficient strictly pre-2023 development observations")
    # Only 2019-22 outcomes fit these coefficients. Clipped to avoid
    # wildly extrapolated predictions on sparse historical data.
    num=sum(r["elo_delta"]*r["actual_margin"] for r in dev)
    den=sum(r["elo_delta"]**2 for r in dev)
    margin_coef=max(0.,min(.1,num/den))
    total_offset=sum(r["actual_total"]-r["raw_trailing_total"]
                     for r in dev)/len(dev)
    total_offset=max(-5.,min(5.,total_offset))
    return margin_coef,total_offset


def evaluate(elo_rows,canonical_rows,market_rows):
    """Always compares identical event IDs; reject missing cohort games."""
    slope,offset=fit_calibration(elo_rows)
    canonical={r["game_id"]:r for r in canonical_rows}
    market={str(r["game_id"]):dict(r) for r in
            market_rows.select(["game_id","spread_line","total_line"]).iter_rows(named=True)}
    paired=[]
    for r in elo_rows:
        if r["season"] not in DIAGNOSTIC_YEARS or r["week"]<FIRST_WEEK:
            continue
        base=canonical.get(r["game_id"])
        quote=market.get(r["game_id"])
        if base is None or quote is None:
            raise ValueError("canonical or archived game missing: "+r["game_id"])
        if abs(base["actual_margin"]-r["actual_margin"])>1e-8 or abs(base["actual_total"]-r["actual_total"])>1e-8:
            raise ValueError("mismatched true game results")
        home_margin=float(quote["spread_line"]) if quote["spread_line"] is not None else None
        total=float(quote["total_line"]) if quote["total_line"] is not None else None
        paired.append({
            **r,
            "elo_margin":r["elo_delta"]*slope,
            "elo_trailing_total":r["raw_trailing_total"]+offset,
            "canonical_margin":float(base["projected_home_margin"]),
            "canonical_total":float(base["projected_total"]),
            "market_margin":home_margin,
            "market_total":total,
        })
    report={
        "model":"PUBLIC_538_STYLE_ELO_PLUS_OWN_SHRUNK_TRAILING_TOTAL_V1",
        "status":"RESEARCH_DIAGNOSTIC_ONLY",
        "external_source":"https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py",
        "external_538_exact_replication":False,
        "external_538_total_forecast_claim":False,
        "elo_parameters":{"k_factor":K_FACTOR,"home_elo":HFA_ELO,
                          "offseason_reversion_fraction":REVERSION},
        "development_only_years":list(CALIBRATION_YEARS),
        "diagnostic_years_already_examined":list(DIAGNOSTIC_YEARS),
        "development_elo_margin_points_per_elo":round(slope,9),
        "development_trailing_total_offset":round(offset,9),
        "not_pristine_holdout":True,
        "market_prices_executable":False,
        "market_stage":"NFLVERSE_ARCHIVE_FINAL_NOT_ENTRY_QUOTE",
        "canonical_modified":False,
        "model_profit_proven":False,
        "decision":"PASS",
        "by_season":{},
    }
    for season in DIAGNOSTIC_YEARS:
        part=[r for r in paired if r["season"]==season]
        per={}
        for target in ("margin","total"):
            q=[r for r in part if r["market_"+target] is not None]
            alternate="elo_margin" if target=="margin" else "elo_trailing_total"
            baseline="canonical_"+target
            per[target]={
                "n":len(part),"paired_archive_n":len(q),
                "external_public_method":_metrics(part,alternate,"actual_"+target),
                "canonical_fair_score":_metrics(part,baseline,"actual_"+target),
                "archive_final_market":_metrics(q,"market_"+target,"actual_"+target),
                "method_minus_canonical_mae_ci95_week_bootstrap":
                    _week_ci(part,alternate,baseline,target),
                "method_minus_archive_mae_ci95_week_bootstrap":
                    _week_ci(q,alternate,"market_"+target,target),
                "beats_canonical_mae":_metrics(part,alternate,"actual_"+target)["mae"]<
                                      _metrics(part,baseline,"actual_"+target)["mae"],
                "market_price_proof":False,
            }
        report["by_season"][str(season)]=per
    report["diagnostic_games"]=len(paired)
    report["row_fingerprint_sha256"]=hashlib.sha256(
        json.dumps(paired,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    report["limitations"]=[
        "All 2023-25 games were inspected previously and cannot support release promotion.",
        "Elo is a 538-style simplified public implementation, not its proprietary contemporary QB adjustment.",
        "Totals are our trailing offense/defense averages; FiveThirtyEight Elo does not project totals.",
        "Source vintage before kickoff is not reproduced from today's revised NFL archive.",
        "Archive-final market stages lack authenticated decision-time U.S. BetMGM prices.",
        "No account-executable odds or calibrated independent P(win,push,loss); betting PASS.",
    ]
    return report


def main():
    from nfl.data import NFLDataClient
    from nfl.priors import build_prior_walkforward
    all_schedules=NFLDataClient().load_schedules(list(range(FIRST_SEASON,2026)))
    elo=pregame_elo_and_scoring(all_schedules)
    baseline=[]
    for season in DIAGNOSTIC_YEARS:
        for g in build_prior_walkforward(all_schedules,season,prior_weight=.10,start_week=FIRST_WEEK,ridge=8.).iter_rows(named=True):
            baseline.append({
                "game_id":str(g["game_id"]),
                "projected_home_margin":float(g["projected_home_margin"]),
                "projected_total":float(g["projected_total"]),
                "actual_margin":float(g["actual_home_margin"]),
                "actual_total":float(g["actual_total"]),
            })
    report=evaluate(elo,baseline,all_schedules)
    report["captured_utc"]=datetime.now(timezone.utc).isoformat()
    report["source_dataset"]="nflverse schedules/nfldata via nflreadpy"
    output=Path("reports/external_elo_replay.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"model":report["model"],"diagnostic_games":report["diagnostic_games"],
                      "fit":report["development_elo_margin_points_per_elo"],
                      "total_offset":report["development_trailing_total_offset"],
                      "by_season":report["by_season"]},indent=2))
    print("PASS / research-only. No verified U.S. BetMGM advantage.")


if __name__=="__main__":
    main()
