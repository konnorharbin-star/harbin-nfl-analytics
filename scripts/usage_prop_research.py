"""Research-only NFL player usage × efficiency challenger (2024/2025).

No sportsbook lines, expert selections, execution, or betting EV. A 2024-only
fixed-grid selection evaluates against previously inspected 2025 diagnostics.
Prior-week-only weekly player statistics, selected on the same pregame roster
cohort as the existing frozen prop_challenger benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
import random

from nfl.data import NFLDataClient
from nfl.prop_challenger import forecast_week, source_frames

VERSION = "usage_efficiency_prop_research_v1"
MARKETS = ("receptions", "receiving_yards", "rushing_yards", "passing_yards")
# Fixed, interpretable and deliberately NOT optimized to 2025.
PSEUDO = {
    "catch_rate": (8.0, 0.65),
    "yards_per_reception": (12.0, 10.0),
    "yards_per_carry": (20.0, 4.2),
    "yards_per_attempt": (50.0, 7.0),
}
OPTIONS = {
    "receptions": (8, 0.0),
    "receiving_yards": (8, 0.0),
    "rushing_yards": (8, 0.0),
    "passing_yards": (3, 0.0),
}
WINDOWS = (3, 5, 8)
NEEDS = {"targets", "carries", "attempts"}
MAX_STALENESS_WEEKS = 2


def _nonnegative(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if isfinite(n) and n >= 0 else None


def add_opportunity_columns(rows, frame):
    """Join optional opportunity columns by validated completed player/game IDs.

    A bad or missing opportunity field is UNKNOWN, not silently a zero.
    """
    missing = NEEDS - set(frame.columns)
    if missing:
        raise ValueError("Missing weekly nflverse opportunity fields: " + ",".join(sorted(missing)))
    lookup = {}
    for item in frame.select(["game_id", "player_id", "targets", "carries", "attempts"]).iter_rows(named=True):
        key = (str(item["game_id"]), str(item["player_id"]))
        if key in lookup:
            # Duplicates on source identity could make future rows ambiguous.
            raise ValueError("Duplicate player-game opportunity observation: " + str(key))
        lookup[key] = {k:_nonnegative(item[k]) for k in NEEDS}
    enriched=[]
    for r in rows:
        row=dict(r)
        match=lookup.get((r["game_id"],r["player_id"]))
        if match is None:
            raise ValueError("Missing opportunity data for validated player-game: "+r["game_id"])
        row.update(match)
        enriched.append(row)
    return enriched


def _usage_forecast(row, past, window):
    market=row["market"]
    if market not in MARKETS or window not in WINDOWS:
        raise ValueError("Unsupported market or fixed window")
    field = {"receptions":"targets","receiving_yards":"targets",
             "rushing_yards":"carries","passing_yards":"attempts"}[market]
    pool=(past.get((row["season"],row["team"],row["player_id"]), [])
          if isinstance(past,dict) else past)
    previous=[r for r in pool
              if r["season"]==row["season"] and r["week"]<row["week"]
              and r["team"]==row["team"] and r["player_id"]==row["player_id"]]
    previous.sort(key=lambda x:(x["week"],x["game_id"]))
    recent=[r for r in previous if r[field] is not None and r[market] is not None][-window:]
    if len(recent)<3:
        return None
    if row["week"]-recent[-1]["week"]>MAX_STALENESS_WEEKS:
        return None
    volume=sum(r[field] for r in recent)/len(recent)
    if market=="passing_yards":
        opportunity=sum(r["attempts"] for r in recent)
        yards=sum(r["passing_yards"] for r in recent)
        pseudo,mean=PSEUDO["yards_per_attempt"]
        rate=(yards+pseudo*mean)/(opportunity+pseudo)
        prediction=volume*max(0,min(rate,15))
    elif market=="rushing_yards":
        opportunity=sum(r["carries"] for r in recent)
        yards=sum(r["rushing_yards"] for r in recent)
        pseudo,mean=PSEUDO["yards_per_carry"]
        rate=(yards+pseudo*mean)/(opportunity+pseudo)
        prediction=volume*max(0,min(rate,15))
    else:
        # Marginal reception and receiving-yard predictions from opportunity
        # × probability of catch × yards conditioned on a catch.
        if any(r["receptions"] is None or r["receiving_yards"] is None
               or r["targets"] is None for r in recent):
            return None
        targets=sum(r["targets"] for r in recent)
        catches=sum(r["receptions"] for r in recent)
        yards=sum(r["receiving_yards"] for r in recent)
        pseudo,prior=PSEUDO["catch_rate"]
        catch_prob=min(1,max(0,(catches+pseudo*prior)/(targets+pseudo)))
        prediction=volume*catch_prob
        if market=="receiving_yards":
            pseudo,prior=PSEUDO["yards_per_reception"]
            per_catch=(yards+pseudo*prior)/(catches+pseudo)
            prediction*=min(40,max(0,per_catch))
    return prediction if isfinite(prediction) and prediction>=0 else None


def _examples(games, stats, season):
    targets = {(r["game_id"],r["player_id"]):r for r in stats}
    examples=[]
    completed_by_week=defaultdict(set)
    for g in games:
        if g["season"]==season and g["completed"]:
            completed_by_week[g["week"]].add(g["game_id"])
    for week in sorted(completed_by_week):
        if week<4:
            continue
        for r in forecast_week(games,stats,season=season,week=week,options=OPTIONS):
            if r["game_id"] not in completed_by_week[week]:
                continue
            actual=targets.get((r["game_id"],r["player_id"]))
            value=actual.get(r["market"]) if actual else 0.0
            if value is None or not isfinite(float(value)):
                continue
            examples.append((r,float(value)))
    return examples


def _stats(errors):
    if not errors:
        return {"n":0,"mae":None,"rmse":None}
    n=len(errors)
    return {"n":n,
            "mae":round(sum(abs(x) for x in errors)/n,6),
            "rmse":round(sqrt(sum(x*x for x in errors)/n),6)}


def _cluster_interval(errors_by_week):
    """Descriptive week-level bootstrap, not a profit CI or independent holdout."""
    weeks=sorted(errors_by_week)
    if len(weeks)<8:
        return None
    averages=[sum(errors_by_week[w])/len(errors_by_week[w]) for w in weeks]
    rng=random.Random(20261011)
    draws=[]
    for _ in range(800):
        draws.append(sum(averages[rng.randrange(len(weeks))]
                         for __ in weeks)/len(weeks))
    draws.sort()
    return [round(draws[19],6),round(draws[779],6)]


def run_study(games, stats):
    """2024 tunes VOLUME WINDOW ONLY; 2025 is post-hoc diagnostic."""
    training=_examples(games,stats,2024)
    test=_examples(games,stats,2025)
    if not training or not test:
        raise ValueError("Missing 2024 or 2025 completed historical player games")
    histories=defaultdict(list)
    for item in stats:
        histories[(item["season"],item["team"],item["player_id"])].append(item)
    report={"status":"HISTORICAL_RESEARCH_ONLY","spec_version":VERSION,
            "training_season":2024,"diagnostic_season":2025,
            "diagnostic_is_untouched_holdout":False,
            "market_odds_available":False,"real_betmgm_ev_validated":False,
            "automatic_wagering":False,"paid_api_used":False,
            "by_market":{}}
    for market in MARKETS:
        train=[(r,y) for r,y in training if r["market"]==market]
        hold=[(r,y) for r,y in test if r["market"]==market]
        if len(train)<75 or len(hold)<75:
            report["by_market"][market]={"status":"INSUFFICIENT_PLAYER_SAMPLE",
                                        "train_rows":len(train),"diagnostic_rows":len(hold)}
            continue
        candidates=[]
        for window in WINDOWS:
            forecasted=[(_usage_forecast(r,histories,window),y) for r,y in train]
            eligible=[(p,y) for p,y in forecasted if p is not None]
            if len(eligible)<75 or len(eligible)/len(train)<.90:
                continue
            candidates.append((sum(abs(p-y) for p,y in eligible)/len(eligible),window))
        if not candidates:
            report["by_market"][market]={"status":"INCOMPLETE_OPPORTUNITY_COVERAGE",
                                        "train_rows":len(train)}
            continue
        chosen=min(candidates)[1]
        ours,existing,rolling=[],[],[]
        per_week=defaultdict(list)
        missing=0
        for r,y in hold:
            p=_usage_forecast(r,histories,chosen)
            if p is None:
                missing+=1
                continue
            a,b=float(r["challenger_mean"]),float(r["baseline_three_game_mean"])
            ours.append(p-y)
            existing.append(a-y)
            rolling.append(b-y)
            per_week[int(r["week"])].append(abs(p-y)-abs(a-y))
        total=len(hold)
        if len(ours)<75 or len(ours)/total<.90:
            report["by_market"][market]={"status":"INCOMPLETE_2025_COVERAGE",
                                        "diagnostic_rows":total,"matched":len(ours)}
            continue
        current=_stats(existing)
        challenger=_stats(ours)
        report["by_market"][market]={
            "status":"DIAGNOSTIC_ONLY_NO_MARKET_PRICES",
            "chosen_2024_volume_window":chosen,
            "train_rows":len(train), "diagnostic_player_game_rows":total,
            "matched_rows":len(ours),"missing_opportunity_rows":missing,
            "usage_challenger":challenger,
            "existing_challenger_same_rows":current,
            "rolling_three_same_rows":_stats(rolling),
            "usage_minus_existing_mae":round(challenger["mae"]-current["mae"],6),
            "week_clustered_mae_difference_ci95":_cluster_interval(per_week),
            "improved_existing_challenger_mae":challenger["mae"]<current["mae"],
            "market_beating_edge_established":False,
        }
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("reports/usage_prop_2024_2025.json"))
    args=parser.parse_args()
    client=NFLDataClient()
    raw_schedules=client.load_schedules([2024,2025])
    raw_players=client.load_player_stats([2024,2025])
    games,base,quality=source_frames(raw_schedules,raw_players)
    enriched=add_opportunity_columns(base,raw_players)
    result=run_study(games,enriched)
    result["source"]="nflverse weekly player_stats + schedules via NFLDataClient"
    result["source_url"]="https://github.com/nflverse/nflverse-data/releases"
    result["captured_at_utc"]=datetime.now(timezone.utc).isoformat()
    # Fingerprints selected numerical observations, NOT original file bytes.
    payload=[{k:r.get(k) for k in ("game_id","player_id","team","season","week",
             "targets","carries","attempts","receptions","receiving_yards",
             "rushing_yards","passing_yards")} for r in enriched]
    result["normalized_selected_rows_sha256"]=sha256(
        json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    result["input_quality"]=quality
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"by_market":result["by_market"]},
                     sort_keys=True,indent=2))
    print("Decision PASS: diagnostics at unpriced historical player-stat thresholds.")


if __name__=="__main__":
    main()
