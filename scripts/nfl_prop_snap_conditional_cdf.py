"""NFL PFR snap-conditioned player-stat CDF challenger (research-only).

The target is P(stat > synthetic threshold | official NFL stat row observed).
2024 source-row-conditional normalized residuals are conditioned on prior
observed PFR offensive snap workload; 2025 is a previously-inspected
retrospective diagnostic. NO BetMGM odds, no real player activation/void data,
no BetMGM profit or approved bet. Do not combine with the stat-row presence
proxy into a tradable unconditional win/void probability.

Freeze 2024 catalog once, 2026 original feature group separately. Existing
2026 full-CDF and v1/v2 forecast originals are not changed.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import nflreadpy as nfl
from nfl.data import NFLDataClient
from nfl.prop_challenger import source_frames
from nfl.prop_probability import PROP_THRESHOLD_GRID,_scale,over_probability
from scripts.participation_conditional_prop_v2 import DEFAULT_OPTIONS,samples
from scripts.nfl_prop_snap_stat_row_diagnostic import make_cohorts
from scripts.nfl_pfr_prior_snap_coverage import unique_crosswalk
from scripts.nfl_player_prop_full_distribution import (
    hash_object,load_or_create_calibration,group_for
)

VERSION="NFL_SNAP_BUCKET_STAT_CDF_RESEARCH_V1"
FIT_YEAR=2024
DIAGNOSTIC_YEAR=2025
SNAP_CUTOFF=35.0
MIN_POSITION=80
MIN_MARKET=160
BOOTSTRAPS=400
HORIZON_DAYS=10

def _utc(x):
    d=datetime.fromisoformat(str(x).replace("Z","+00:00"))
    if d.tzinfo is None or d.utcoffset()!=timedelta(0):
        raise ValueError("UTC timestamp required")
    return d.astimezone(timezone.utc)

def _lineage(catalog):
    sha=catalog.get("sha256")
    if not isinstance(sha,str) or sha!=hash_object({k:v for k,v in catalog.items() if k!="sha256"}):
        raise ValueError("corrupted original 2024 full-CDF reference catalog")
    return sha

def bucket(prior_snap_mean):
    x=float(prior_snap_mean)
    if not math.isfinite(x) or x<0 or x>150:
        raise ValueError("invalid prior observed snap mean")
    return "HIGH_35_PLUS" if x>=SNAP_CUTOFF else "LOW_UNDER_35"

def stat_samples_with_workload(games,player_stats,raw_schedules,raw_snaps,ids,year):
    contexts,missing=make_cohorts(games,player_stats,raw_schedules,raw_snaps,ids,season=year)
    unique={}
    for c in contexts:
        key=(str(c["game_id"]),str(c["player_id"]))
        if key in unique:raise ValueError("duplicate player-game prior snap context")
        unique[key]=bucket(60*float(c["features"][1]))
    matched=[]
    selected=samples(games,player_stats,year,DEFAULT_OPTIONS)["participation_proxy"]
    for p,actual in selected:
        b=unique.get((str(p["game_id"]),str(p["player_id"])))
        if b is None:continue
        market=p["market"]
        if market not in PROP_THRESHOLD_GRID:continue
        pred=float(p["challenger_mean"])
        if not math.isfinite(pred) or not math.isfinite(float(actual)):
            raise ValueError("non-finite conditional player outcome")
        matched.append((p,float(actual),b))
    return matched,{"source_player_games_with_snap_context":len(unique),
                    "conditional_training_or_diagnostic_stat_rows":len(matched),
                    "source_rejected_snap_context":missing}

def fit_buckets(rows,full_original_catalog):
    _lineage(full_original_catalog)
    groups=defaultdict(list)
    for pred,actual,b in rows:
        metric=pred["market"]
        position=pred["position"]
        # Position fallback and pooled market are separately fixed on 2024.
        error=(actual-float(pred["challenger_mean"]))/_scale(metric,float(pred["challenger_mean"]))
        for label in (position,"ALL"):
            groups[f"{metric}|{label}|{b}"].append(float(error))
    fitted={k:sorted(v) for k,v in sorted(groups.items())}
    cat={
        "model":VERSION,
        "development_year":FIT_YEAR,
        "threshold_bucket_definition":{"high_minimum_observed_snaps":SNAP_CUTOFF},
        "minimum_position_residuals":MIN_POSITION,
        "minimum_market_residuals":MIN_MARKET,
        "point_forecast_options_2024":{k:list(v) for k,v in DEFAULT_OPTIONS.items()},
        "base_full_distribution_catalog_sha256":full_original_catalog["sha256"],
        "development_stat_player_market_rows":len(rows),
        "2025_outcomes_used_to_fit":False,
        "fitted_normalized_residual_vectors":fitted,
        "sportsbook_odds_used":False,
    }
    cat["sha256"]=hash_object(cat)
    return cat

def candidate_residuals(catalog,market,position,b):
    key=f"{market}|{position}|{b}"
    pool=f"{market}|ALL|{b}"
    vectors=catalog["fitted_normalized_residual_vectors"]
    if len(vectors.get(key,[]))>=MIN_POSITION:
        return vectors[key],key
    if len(vectors.get(pool,[]))>=MIN_MARKET:
        return vectors[pool],pool
    return None,None

def selection(pred,b,catalog,baseline):
    market=pred["market"]
    key=group_for(market,pred["position"],baseline)
    if key is None:return None
    original=baseline["vectors"][key]
    snap,cand_key=candidate_residuals(catalog,market,pred["position"],b)
    # Insufficient split residuals must transparently FALL BACK to original.
    if snap is None:return original,key,original,key,True
    return original,key,snap,cand_key,False

def score(rows,cat,baseline,*,bootstrap=BOOTSTRAPS,seed=20261012):
    _lineage(baseline)
    if cat["sha256"]!=hash_object({k:v for k,v in cat.items() if k!="sha256"}):
        raise ValueError("tampered snap bucket catalog")
    grouped=defaultdict(list)
    fallback=defaultdict(int)
    unique_player_market=defaultdict(set)
    for pred,actual,b in rows:
        market=pred["market"]
        v=selection(pred,b,cat,baseline)
        if v is None:continue
        base,_,challenger,_,is_fallback=v
        if is_fallback:fallback[market]+=1
        mean=float(pred["challenger_mean"])
        unique_player_market[market].add((pred["game_id"],pred["player_id"],market))
        for line in PROP_THRESHOLD_GRID[market]:
            truth=int(actual>line)
            old=over_probability(mean,line,market,base)
            new=over_probability(mean,line,market,challenger)
            grouped[market].append((
                str(pred["game_id"]),truth,old,new,line,
                str(pred["player_id"])
            ))
    rng=random.Random(seed)
    out={}
    for market,items in sorted(grouped.items()):
        games=defaultdict(list)
        before=sum((p-y)**2 for gid,y,p,q,line,pid in items)/len(items)
        after=sum((q-y)**2 for gid,y,p,q,line,pid in items)/len(items)
        logold=sum(-y*math.log(max(1e-9,p))-(1-y)*math.log(max(1e-9,1-p))
                   for gid,y,p,q,line,pid in items)/len(items)
        lognew=sum(-y*math.log(max(1e-9,q))-(1-y)*math.log(max(1e-9,1-q))
                   for gid,y,p,q,line,pid in items)/len(items)
        for gid,y,p,q,line,pid in items:
            games[gid].append((q-y)**2-(p-y)**2)
        ci=None
        keys=sorted(games)
        if len(keys)>=20:
            values=[]
            for _ in range(bootstrap):
                picked=[keys[rng.randrange(len(keys))] for _ in keys]
                diffs=[d for k in picked for d in games[k]]
                values.append(sum(diffs)/len(diffs))
            values.sort()
            ci=[round(values[int((bootstrap-1)*.025)],8),
                round(values[int((bootstrap-1)*.975)],8)]
        out[market]={
            "source_official_stat_row_player_market_groups":len(unique_player_market[market]),
            "unique_nfl_games":len(keys),
            "correlated_synthetic_lines":len(items),
            "source_stat_rows_using_original_fallback":fallback[market],
            "baseline_original_full_cdf_brier":round(before,8),
            "snap_conditioned_cdf_brier":round(after,8),
            "brier_delta_snap_minus_original":round(after-before,8),
            "game_cluster_bootstrap_ci95":ci,
            "baseline_log_loss":round(logold,8),
            "snap_challenger_log_loss":round(lognew,8),
            "positive_2025_diagnostic":after<before,
            "source_2025_already_inspected":True,
            "account_BetMGM_offer_verified":False,
            "validated_market_profit_edge":False,
        }
    return out

def first_catalog(path,train,base):
    path=Path(path)
    if path.exists():
        result=json.loads(path.read_text(encoding="utf8"))
        if result.get("sha256")!=hash_object({k:v for k,v in result.items() if k!="sha256"}):
            raise ValueError("original 2024 snap CDF catalog SHA invalid")
        fresh=fit_buckets(train,base)
        if fresh["sha256"]!=result["sha256"]:
            raise ValueError("original 2024 snap CDF source changed: do NOT silently refit")
        return result,False
    x=fit_buckets(train,base)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        f.write(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return x,True

def freeze_future(full_originals,snap_context,cat,baseline,*,observed_utc,path):
    _lineage(baseline)
    if cat["sha256"]!=hash_object({k:v for k,v in cat.items() if k!="sha256"}):
        raise ValueError("snap catalog hash mismatch")
    at=_utc(observed_utc)
    by_snap={}
    for s in snap_context:
        key=(str(s["game_id"]),str(s["player_id"]),str(s["market"]))
        if key in by_snap:raise ValueError("duplicate snap source first-seen identity")
        by_snap[key]=s
    dest=Path(path)
    old=[json.loads(x) for x in dest.read_text(encoding="utf8").splitlines() if x.strip()] if dest.exists() else []
    seen={(r["game_id"],r["player_id"],r["market"]) for r in old}
    if len(seen)!=len(old):raise ValueError("duplicate original source candidate")
    fresh=[];used=set();ignored=defaultdict(int)
    for f in full_originals:
        key=(str(f["game_id"]),str(f["player_id"]),str(f["market"]))
        if key in used:raise ValueError("duplicate frozen full distribution")
        used.add(key)
        if key in seen:continue
        ko=_utc(f["kickoff_utc"]);freeze=_utc(f["first_observed_at_utc"])
        if not freeze<=at<ko<=at+timedelta(days=HORIZON_DAYS):
            ignored["OUTSIDE_10_DAY_OR_CHRONOLOGY"]+=1
            continue
        if f["residual_catalog_sha256"]!=baseline["sha256"]:
            raise ValueError("original player-stat CDF catalog hash changed")
        info=by_snap.get(key)
        if info is None or info.get("status")!="HISTORICAL_OFFENSIVE_SNAPS_OBSERVED":
            ignored["NO_QUALIFIED_PRIOR_SNAP_CONTEXT"]+=1
            continue
        snap_at=_utc(info["observed_at_utc"])
        if not freeze<=snap_at<=at<ko:
            raise ValueError("prior snaps originally captured after model freeze")
        b=bucket(info["observed_mean"])
        data=selection({"market":f["market"],"position":f["position"]},b,cat,baseline)
        if data is None:
            ignored["NO_BASE_CDF"]+=1
            continue
        original,oldkey,challenger,newkey,fallback=data
        fresh.append({
            "model":VERSION,"game_id":f["game_id"],
            "player_id":f["player_id"],"player_name":f["player_name"],
            "market":f["market"],"position":f["position"],
            "season":f["season"],"week":f["week"],"kickoff_utc":f["kickoff_utc"],
            "first_observed_at_utc":at.isoformat(),
            "original_player_CDF_first_seen_utc":f["first_observed_at_utc"],
            "original_snap_context_first_seen_utc":info["observed_at_utc"],
            "original_forecast_mean":f["original_point_mean"],
            "original_forecast_scale":f["original_scale"],
            "prior_snap_mean_observed":info["observed_mean"],
            "frozen_bucket":b,"original_CDF_group":oldkey,
            "snap_CDF_group":newkey,"fallback_to_original_full_CDF":fallback,
            "base_2024_CDF_sha256":baseline["sha256"],
            "snap_2024_CDF_sha256":cat["sha256"],
            "current_player_active_verified":False,
            "p_bet_win":None,"p_bet_void":None,
            "offer_line":None,"BetMGM_US_accepted_quote":None,
            "true_EV":None,"decision":"PASS","wager":False,
        })
    if fresh:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("a",encoding="utf8") as f:
            for x in fresh:f.write(json.dumps(x,sort_keys=True,allow_nan=False)+"\n")
    return {"original_prior":len(old),"new_originals":len(fresh),
            "total_original":len(old)+len(fresh),"excluded":dict(ignored)}

def run():
    client=NFLDataClient()
    years=[2024,2025,2026]
    schedules=client.load_schedules(years,refresh=True)
    stats=client.load_player_stats(years,refresh=True)
    nfl_players=nfl.load_players().to_dicts()
    gsis_pfr=unique_crosswalk(nfl_players)
    snaps=nfl.load_snap_counts(years).to_dicts()
    games,player_rows,source=source_frames(schedules,stats)
    original,created_original=load_or_create_calibration(
        "history/player_prop_2024_residual_catalog_v1.json")
    raw=schedules.to_dicts()
    t2024,meta24=stat_samples_with_workload(
        games,player_rows,raw,snaps,gsis_pfr,2024)
    catalog,newly=first_catalog(
        "history/player_prop_prior_snap_cdf_2024_v1.json",t2024,original)
    d2025,meta25=stat_samples_with_workload(
        games,player_rows,raw,snaps,gsis_pfr,2025)
    diagnostics=score(d2025,catalog,original)
    full_path=Path("history/player_prop_full_distributions_2026.jsonl")
    snap_path=Path("history/player_prop_prior_snap_coverage_2026.jsonl")
    future_full=[json.loads(x) for x in full_path.read_text().splitlines() if x.strip()]
    future_snap=[json.loads(x) for x in snap_path.read_text().splitlines() if x.strip()]
    captured=datetime.now(timezone.utc).isoformat()
    freeze=freeze_future(
        future_full,future_snap,catalog,original,observed_utc=captured,
        path="history/player_prop_snap_cdf_sidecar_v1_2026.jsonl")
    result={
        "model":VERSION,"decision":"PASS",
        "development_year":2024,"diagnostic_year":2025,
        "reference_2024_CDF_SHA":original["sha256"],
        "2024_first_snap_catalog_created":newly,
        "2024_snap_CDF_sha256":catalog["sha256"],
        "snap_bucket_cutoff":SNAP_CUTOFF,
        "2024_snap_conditioned_rows":len(t2024),
        "2024_group_counts":{k:len(v) for k,v in catalog["fitted_normalized_residual_vectors"].items()},
        "2025_same_population_snap_rows":len(d2025),
        "2025_market_diagnostics":diagnostics,
        "2024_context":meta24,"2025_context":meta25,
        "2025_already_inspected_not_independent_clean_test":True,
        "historical_PFR_source_original_publication_UTC_verified":False,
        "2026_first_seen_freeze":freeze,
        "source_rows":source,
        "synthetic_grid_not_real_bookmaker_lines":True,
        "2026_original_full_and_v2_forecasts_modified":False,
        "player_active_or_void_model_verified":False,
        "US_BetMGM_account_accepted_prices":0,
        "profitability_validated":False,"approved_bets":0,
        "true_EV":None,
    }
    p=Path("reports/player_prop_prior_snap_cdf_research.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in (
        "model","decision","2024_snap_conditioned_rows",
        "2025_same_population_snap_rows","2025_market_diagnostics",
        "2026_first_seen_freeze","profitability_validated")},indent=2,sort_keys=True))
    return result

if __name__=="__main__":run()
