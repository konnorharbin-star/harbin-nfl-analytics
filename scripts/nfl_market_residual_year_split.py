"""Chronological market-residual falsification on retrospective NFL archive.

One and only one fixed selected-side record per game and market in archived
provider-labeled *open* records. Select ONE shrinkage coefficient on 2024
observations, evaluate on 2025, and report full kickoff-week-clustered CI.

Neither season has verified original sportsbook quote timestamps. This
retrospective re-analysis is NOT prospective, cannot establish executable
profitability, and must not be used to certify the new 2026 simulator.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite, log
from pathlib import Path
import csv
import json
import random
import hashlib

SPEC="NFL_ARCHIVED_MARKET_RESIDUAL_DIAGNOSTIC_V1"
MARKETS=("moneyline","spread","total")
WEIGHTS=(-0.5,-0.25,-0.1,0.0,0.1,0.25,0.5)
TRAIN=2024
TEST=2025
SEED=20261013
REPS=4000

def _float(v, name):
    if isinstance(v,bool):
        raise ValueError("INVALID_"+name)
    try:n=float(v)
    except (ValueError,TypeError) as e:raise ValueError("INVALID_"+name) from e
    if not isfinite(n):
        raise ValueError("NONFINITE_"+name)
    return n

def _probs(row):
    market=_float(row.get("no_vig_probability"),"MARKET_PROBABILITY")
    model=_float(row.get("model_probability"),"MODEL_PROBABILITY")
    if not 0<market<1 or not 0<model<1:
        raise ValueError("INVALID_PROBABILITY")
    return market,model

def _prediction(q,p,weight):
    return max(1.e-8,min(1-1.e-8,q+weight*(p-q)))

def _score(records,weight):
    if not records:return None
    model_brier=market_brier=model_ll=market_ll=0.
    for r in records:
        y=r["y"]; q=r["q"]; p=_prediction(q,r["p"],weight)
        model_brier+=(p-y)**2
        market_brier+=(q-y)**2
        model_ll-=y*log(p)+(1-y)*log(1-p)
        market_ll-=y*log(q)+(1-y)*log(1-q)
    n=len(records)
    return {"games":n,"model_brier":model_brier/n,
            "market_brier":market_brier/n,
            "market_minus_model_brier":(market_brier-model_brier)/n,
            "model_log_loss":model_ll/n,
            "market_log_loss":market_ll/n,
            "market_minus_model_log_loss":(market_ll-model_ll)/n}

def _cluster_ci(records,weight,repetitions=REPS):
    groups=defaultdict(list)
    for r in records:
        groups[(r["season"],r["week"])].append(r)
    weeks=list(groups.values())
    if len(weeks)<8:return None
    rng=random.Random(SEED)
    draws=[]
    for _ in range(repetitions):
        batch=[]
        for unused in weeks:
            batch.extend(weeks[rng.randrange(len(weeks))])
        draws.append(_score(batch,weight)["market_minus_model_brier"])
    draws.sort()
    return {"brier_lift_interval_95":[draws[int(.025*repetitions)],
                                      draws[int(.975*repetitions)-1]],
            "distinct_week_clusters":len(weeks),
            "cluster_bootstrap_repetitions":repetitions,
            "multiplicity_adjusted":False}

def analyze(rows):
    """Separate earlier-year training from later-year diagnostic; no wagers."""
    if not isinstance(rows,list):raise ValueError("ROWS_MUST_BE_LIST")
    cohort={m:{TRAIN:[],TEST:[]} for m in MARKETS}
    seen=set()
    for r in rows:
        if r.get("market_type") not in MARKETS or r.get("season") not in ("2024","2025"):
            continue
        market=r["market_type"];year=int(r["season"])
        gid=r.get("game_id")
        if not gid:raise ValueError("MISSING_GAME_ID")
        key=(market,year,gid)
        if key in seen:raise ValueError("DUPLICATE_ARCHIVED_CONTRACT")
        seen.add(key)
        if (r.get("price_stage")!="espn_archived_open" or
            r.get("entry_timestamp_verified")!="false"):
            raise ValueError("NOT_SUPPORTED_ARCHIVE_SOURCE_STAGE")
        if r.get("side") not in ("home","away","over","under"):
            raise ValueError("INVALID_SELECTED_SIDE")
        if r.get("result") not in ("win","loss","push"):
            raise ValueError("INVALID_RESULT")
        wk=r.get("week")
        try:week=int(wk)
        except (ValueError,TypeError) as exc:raise ValueError("INVALID_WEEK") from exc
        if not 1<=week<=22:raise ValueError("INVALID_WEEK")
        q,p=_probs(r)
        if r["result"]=="push":
            # A push is still one game, but not a decided binary trial.
            # Keep it out of winner/loser Brier; return counts transparently.
            continue
        cohort[market][year].append({
            "season":year,"week":week,"gid":gid,
            "q":q,"p":p,"y":int(r["result"]=="win")})
    reports={}
    for m in MARKETS:
        train=cohort[m][TRAIN];test=cohort[m][TEST]
        if len(train)<100 or len(test)<100:
            raise ValueError("INSUFFICIENT_ARCHIVE_FOR_YEAR_SPLIT")
        training=[(w,_score(train,w)) for w in WEIGHTS]
        # Fixed tie-break: prefer no adjustment when training errors tie.
        chosen=min(training,key=lambda x:(x[1]["model_brier"],abs(x[0])))[0]
        scored=_score(test,chosen)
        interval=_cluster_ci(test,chosen)
        reports[m]={
           "train_year":TRAIN,"diagnostic_year":TEST,
           "training_games":len(train),"diagnostic_games":len(test),
           "fixed_candidate_weights":list(WEIGHTS),
           "weight_selected_from_training_only":chosen,
           "training_market_brier":_score(train,0)["model_brier"],
           "training_candidate_brier":_score(train,chosen)["model_brier"],
           "later_year":scored,
           "later_year_cluster_uncertainty":interval,
           "distinct_independent_forward_games":0,
           "genuinely_out_of_sample_in_2026":False,
           "positive_market_lift_statistically_established":bool(
               interval is not None and interval["brier_lift_interval_95"][0]>0),
           "betting_authorized":False,
        }
    return {
      "spec":SPEC,
      "status":"REJECT_MARKET_RESIDUAL_AS_PROVEN_EDGE",
      "source":"ARCHIVED_PROVIDER_LABELED_OPEN_NOT_ACCEPTED_QUOTE",
      "original_quote_publication_timestamp_verified":False,
      "years_already_consulted_in_prior_research":True,
      "2025_truly_untouched_confirmatory_holdout":False,
      "familywise_multiplicity_adjusted":False,
      "model_family":"ARCHIVED_2024_2025_CONDITIONAL_STUDENT_T",
      "new_2026_joint_drive_family_validated":False,
      "markets":reports,
      "promoted_family_count":0,
      "claimed_profitable_edge":False,
      "estimated_true_ev":None,
      "betting_authorized":False,
      "action":"PASS_RESEARCH_ONLY",
    }

def run(path="reports/verified_market_bets.csv",
        out="reports/market_residual_year_split_diagnostic.json"):
    with Path(path).open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    result=analyze(rows)
    result["input_archive_sha256"]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result["input_archive_path"]=str(path)
    target=Path(out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return result

if __name__=="__main__":
    report=run()
    print(json.dumps({
        "status":report["status"],
        "markets":{k:{
            "weight":v["weight_selected_from_training_only"],
            "test_games":v["diagnostic_games"],
            "market_brier":v["later_year"]["market_brier"],
            "candidate_brier":v["later_year"]["model_brier"],
            "lift":v["later_year"]["market_minus_model_brier"],
            "ci":v["later_year_cluster_uncertainty"]["brier_lift_interval_95"],
            "proven":v["positive_market_lift_statistically_established"],
        } for k,v in report["markets"].items()},
        "wager_authorized":False,
    },sort_keys=True))
