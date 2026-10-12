"""Reconcile TWO genuinely frozen 2026 NFL player-prop research cohorts.

New pregame full residual distributions SHOULD regenerate original v2
halfpoint probabilities at their same game/player/market/line, unless source
vintage/feature selections changed between original pregame captures.
Report all differences; NEVER silently rewrite either first-seen original.
No bookmaker prices, bet recommendations, or present-market EV.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
from nfl.prop_probability import over_probability
from scripts.nfl_player_prop_full_distribution import (
    CALIBRATION_VERSION, VERSION as FULL_VERSION, hash_object,
)

VERSION="NFL_ORIGINAL_V2_TO_FULL_DISTRIBUTION_RECONCILIATION_V1"


def utc(x):
    t=datetime.fromisoformat(str(x).replace("Z","+00:00"))
    if t.tzinfo is None or t.utcoffset()!=timedelta(0):
        raise ValueError("timestamp must be UTC")
    return t.astimezone(timezone.utc)


def reconcile(v2, full, catalog, *, tol=1e-5):
    raw={k:v for k,v in catalog.items() if k!="sha256"}
    if hash_object(raw)!=catalog.get("sha256"):
        raise ValueError("full empirical catalog hash does not match frozen content")
    if catalog.get("calibration_version")!=CALIBRATION_VERSION or catalog.get("training_year")!=2024:
        raise ValueError("unexpected fitted outcome vintage")
    full_by={}
    for f in full:
        k=(str(f["game_id"]),str(f["player_id"]),str(f["market"]))
        if k in full_by:
            raise ValueError("duplicate full original player-market cohort")
        if f["model"]!=FULL_VERSION or f["residual_catalog_sha256"]!=catalog["sha256"]:
            raise ValueError("full original model/catalog lineage wrong")
        if not utc(f["first_observed_at_utc"])<utc(f["kickoff_utc"]):
            raise ValueError("late original full distribution")
        full_by[k]=f
    overlap=0
    unmatched=0
    mismatched=0
    max_abs=0.
    differences=[]
    unique=set()
    original_full=set()
    market_diffs=defaultdict(list)
    for r in v2:
        key=(str(r["game_id"]),str(r["player_id"]),str(r["market"]))
        try:
            line=float(r["synthetic_line"])
            archived=float(r["p_over"])
        except (TypeError,ValueError):
            raise ValueError("non-numeric original synthetic line/probability")
        if not (math.isfinite(archived) and 0<=archived<=1):
            raise ValueError("invalid archived v2 probability")
        vk=(*key,line)
        if vk in unique:
            raise ValueError("duplicate v2 exact threshold")
        unique.add(vk)
        f=full_by.get(key)
        if f is None:
            unmatched+=1
            continue
        if (str(f["kickoff_utc"])!=str(r["kickoff_utc"])
            and utc(f["kickoff_utc"])!=utc(r["kickoff_utc"])):
            raise ValueError("different actual kickoff at same game ID")
        if not utc(r["first_observed_at_utc"])<utc(r["kickoff_utc"]):
            raise ValueError("late original v2 threshold")
        errors=catalog["vectors"][f["residual_group_key"]]
        p=over_probability(float(f["original_point_mean"]),line,
                           str(f["market"]),errors)
        delta=p-archived
        overlap+=1
        original_full.add(key)
        max_abs=max(max_abs,abs(delta))
        market_diffs[key[2]].append(delta)
        if abs(delta)>tol:
            mismatched+=1
            if len(differences)<20:
                differences.append({"game_id":key[0],"player_id":key[1],
                    "market":key[2],"line":line,
                    "original_v2_p_over":archived,
                    "new_full_cdf_p_over":round(p,9),
                    "difference":round(delta,9),
                    "v2_first_seen":r["first_observed_at_utc"],
                    "full_first_seen":f["first_observed_at_utc"]})
    return {
        "version":VERSION,"decision":"PASS",
        "old_v2_original_thresholds":len(v2),
        "new_full_original_player_market_groups":len(full),
        "overlap_exact_line_pairs":overlap,
        "v2_thresholds_without_full_distribution_group":unmatched,
        "overlap_unique_player_game_market_groups":len(original_full),
        "threshold_probability_differences_gt_tol":mismatched,
        "tolerance":tol,"max_absolute_probability_delta":round(max_abs,9),
        "candidate_cross_version_equivalence_proven":bool(overlap and mismatched==0 and unmatched==0),
        "by_market":{k:{"line_pairs":len(d),
              "mean_signed_diff":round(sum(d)/len(d),9),
              "maximum_abs_diff":round(max(abs(x) for x in d),9)}
              for k,d in sorted(market_diffs.items())},
        "largest_difference_examples":differences,
        "outcomes_used_for_reconciliation":False,
        "user_BetMGM_account_quote_verified":False,
        "bookmaker_EV_or_profitability_validated":False,
        "wagers":0,
        "source_originals_modified":False,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2",type=Path,default=Path("history/prop_participation_shadow_v2_2026.csv"))
    ap.add_argument("--full",type=Path,default=Path("history/player_prop_full_distributions_2026.jsonl"))
    ap.add_argument("--catalog",type=Path,default=Path("history/player_prop_2024_residual_catalog_v1.json"))
    ap.add_argument("--report",type=Path,default=Path("reports/player_full_vs_v2_original_audit.json"))
    a=ap.parse_args()
    with a.v2.open(newline="",encoding="utf8") as f:
        v2=list(csv.DictReader(f))
    full=[json.loads(t) for t in a.full.read_text(encoding="utf8").splitlines() if t.strip()]
    catalog=json.loads(a.catalog.read_text(encoding="utf8"))
    result=reconcile(v2,full,catalog)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in (
        "old_v2_original_thresholds","new_full_original_player_market_groups",
        "overlap_exact_line_pairs","v2_thresholds_without_full_distribution_group",
        "threshold_probability_differences_gt_tol","max_absolute_probability_delta",
        "candidate_cross_version_equivalence_proven","decision",
    )},indent=2))


if __name__=="__main__":
    main()
