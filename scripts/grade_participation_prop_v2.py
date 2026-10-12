"""Forward grade only original frozen v2 prop shadows after official NFL results.

No retroactive probability rebuild, real bookmaker odds, auto bet or ROI.
Independent immutable ledger separate from original v1. Each original
half-point threshold has a unique first-seen immutable forecast identifier.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts.forward_prop_outcome_audit import grade,append_results,read_csv
from scripts.participation_conditional_prop_v2 import VERSION as SHADOW_VERSION

GRADE_VERSION="PROP_PARTICIPATION_CONDITIONAL_V2_FORWARD_GRADE_V1"


def adapt_original_source(original):
    seen=set()
    mapped=[]
    for r in original:
        if r.get("version")!=SHADOW_VERSION:
            raise ValueError("non-v2 frozen source or source version overwritten")
        if str(r.get("recommended_bet","")).lower()!="false" or r.get("american_odds"):
            raise ValueError("original research-only forecast must not claim quoted bet")
        if r.get("bookmaker"):
            raise ValueError("original frozen research-only model unexpectedly has bookmaker")
        pair=(r["game_id"],r["player_id"],r["market"],str(float(r["synthetic_line"])))
        if pair in seen:
            raise ValueError("duplicate original v2 first-seen contract")
        seen.add(pair)
        mapped.append({
            **r,"probability_over":r["p_over"],
            "probability_under":r["p_under"],
        })
    return mapped


def grade_new(source,game_rows,player_rows,*,observed_utc,existing):
    candidate=adapt_original_source(source)
    if any(x.get("version")!=GRADE_VERSION for x in existing):
        raise ValueError("another scoring model cannot overwrite v2 history")
    newly,report=grade(candidate,game_rows,player_rows,
                       captured_utc=observed_utc,previous=existing)
    for row in newly:
        row["version"]=GRADE_VERSION
        row["original_frozen_model_version"]=SHADOW_VERSION
        row["participation_and_market_void_status"]="UNVERIFIED_IF_MISSING_STATS"
    report["graded_model_version"]=SHADOW_VERSION
    report["scoring_version"]=GRADE_VERSION
    report["historical_trained_only_before_2026"]=True
    report["bookmaker_account_price_verified"]=False
    report["approved_betting_model"]=False
    report["predictive_improvement_not_validated_from_unsettled_games"]=True
    report["first_seen_probability_count"]=len(source)
    return newly,report


def run(*,frozen,history,report_path,season=2026):
    if season!=2026:raise ValueError("v2 original freeze is only from 2026")
    source=read_csv(frozen)
    past=[]
    path=Path(history)
    if path.exists():
        past=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines()
              if x.strip()]
    from nfl.data import NFLDataClient
    client=NFLDataClient()
    now=datetime.now(timezone.utc).isoformat()
    schedules=client.load_schedules([season],refresh=True).to_dicts()
    stats=client.load_player_stats([season],refresh=True).to_dicts()
    entries,report=grade_new(source,schedules,stats,
                             observed_utc=now,existing=past)
    if entries:
        append_results(path,entries)
    report["new_grades_persisted"]=len(entries)
    report["existing_original_grades_preserved"]=len(past)
    report["observed_utc"]=now
    report["prediction_original_journal"]=str(frozen)
    report["grade_history"]=str(history)
    file=Path(report_path)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen",type=Path,
                   default=Path("history/prop_participation_shadow_v2_2026.csv"))
    p.add_argument("--history",type=Path,
                   default=Path("history/prop_participation_outcomes_v2_2026.jsonl"))
    p.add_argument("--report",type=Path,
                   default=Path("reports/prop_participation_forward_grade_v2.json"))
    a=p.parse_args()
    run(frozen=a.frozen,history=a.history,report_path=a.report)
if __name__=="__main__":
    main()
