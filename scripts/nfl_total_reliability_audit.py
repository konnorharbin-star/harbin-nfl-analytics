"""Price-first diagnostic: does prior NFL total probability signal hold up?

DO NOT tune cutoffs on observed outcomes. This evaluates a fixed 55%-60%
*older* model over-confidence diagnostic on provider-labeled archived prices;
it is NOT original timestamped bookmaker odds, NOT the 2026 joint-drive
simulator, and NOT betting permission. The current 2026 simulator must obtain
its own untouched forward calibration and exact accepted contract quotes.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from math import sqrt, isfinite
from pathlib import Path

SPEC="NFL_ARCHIVED_TOTAL_RELIABILITY_FALSIFICATION_V1"
BAND_LOW=0.55
BAND_HIGH=0.60
SOURCES={"espn_archived_open"}
YEARS={2024,2025}

def _wilson(wins,decided,z=1.959963984540054):
    if not isinstance(wins,int) or not isinstance(decided,int) or not 0<=wins<=decided:
        raise ValueError("BAD_BINOMIAL_SAMPLE")
    if decided==0:return None
    p=wins/decided
    z2=z*z
    mid=(p+z2/(2*decided))/(1+z2/decided)
    half=z*sqrt((p*(1-p)+z2/(4*decided))/decided)/(1+z2/decided)
    return [max(0.,mid-half),min(1.,mid+half)]

def _number(value,*,min_value=None,max_value=None):
    if isinstance(value,bool):
        raise ValueError("NUMERIC_BOOLEAN")
    try:v=float(value)
    except (TypeError,ValueError) as e:raise ValueError("NUMBER_INVALID") from e
    if not isfinite(v) or (min_value is not None and v<min_value) or (
        max_value is not None and v>max_value):
        raise ValueError("NONFINITE_OR_OUT_OF_RANGE")
    return v

def audit(raw_rows):
    """Audit an entire static archived candidate universe, not selected winners.

    One selection per game/market/season. Historical provider stage is not a
    timestamp of an executable offer and cannot support validated ROI.
    """
    if not isinstance(raw_rows,list):raise ValueError("LIST_REQUIRED")
    selected=[]
    seen=set()
    for row in raw_rows:
        if row.get("market_type")!="total" or row.get("season") not in ("2024","2025"):
            continue
        season=int(row["season"])
        if row.get("side") not in ("over","under"):
            raise ValueError("BAD_TOTAL_SIDE")
        gid=row.get("game_id")
        if not gid:raise ValueError("MISSING_GAME_ID")
        key=(season,gid)
        if key in seen:
            raise ValueError("MULTIPLE_PRICE_SELECTIONS_SAME_GAME_IN_ARCHIVE")
        seen.add(key)
        if row.get("price_stage") not in SOURCES or (
            row.get("entry_timestamp_verified")!="false"):
            raise ValueError("HISTORICAL_SOURCE_STAGE_OR_TIME_NOT_SUPPORTED")
        p=_number(row["model_probability"],min_value=0,max_value=1)
        ev=_number(row["expected_value_per_unit"])
        net=_number(row["net_units"])
        result=row.get("result")
        if result not in ("win","loss","push"):
            raise ValueError("UNMATURED_HISTORICAL_RESULT")
        if result=="loss" and abs(net+1)>1e-9:
            raise ValueError("BAD_LOSS_SETTLEMENT")
        if result=="push" and abs(net)>1e-9:
            raise ValueError("BAD_PUSH_SETTLEMENT")
        selected.append((season,gid,row["side"],p,ev,net,result))

    if not selected:raise ValueError("NO_HISTORICAL_ARCHIVE")
    report={}
    for season in sorted(YEARS):
        all_rows=[r for r in selected if r[0]==season]
        raw_positive=[r for r in all_rows if r[4]>0]
        over_bin=[r for r in all_rows if r[2]=="over" and BAND_LOW<=r[3]<BAND_HIGH]
        def summarize(rows):
            n=len(rows)
            decided=sum(x[6]!="push" for x in rows)
            wins=sum(x[6]=="win" for x in rows)
            return {
              "games":n,"decided":decided,"wins":wins,
              "losses":sum(x[6]=="loss" for x in rows),
              "pushes":sum(x[6]=="push" for x in rows),
              "archived_model_average_probability":sum(x[3] for x in rows)/n if n else None,
              "actual_win_rate_decided":wins/decided if decided else None,
              "actual_decided_win_wilson_95":_wilson(wins,decided),
              "archived_hypothetical_flat_unit_roi":sum(x[5] for x in rows)/n if n else None,
              "raw_model_mean_ev":sum(x[4] for x in rows)/n if n else None,
              "independently_verified_accepted_bookmaker_wagers":0,
            }
        report[str(season)]={
            "all_eligible_total_candidate_rows":summarize(all_rows),
            "raw_model_ev_positive_unreliable":summarize(raw_positive),
            "older_model_over_probability_55_60pct":summarize(over_bin),
        }
    return {
        "spec":SPEC,"status":"NO_ARCHIVED_TOTAL_PROFITABLE_EDGE_VERIFIED",
        "data_type":"PROVIDER_LABELED_OPEN_CLOSE_HISTORICAL_RESEARCH",
        "original_sportsbook_offer_timestamps_verified":False,
        "past_account_execution_confirmed":False,
        "probability_bin":{"lower_inclusive":BAND_LOW,"upper_exclusive":BAND_HIGH},
        "candidate_model_family":"ARCHIVED_CONDITIONAL_STUDENT_T_NOT_CURRENT_JOINT_DRIVE",
        "future_joint_drive_simulator_calibrated_by_this_audit":False,
        "declared_analysis":"Fixed bin/positive-raw-EV diagnostics for 2024 and 2025; NO threshold optimization.",
        "season_results":report,
        "edge_promoted":False,"wager_authorized":False,
        "minimum_future_independent_distinct_games":128,
        "next_gate":"Distinct pregame first-seen 2026+ source + same-contract line/price + 72h mature official result + week-clustered proper scoring and return benchmark",
        "action":"PASS_RESEARCH_ONLY",
    }

def run(input_path="reports/verified_market_bets.csv",
        output_path="reports/total_edge_reliability_audit.json"):
    with Path(input_path).open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    result=audit(rows)
    out=Path(output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    return result

if __name__=="__main__":
    a=run()
    print(json.dumps({"status":a["status"],
      "by_season":{year:{
         "all_games":x["all_eligible_total_candidate_rows"]["games"],
         "raw_positive_game_count":x["raw_model_ev_positive_unreliable"]["games"],
         "raw_positive_hypothetical_roi":x["raw_model_ev_positive_unreliable"]["archived_hypothetical_flat_unit_roi"],
         "over_55_60_count":x["older_model_over_probability_55_60pct"]["games"],
         "over_55_60_actual_rate":x["older_model_over_probability_55_60pct"]["actual_win_rate_decided"],
       } for year,x in a["season_results"].items()},
       "wager_authorized":False},sort_keys=True))
