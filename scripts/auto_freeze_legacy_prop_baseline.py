"""Freeze original unchanged 2026 v1 NFL player-prop benchmark in future weeks.

The v1 model is RETIRED as a betting signal, but a fair prospective v2-v1
comparison needs the ORIGINAL v1 source forecasts from BEFORE each kickoff.
This wrapper does not refit 2026 outcomes, change model v1, or bet.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from nfl.data import NFLDataClient
from run_player_prop_shadow import build_shadow_report
from scripts.auto_freeze_participation_v2 import future_weeks

VERSION="LEGACY_PROP_V1_PROSPECTIVE_BENCHMARK_FREEZE_V1"


def freeze_baseline(*, schedules, observed_utc, season, horizon_days,
                    runner=build_shadow_report):
    selected=future_weeks(schedules,observed_utc=observed_utc,
                          season=season,horizon_days=horizon_days)
    details=[]
    new=0
    for target in selected:
        # Reuse the original code, original 2024 development parameters,
        # and original journal; never use the new v2 probability distribution.
        result=runner(
            client=NFLDataClient(),
            season=season,week=target["week"],
            as_of=datetime.fromisoformat(observed_utc.replace("Z","+00:00")),
        )
        if (result.get("status")!="RESEARCH_ONLY_UNPRICED_PLAYER_FORECASTS" or
            result.get("actual_sportsbook_edge_proven") is not False):
            raise ValueError("original v1 research benchmark contract violated")
        exp=result.get("player_prop_probability_experiment",{})
        if exp.get("status")!="SYNTHETIC_LINES_DIAGNOSTIC_RESEARCH":
            raise ValueError("original v1 probability calibration blocked")
        forward=exp.get("forward_probability_rows",0)
        latest=result["forward_ledger"]
        if not isinstance(forward,int) or forward<0:
            raise ValueError("corrupted baseline count")
        new+=int(result["player_prop_probability_experiment"].get(
            "frozen_probability_new_entries",0
        ))
        details.append({
            **target,
            "fresh_player_market_point_forecasts":result["pregame_player_market_forecasts"],
            "v1_calibrated_synthetic_thresholds_in_capture":forward,
            "no_bookmaker_offer_verified":True,
            "v1_model_remains_benchmark_only":True,
        })
    return {
        "version":VERSION,"decision":"PASS",
        "season":season,"observed_utc":observed_utc,
        "future_weeks":len(details),
        "screened_weeks":details,
        "v1_legacy_unchanged":True,
        "reference_prices_or_independent_profitability_verified":False,
        "approved_betting_signals":0,
        "no_wager_placed":True,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--horizon-days",type=int,default=10)
    ap.add_argument("--report",type=Path,
                    default=Path("reports/legacy_v1_prop_benchmark_freeze.json"))
    a=ap.parse_args()
    if a.season!=2026:ap.error("legacy benchmark lineage registered for 2026 only")
    now=datetime.now(timezone.utc).isoformat()
    client=NFLDataClient()
    schedule=client.load_schedules([a.season],refresh=True).to_dicts()
    result=freeze_baseline(
        schedules=schedule,observed_utc=now,season=a.season,
        horizon_days=a.horizon_days,
    )
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
