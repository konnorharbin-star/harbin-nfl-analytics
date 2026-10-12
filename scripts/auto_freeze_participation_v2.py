"""Freeze v2 pregame NFL player probabilities for EVERY approaching NFL week.

2024 residual fit and 2025 already-inspected diagnostics are reused by the
v2 command. This wrapper ONLY selects season/week from public upcoming games,
then runs the unchanged fixed v2 research model. It NEVER changes coefficients,
fits on 2026 outcomes, overwrites previously frozen quotes, or authorizes bets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc
from scripts.professional_method_audit import _utc
from scripts.participation_conditional_prop_v2 import run

VERSION="V2_AUTOFREEZE_FUTURE_NFL_WEEKS_V1"


def future_weeks(schedules, *, observed_utc, season=2026, horizon_days=10):
    at=_utc(observed_utc)
    if season!=2026:
        raise ValueError("v2 training lineage only registered for season 2026")
    if not 0<horizon_days<=21:
        raise ValueError("horizon must be 1 to 21 days")
    grouped={}
    for game in schedules:
        if int(game.get("season",0))!=season:
            continue
        if game.get("game_type")!="REG" or not game.get("game_id"):
            continue
        if game.get("home_score") is not None or game.get("away_score") is not None:
            continue
        kickoff=schedule_kickoff_utc(game)
        if kickoff is None or not at<kickoff<=at+timedelta(days=horizon_days):
            continue
        week=int(game["week"])
        if not 1<=week<=18: raise ValueError("invalid week in schedule")
        grouped.setdefault(week,[]).append({
            "game_id":str(game["game_id"]),
            "kickoff_utc":kickoff.isoformat(),
        })
    return [{
        "week":week,
        "future_games":len(grouped[week]),
        "first_game_utc":min(g["kickoff_utc"] for g in grouped[week]),
    } for week in sorted(grouped)]


def freeze_upcoming(*, schedules, observed_utc, season, horizon_days,
                    freeze_path, runner=run):
    groups=future_weeks(schedules,observed_utc=observed_utc,
                        season=season,horizon_days=horizon_days)
    rows=[]
    total=0
    for selected in groups:
        # Original run() will recompute strictly historical 2024 fitting, but
        # the model and feature choice are frozen across weeks.
        report=runner(
            season=season,week=selected["week"],freeze_path=freeze_path,
            report_path=None,
        )
        if report.get("decision")!="PASS" or report.get("approved_betting_model") or not report.get("2026_not_used_for_fit"):
            raise ValueError("research-only v2 run contract violated")
        n=int(report["forward_frozen_records"]["new"])
        total+=n
        rows.append({
            **selected,"first_seen_added":n,
            "model_version":report["version"],
            "no_bookmaker_prices":report.get("current_account_betmgm_prices") is None,
        })
    return {
        "version":VERSION,"decision":"PASS",
        "schedule_screen_observed_utc":_utc(observed_utc).isoformat(),
        "year":season,"lookahead_days":horizon_days,
        "eligible_upcoming_weeks":len(groups),
        "first_seen_unique_synthetic_forecasts_added":total,
        "upcoming_week_results":rows,
        "original_predictions_never_overwritten":True,
        "no_2026_outcomes_used_to_fit_calibration":True,
        "account_verified_betmgm_prices":False,
        "approved_betting_model":False,"actual_bet_count":0,
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--horizon-days",type=int,default=10)
    ap.add_argument("--freeze",type=Path,
                    default=Path("history/prop_participation_shadow_v2_2026.csv"))
    ap.add_argument("--report",type=Path,
                    default=Path("reports/auto_freeze_participation_v2.json"))
    args=ap.parse_args()
    if args.season!=2026:
        ap.error("v2 frozen study is limited to registered 2026 football season")
    source=NFLDataClient().load_schedules([args.season],refresh=True).to_dicts()
    observed=datetime.now(timezone.utc).isoformat()
    result=freeze_upcoming(
        schedules=source,observed_utc=observed,season=args.season,
        horizon_days=args.horizon_days,freeze_path=args.freeze,
    )
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
