"""No-fee Action Network original-offer-time probe for an upcoming NFL game.

This is an *availability measurement* of public attributed quotes, not proof
that BetMGM US lets an account place a wager. Saves origin/freshness exclusions.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import polars as pl

from nfl.action_network import ActionNetworkNFLClient
from scripts.action_network_contract_feed import probe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    ap.add_argument("--event",required=True)
    ap.add_argument("--home",required=True)
    ap.add_argument("--away",required=True)
    ap.add_argument("--kickoff",required=True)
    ap.add_argument("--report",type=Path,default=Path("reports/free_source_probe.json"))
    ap.add_argument("--quotes",type=Path,default=Path("reports/free_source_quotes.json"))
    a=ap.parse_args()
    time=datetime.now(timezone.utc).isoformat()
    targets=pl.DataFrame([dict(season=a.season,game_id=a.event,
                               home_team=a.home,away_team=a.away)])
    reader=ActionNetworkNFLClient()
    errors=[]
    try:
        raw=reader.current_markets(targets,week=a.week)
    except Exception as exc:
        raw=[]
        errors.append(f"{type(exc).__name__}: {str(exc)[:350]}")
    request,report=probe(raw,{a.event:a.kickoff},time)
    report["source_call_errors"]=errors
    report["source_parser_diagnostics"]=reader.last_diagnostic
    report["source_raw_two_sided_rows"]=len(raw)
    report["target_game_id"]=a.event
    report["target_kickoff_utc"]=a.kickoff
    report["fetch_is_not_authorized_to_impute_missing_offer_origin"]=True
    report["note"]="Aggregated sportsbook attribution; NOT real accepted BetMGM US ticket."
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.quotes.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n")
    a.quotes.write_text(json.dumps(request,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"decision":"PASS","observed_utc":time,
                      "source_raw_two_sided_rows":len(raw),
                      "original_time_fresh_eligible_pairs":report["eligible_quote_pairs"],
                      "bookmakers":report["bookmakers_seen"],
                      "source_errors":errors,
                      "source_diagnostic":reader.last_diagnostic,
                      "account_executable_verified":False,
                      "approved_model":False,
                      "no_wager_placed":True},indent=2,default=str))


if __name__=="__main__":
    main()
