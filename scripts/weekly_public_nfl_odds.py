"""Chronological public NFL odds journal across *all upcoming* regular games.

Does not predict wins, use bookmaker credentials, infer quote freshness from
collector timing, or place wagers. Fetches no-key public Action Network data,
saves first-seen observation-time snapshots and rejected-origin diagnostics.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import polars as pl
from nfl.action_network import ActionNetworkNFLClient
from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc
from scripts.professional_method_audit import _utc
from scripts.public_quote_capture_journal import capture, append_file

ORIGIN = "https://api.actionnetwork.com/web/v2/scoreboard/nfl"
VERSION = "NFL_WEEKLY_PUBLIC_CAPTURE_V1"


def upcoming(rows, *, now, hours=192, max_games=32):
    at=_utc(now)
    if not (0 < hours <= 336) or not (1 <= max_games <= 64):
        raise ValueError("invalid capture horizon or limit")
    found=[]
    seen=set()
    for g in rows:
        gid=str(g.get("game_id") or "")
        if not gid or gid in seen:
            raise ValueError("missing or duplicated NFL game ID")
        seen.add(gid)
        if g.get("game_type")!="REG":
            continue
        if g.get("home_score") is not None or g.get("away_score") is not None:
            continue
        kick=schedule_kickoff_utc(g)
        if kick is None:
            continue
        if not (at<kick<=at+timedelta(hours=hours)):
            continue
        found.append({
            "game_id":gid,"season":int(g["season"]),"week":int(g["week"]),
            "home_team":str(g["home_team"]),"away_team":str(g["away_team"]),
            "kickoff_utc":kick.isoformat(),
        })
    found.sort(key=lambda x:(x["kickoff_utc"],x["game_id"]))
    return found[:max_games]


def collect(games, fetch_week, *, observed, journal=None):
    """fetch_week(season,week,games)-> source-two-way records, observed iso, diagnostics.
    Fail independently for each week; never synthesize market rows after source failure.
    """
    at=_utc(observed)
    if any(_utc(g["kickoff_utc"])<=at for g in games):
        raise ValueError("no postkickoff game may enter forward collection")
    weeks={}
    for g in games:
        weeks.setdefault((g["season"],g["week"]),[]).append(g)
    output={
        "version":VERSION, "decision":"PASS",
        "collector_run_started_utc":at.isoformat(),
        "target_games":len(games),"games_with_capture":0,
        "raw_paired_rows":0,"captured_paired_rows":0,
        "origin_verified_paired_rows":0,"rejections":[],
        "captures":[],"sources_by_week":[],
        "approved_bets":0,"validated_edge":False,
    }
    for (season,week), targets in sorted(weeks.items()):
        try:
            markets,observed_source,diagnostic=fetch_week(season,week,targets)
            exact_observed=_utc(observed_source)
            if exact_observed < at or exact_observed > at+timedelta(minutes=5):
                raise ValueError("invalid fetch observation time")
            if any(exact_observed>=_utc(x["kickoff_utc"]) for x in targets):
                # Individual games might have kicked off during the read; let
                # remaining games be collected but skip those already started.
                targets=[x for x in targets if exact_observed<_utc(x["kickoff_utc"])]
            output["raw_paired_rows"]+=len(markets)
            output["sources_by_week"].append({
                "season":season,"week":week,
                "collector_observed_utc":exact_observed.isoformat(),
                "diagnostic":diagnostic,"raw_rows":len(markets),
            })
            for target in targets:
                subset=[m for m in markets if str(m.game_id)==target["game_id"]]
                if not subset:
                    continue
                entry=capture(
                    subset,event_id=target["game_id"],
                    kickoff_utc=target["kickoff_utc"],
                    observed_utc=exact_observed.isoformat(),
                    source_url=ORIGIN,
                )
                if entry["paired_public_market_count"]==0:
                    output["rejections"].append({
                        "event_id":target["game_id"],
                        "reason":"NO_VALID_TWO_SIDED_MARKET_ROWS",
                    })
                    continue
                if journal is not None:
                    append_file(journal,entry)
                output["captures"].append({
                    "event_id":target["game_id"],
                    "kickoff_utc":target["kickoff_utc"],
                    "observed_utc":entry["collector_observed_utc"],
                    "snapshot_sha256":entry["snapshot_sha256"],
                    "public_book_pairs":entry["paired_public_market_count"],
                    "original_update_times":entry["bookmaker_original_quote_times_available"],
                    "account_accepted_quotes":0,
                })
                output["games_with_capture"]+=1
                output["captured_paired_rows"]+=entry["paired_public_market_count"]
                output["origin_verified_paired_rows"]+=entry["bookmaker_original_quote_times_available"]
        except (ValueError,KeyError,TypeError,RuntimeError) as exc:
            output["rejections"].append({
                "season":season,"week":week,
                "reason":"SOURCE_WEEK_FAILED",
                "error":f"{type(exc).__name__}: {str(exc)[:220]}",
            })
    output["sources_with_verified_original_book_times"]=(
        output["origin_verified_paired_rows"]
    )
    output["source_capture_only_not_offer_origin"]=True
    return output


def source_fetcher(client):
    def read(season,week,games):
        targets=pl.DataFrame([
            {"season":season,"game_id":g["game_id"],"home_team":g["home_team"],
             "away_team":g["away_team"]} for g in games
        ])
        raw=client.current_markets(targets,week=week)
        observed=datetime.now(timezone.utc).isoformat()
        return raw, observed, client.last_diagnostic
    return read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season",required=True,type=int)
    p.add_argument("--hours",type=int,default=192)
    p.add_argument("--max-games",type=int,default=32)
    p.add_argument("--history",type=Path,
                   default=Path("history/public_aggregator_quotes_2026.jsonl"))
    p.add_argument("--report",type=Path,
                   default=Path("reports/weekly_public_quote_capture.json"))
    p.add_argument("--refresh-schedules",action="store_true")
    args=p.parse_args()
    now=datetime.now(timezone.utc).isoformat()
    schedules=NFLDataClient().load_schedules(
        [args.season],refresh=args.refresh_schedules
    )
    selected=upcoming(schedules.iter_rows(named=True),
                      now=now,hours=args.hours,max_games=args.max_games)
    if selected:
        result=collect(
            selected,source_fetcher(ActionNetworkNFLClient()),
            observed=now,journal=args.history
        )
    else:
        result=collect([],lambda *args: ([],now,{}),observed=now)
        result["rejections"].append({"reason":"NO_ELIGIBLE_FUTURE_GAME"})
    result["target_game_rows"] = selected
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps({
        "decision":result["decision"],
        "target_games":result["target_games"],
        "games_with_capture":result["games_with_capture"],
        "captured_paired_rows":result["captured_paired_rows"],
        "original_book_time_verified_pairs":result["origin_verified_paired_rows"],
        "betmgm_account_prices_confirmed":0,
        "validated_positive_ev_bets":0,
        "rejections":result["rejections"][:8],
    },indent=2))
if __name__=="__main__":
    main()
