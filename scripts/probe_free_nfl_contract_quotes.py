"""No-fee NFL odds probe: original book quote times vs public captured quotes.

A public API response can be recorded at collector time even if its sportsbook
origin-update time is unknown. Captured source data is NOT an executable
BetMGM U.S. offer. No model betting advice, no account access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import polars as pl

from nfl.action_network import ActionNetworkNFLClient
from scripts.action_network_contract_feed import probe
from scripts.public_quote_capture_journal import capture, append_file

SOURCE_URL = "https://api.actionnetwork.com/web/v2/scoreboard/nfl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--kickoff", required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/free_source_probe.json"))
    parser.add_argument("--quotes", type=Path, default=Path("reports/free_source_quotes.json"))
    parser.add_argument("--capture-history", type=Path, default=None)
    args = parser.parse_args()

    observed = datetime.now(timezone.utc).isoformat()
    targets = pl.DataFrame([{
        "season": args.season, "game_id": args.event,
        "home_team": args.home, "away_team": args.away,
    }])
    client = ActionNetworkNFLClient()
    failures = []
    try:
        markets = client.current_markets(targets, week=args.week)
    except Exception as error:
        markets = []
        failures.append(f"{type(error).__name__}: {str(error)[:350]}")

    strict_quotes, origin_report = probe(markets, {args.event: args.kickoff}, observed)
    captured = capture(
        markets, event_id=args.event, kickoff_utc=args.kickoff,
        observed_utc=observed, source_url=SOURCE_URL,
    )
    if args.capture_history is not None:
        origin_report["capture_archive_record_number"] = append_file(
            args.capture_history, captured
        )
    origin_report["capture_only_snapshot_sha256"] = captured["snapshot_sha256"]
    origin_report["capture_only_public_market_count"] = captured["paired_public_market_count"]
    origin_report["source_call_errors"] = failures
    origin_report["source_parser_diagnostics"] = client.last_diagnostic
    origin_report["source_raw_two_sided_rows"] = len(markets)
    origin_report["target_game_id"] = args.event
    origin_report["target_kickoff_utc"] = args.kickoff
    origin_report["betmgm_us_account_offer_confirmed"] = False

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.quotes.parent.mkdir(parents=True, exist_ok=True)
    capture_file = args.report.with_name("free_source_capture.json")
    capture_file.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(origin_report, indent=2, sort_keys=True, default=str) + "\n")
    args.quotes.write_text(json.dumps(strict_quotes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": "PASS", "observed_utc": observed,
        "raw_two_way_pairs": len(markets),
        "capture_only_pairs": captured["paired_public_market_count"],
        "source_origin_fresh_pairs": origin_report["eligible_quote_pairs"],
        "source_errors": failures,
        "diagnostic": client.last_diagnostic,
        "account_offer_accepted": False,
        "verified_profitable_edge": False,
        "wager_placed": False,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
