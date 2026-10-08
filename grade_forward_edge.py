"""Grade locked NFL pre-kickoff model/market candidate pairs after final scores."""
from __future__ import annotations

import argparse
import json

from nfl.data import NFLDataClient, completed_games
from nfl.forward_edge_validation import (
    SEASON,
    grade_forward_candidates,
    load_forward_candidates,
    write_forward_validation,
)
from nfl.line_history import load_market_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL prospective market-relative grading of immutable quote pairs"
    )
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument(
        "--ledger", default="history/edge_forward_candidates_v1.csv"
    )
    parser.add_argument(
        "--market-history", default="history/market_snapshots.csv"
    )
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()
    if args.season != SEASON:
        raise SystemExit("spec version is frozen to the 2026 NFL season")

    frozen = load_forward_candidates(args.ledger)
    if frozen.is_empty():
        # Do not fetch anything and certainly do not synthesize historical
        # first-entry timestamps for an empty prospective cohort.
        graded, report = grade_forward_candidates(frozen, frozen)
    else:
        source = NFLDataClient()
        schedules = source.load_schedules([SEASON], refresh=True)
        finals = completed_games(schedules)
        if "game_type" in finals.columns:
            finals = finals.filter(finals.get_column("game_type") == "REG")
        market_history = load_market_snapshots(args.market_history)
        graded, report = grade_forward_candidates(
            frozen, finals, snapshots=market_history
        )
    write_forward_validation(
        graded, report,
        report_dir=args.reports_dir,
        docs_dir=args.docs_dir,
        outputs_dir=args.outputs_dir,
    )
    print(json.dumps({
        "status": report["status"],
        "frozen": report["summary"]["frozen"],
        "graded": report["summary"]["graded"],
        "decided": report["summary"]["decided"],
        "brier_lift_vs_market": report["summary"]["brier_lift_vs_market"],
        "eligible_for_staking": report["staking_authorized"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
