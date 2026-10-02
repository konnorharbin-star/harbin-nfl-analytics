from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.free_market_backtest import build_archive_projection_dataset
from nfl.market_history import select_market_snapshots_asof
from nfl.odds_api import (
    TheOddsAPIClient,
    closing_decisions_from_history,
    fetch_market_history_for_decisions,
)
from nfl.policy_calibration import derive_production_policy
from nfl.proof import write_evidence_report
from nfl.verified_market_backtest import (
    DEFAULT_CLOSE_MINUTES_BEFORE_KICKOFF,
    DEFAULT_ENTRY_MINUTES_BEFORE_KICKOFF,
    DEFAULT_MAX_QUOTE_AGE_MINUTES,
    build_market_decisions,
    build_verified_market_bets,
    eligible_projection_game_ids,
    write_verified_market_backtest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build promotion-quality NFL historical evidence from timestamped "
            "The Odds API snapshots."
        )
    )
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument(
        "--entry-minutes-before-kickoff",
        type=int,
        default=DEFAULT_ENTRY_MINUTES_BEFORE_KICKOFF,
    )
    parser.add_argument(
        "--close-minutes-before-kickoff",
        type=int,
        default=DEFAULT_CLOSE_MINUTES_BEFORE_KICKOFF,
    )
    parser.add_argument(
        "--max-quote-age-minutes",
        type=int,
        default=DEFAULT_MAX_QUOTE_AGE_MINUTES,
    )
    parser.add_argument("--regions", default="us")
    parser.add_argument("--cache-dir", default="data/odds_api_cache")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.start_season >= args.end_season:
        raise SystemExit("--start-season must be earlier than --end-season")
    if args.entry_minutes_before_kickoff <= args.close_minutes_before_kickoff:
        raise SystemExit(
            "--entry-minutes-before-kickoff must be greater than "
            "--close-minutes-before-kickoff"
        )
    if args.max_quote_age_minutes < 0:
        raise SystemExit("--max-quote-age-minutes must be >= 0")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    schedules = NFLDataClient().load_schedules(
        list(range(args.start_season - 1, args.end_season + 1)),
        refresh=args.refresh,
    )
    projections = build_archive_projection_dataset(
        schedules,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    eligible_ids = eligible_projection_game_ids(projections)
    entry_decisions = build_market_decisions(
        schedules,
        eligible_ids,
        minutes_before_kickoff=args.entry_minutes_before_kickoff,
    )

    regions = tuple(
        value.strip()
        for value in str(args.regions).split(",")
        if value.strip()
    )
    if not regions:
        raise SystemExit("--regions must include at least one region")

    provider = TheOddsAPIClient(cache_dir=args.cache_dir)
    entry_history = fetch_market_history_for_decisions(
        provider,
        schedules,
        entry_decisions,
        regions=regions,
        refresh=args.refresh,
    )
    selected_entry = select_market_snapshots_asof(
        entry_history,
        entry_decisions,
        max_quote_age_minutes=args.max_quote_age_minutes,
    )
    if selected_entry.is_empty():
        raise SystemExit("timestamped provider produced no eligible entry quotes")

    closing_decisions = closing_decisions_from_history(
        entry_history,
        minutes_before_kickoff=args.close_minutes_before_kickoff,
    )
    closing_history = fetch_market_history_for_decisions(
        provider,
        schedules,
        closing_decisions,
        regions=regions,
        refresh=args.refresh,
    )
    selected_closing = select_market_snapshots_asof(
        closing_history,
        closing_decisions,
        max_quote_age_minutes=args.max_quote_age_minutes,
    )

    bets = build_verified_market_bets(
        projections,
        selected_entry,
        selected_closing,
    )
    if bets.is_empty():
        raise SystemExit("timestamped provider backtest produced no verified bets")

    bets_path = reports / "verified_market_bets.csv"
    report_path = reports / "verified_market_backtest.json"
    report = write_verified_market_backtest(
        bets,
        entry_minutes_before_kickoff=args.entry_minutes_before_kickoff,
        close_minutes_before_kickoff=args.close_minutes_before_kickoff,
        max_quote_age_minutes=args.max_quote_age_minutes,
        bets_path=bets_path,
        report_path=report_path,
    )

    evidence = write_evidence_report(
        output=reports / "evidence_report.json",
        backtest_path=reports / "free_market_backtest.json",
        bets_path=reports / "free_market_bets.csv",
        verified_bets_path=bets_path,
    )
    policy = derive_production_policy(
        bets_path=reports / "free_market_bets.csv",
        verified_bets_path=bets_path,
        evidence_path=reports / "evidence_report.json",
        output_path=reports / "production_policy.json",
    )

    print(
        json.dumps(
            {
                "verified_market_backtest": report,
                "evidence_status": evidence["status"],
                "promotion_sample": evidence["promotion_sample"],
                "policy_mode": policy["deployment_mode"],
                "policy_source": policy["source"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
