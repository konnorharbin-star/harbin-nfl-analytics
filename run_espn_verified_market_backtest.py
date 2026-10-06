from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.espn_historical import (
    ESPNArchiveClient,
    build_espn_archive_bets,
    build_espn_archive_report,
    fetch_espn_archive_quotes,
)
from nfl.free_market_backtest import build_archive_projection_dataset
from nfl.policy_calibration import derive_production_policy
from nfl.proof import write_evidence_report


def schedule_seasons_for_evidence(
    start_season: int,
    end_season: int,
) -> list[int]:
    """Load enough seasons for point-in-time probability validation."""

    return list(range(start_season - 5, end_season + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build verified NFL historical evidence from free ESPN archived "
            "provider opening and closing snapshots."
        )
    )
    parser.add_argument("--start-season", type=int, default=2024)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.start_season >= args.end_season:
        raise SystemExit("--start-season must be earlier than --end-season")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    schedules = NFLDataClient().load_schedules(
        schedule_seasons_for_evidence(
            args.start_season,
            args.end_season,
        ),
        refresh=args.refresh,
    )
    projections = build_archive_projection_dataset(
        schedules,
        start_season=args.start_season - 4,
        end_season=args.end_season,
    )
    evidence_targets = projections.filter(
        pl.col("season") >= args.start_season
    )
    entries, closings, coverage = fetch_espn_archive_quotes(
        evidence_targets,
        client=ESPNArchiveClient(max_workers=args.max_workers),
    )
    if entries.is_empty():
        raise SystemExit("ESPN archive produced no complete opening market pairs")
    if closings.is_empty():
        raise SystemExit("ESPN archive produced no complete closing market pairs")

    bets = build_espn_archive_bets(
        projections,
        entries,
        closings,
    )
    if bets.is_empty():
        raise SystemExit("ESPN archive backtest produced no verified bets")

    bets_path = reports / "verified_market_bets.csv"
    bets.write_csv(bets_path)

    report = build_espn_archive_report(bets, coverage)
    report_path = reports / "verified_market_backtest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
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
