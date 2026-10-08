"""CLI for NFL historical edge failure research; never changes betting policy."""
from __future__ import annotations

import argparse
import json

import polars as pl

from nfl.historical_edge_diagnostics import (
    _json,
    build_historical_edge_failure_report,
    write_historical_edge_failure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NFL historical raw EV and probability calibration versus market."
    )
    parser.add_argument("--bets", default="reports/verified_market_bets.csv")
    parser.add_argument("--archive", default="reports/verified_market_backtest.json")
    parser.add_argument("--shrinkage", default="reports/market_edge_shrinkage.json")
    parser.add_argument("--regime", default="reports/regime_edge_reliability.json")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    bets = pl.read_csv(args.bets, infer_schema_length=1000, try_parse_dates=False)
    report = build_historical_edge_failure_report(
        bets,
        archive_backtest=_json(args.archive),
        market_shrinkage=_json(args.shrinkage),
        regime_reliability=_json(args.regime),
    )
    write_historical_edge_failure(
        report,
        report_dir=args.reports_dir,
        outputs_dir=args.outputs_dir,
        docs_dir=args.docs_dir,
    )
    print(json.dumps({
        "status": report["status"],
        "research_only": report["research_only"],
        "source_rows": report["data_integrity"]["source_rows"],
        "included_rows": report["data_integrity"]["included_rows"],
        "entry_timestamp_verified_rows": (
            report["data_integrity"]["entry_timestamp_verified_rows"]
        ),
        "model_minus_market_brier": report["overall"]["model_minus_market_brier"],
        "mean_raw_ev": report["overall"]["mean_expected_value_per_unit"],
        "hypothetical_archive_roi": report["overall"]["archive_roi_per_unit"],
        "market_failures": report["by_market_failure"],
    }, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
