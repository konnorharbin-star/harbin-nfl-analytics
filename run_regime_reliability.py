from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.regime_reliability import build_regime_reliability_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NFL probability and edge reliability by fixed regimes."
    )
    parser.add_argument("--bets", default="reports/verified_market_bets.csv")
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument(
        "--output",
        default="reports/regime_edge_reliability.json",
    )
    parser.add_argument(
        "--docs-output",
        default="docs/regime_edge_reliability.json",
    )
    args = parser.parse_args()

    bets = pl.read_csv(args.bets, try_parse_dates=True)
    report = build_regime_reliability_report(
        bets,
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    for path in (args.output, args.docs_output):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
