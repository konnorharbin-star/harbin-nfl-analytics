from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.market_shrinkage import evaluate_market_edge_shrinkage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chronological NFL market-edge probability shrinkage."
    )
    parser.add_argument("--bets", default="reports/free_market_bets.csv")
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--output", default="reports/market_edge_shrinkage.json")
    parser.add_argument("--docs-output", default="docs/market_edge_shrinkage.json")
    args = parser.parse_args()

    bets = pl.read_csv(args.bets, try_parse_dates=True)
    report = evaluate_market_edge_shrinkage(
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
