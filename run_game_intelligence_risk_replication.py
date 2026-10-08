"""Audit risk-combination replication using the source-graded Phase 6 ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.game_intelligence_risk_replication import audit_risk_combinations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/game_intelligence_risk.csv")
    parser.add_argument(
        "--output", default="reports/game_intelligence_risk_replication.json"
    )
    args = parser.parse_args()
    games = pl.read_csv(args.input, try_parse_dates=True)
    report = audit_risk_combinations(games)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "status": report["status"],
        "replicated_margin": [
            name for name, value in report["targets"]["margin"].items()
            if value["status"] == "REPLICATED_DESCRIPTIVE_ASSOCIATION"
        ],
        "replicated_total": [
            name for name, value in report["targets"]["total"].items()
            if value["status"] == "REPLICATED_DESCRIPTIVE_ASSOCIATION"
        ],
    }))


if __name__ == "__main__":
    main()
