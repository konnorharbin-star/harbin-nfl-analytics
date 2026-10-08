"""Audit ROI provenance of NFL historical bet candidates."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nfl.bet_evidence_audit import audit_bet_evidence


def main() -> None:
    bets = pl.read_csv("reports/free_market_bets.csv")
    report = audit_bet_evidence(bets)
    path = Path("reports/bet_evidence_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for market, cohorts in report["targets"].items():
        for threshold, outcome in cohorts.items():
            print(market, threshold, outcome["research_bets"],
                  outcome["verified_entry_bets"])


if __name__ == "__main__":
    main()
