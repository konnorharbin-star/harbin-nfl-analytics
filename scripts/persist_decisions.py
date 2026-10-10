"""Archive current release-gated decisions; no wager execution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.final_edge_board import build
from scripts.recommendation_ledger import archive_decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("reports/release_gate.json"))
    parser.add_argument("--coverage-source", type=Path, default=Path("docs/latest.csv"))
    parser.add_argument("--root", type=Path, default=Path("history/recommendations_v1"))
    parser.add_argument("--summary", type=Path, default=Path("docs/recommendation_status.json"))
    args = parser.parse_args()
    now = datetime.now(UTC)
    gate = json.loads(args.gate.read_text())
    with args.source.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    if args.coverage_source.exists():
        with args.coverage_source.open(newline="") as handle:
            coverage = list(csv.DictReader(handle))
        represented = {str(r.get("game_id")) for r in source}
        for row in coverage:
            if str(row.get("game_id")) in represented:
                continue
            if args.sport == "cfb":
                row.update(
                    {
                        "market": row.get("quant_market"),
                        "side": row.get("quant_side"),
                        "line": row.get("quant_price"),
                        "odds": row.get("quant_odds"),
                        "book": row.get("quant_book"),
                        "quote_at": row.get("quant_quote_at"),
                        "probability": row.get("quant_probability"),
                        "ev": row.get("quant_ev"),
                    }
                )
            source.append(row)
    rows, board = build(source, sport=args.sport, gate=gate, as_of=now)
    receipts = [archive_decision(row, gate=gate, root=args.root, observed_at=now) for row in rows]
    payload = {
        "observed_at": now.isoformat(),
        "sport": args.sport,
        "release_state": gate.get("release_state"),
        "release_blockers": gate.get("blockers", []),
        "board": board,
        "decision_counts": dict(Counter(r["decision"] for r in receipts)),
        "receipt_ids": [r["bet_id"] for r in receipts],
        "publication_requirement": "Verify receipt bytes in GitHub commit before communicating BET",
        "actual_wagers": 0,
        "grading_status": "Scheduled immutable receipt grading; research ledger remains separate",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
