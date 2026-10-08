"""Audit the availability and internal consistency of NFL forward forecast evidence.

Read-only. Never revises predictions, generates bets, or declares profitability.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def before_kickoff(row):
    try:
        capture = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
        kickoff = datetime.fromisoformat(row["kickoff"].replace("Z", "+00:00"))
        return capture.tzinfo is not None and kickoff.tzinfo is not None and capture < kickoff
    except (ValueError, KeyError, TypeError):
        return False


def audit(records):
    groups = {}
    for r in records:
        game_id = r.get("game_id", "")
        if game_id:
            groups.setdefault(game_id, []).append(r)

    settled = [r for r in records if r.get("observation_status") not in ("", "PENDING_RESULT", None)
               and r.get("result") not in ("", None)]
    score_rows = [r for r in records if r.get("home_score") not in ("", None)
                  and r.get("away_score") not in ("", None)]
    valid = [r for r in records if before_kickoff(r)]
    # One forecast snapshot per game and market, no row weighting by sportsbook multiplicity.
    keys = [(r.get("game_id"), r.get("market")) for r in valid]
    duplicate_keys = sum(n - 1 for n in Counter(keys).values() if n > 1)
    return {
        "status": "SCORED_EVIDENCE_AVAILABLE" if score_rows else "AWAITING_COMPLETED_GAMES",
        "total_market_rows": len(records),
        "unique_games": len(groups),
        "timestamp_valid_pregame_rows": len(valid),
        "invalid_or_late_timestamp_rows": len(records) - len(valid),
        "duplicate_game_market_rows": duplicate_keys,
        "result_labeled_rows": len(settled),
        "rows_with_final_scores": len(score_rows),
        "market_counts": dict(sorted(Counter(r.get("market", "unknown") for r in records).items())),
        "accuracy_claim_permitted": False,
        "notes": [
            "This audit checks evidence readiness only; it does not score forecasts.",
            "Market-side probabilities are not independent game-level win predictions.",
            "A proper score-accuracy comparison requires point-in-time margin/total forecasts joined to verified final scores.",
            "Do not backfill a newer model's predictions and label them historical pregame forecasts."
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/forward_edge_graded.csv")
    parser.add_argument("--output", default="reports/accuracy_evidence_readiness.json")
    args = parser.parse_args()
    data = audit(rows(args.input))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
