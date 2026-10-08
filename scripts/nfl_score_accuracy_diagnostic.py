"""Strict NFL pregame score accuracy evaluation; no changes to betting policy.

Inputs: frozen forecast snapshots with game_id, kickoff, captured_at,
model_margin_home, model_total; and independently graded final-score rows
with game_id, home_score, away_score. Multiple snapshots are reduced to the
earliest valid pre-kickoff snapshot, never retrospectively chosen by error.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def timestamp(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timezone required")
    return dt.astimezone(timezone.utc)


def number(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("nonfinite")
    return n


def analyze(forecasts, final_rows):
    finals = {}
    conflicted = set()
    for row in final_rows:
        gid = row.get("game_id")
        if not gid or row.get("home_score") in ("", None) or row.get("away_score") in ("", None):
            continue
        try:
            pair = (number(row["home_score"]), number(row["away_score"]))
            if any(x < 0 or int(x) != x for x in pair):
                continue
        except (ValueError, TypeError):
            continue
        if gid in finals and finals[gid] != pair:
            conflicted.add(gid)
        finals[gid] = pair
    for gid in conflicted:
        finals.pop(gid, None)

    by_game = defaultdict(list)
    rejected = defaultdict(int)
    for row in forecasts:
        gid = row.get("game_id")
        if not gid:
            rejected["missing_game_id"] += 1
            continue
        try:
            captured, kickoff = timestamp(row["captured_at"]), timestamp(row["kickoff"])
            margin, total = number(row["model_margin_home"]), number(row["model_total"])
            if captured >= kickoff:
                rejected["late_snapshot"] += 1
                continue
            if total < abs(margin) or total < 0:
                rejected["invalid_projection"] += 1
                continue
        except (ValueError, KeyError, TypeError, OverflowError):
            rejected["invalid_timestamp_or_forecast"] += 1
            continue
        by_game[gid].append((captured, margin, total, kickoff))

    evaluated = []
    for gid, snapshots in sorted(by_game.items()):
        if gid not in finals:
            continue
        snapshots.sort(key=lambda v: v[0])
        first = snapshots[0]
        # Same-earliest-time conflicting scores are ambiguous and ineligible.
        if any(s[0] == first[0] and (s[1], s[2]) != (first[1], first[2]) for s in snapshots):
            rejected["conflicting_earliest_snapshot"] += 1
            continue
        home, away = finals[gid]
        actual_margin, actual_total = home - away, home + away
        evaluated.append({
            "game_id": gid,
            "margin_error": first[1] - actual_margin,
            "total_error": first[2] - actual_total,
            "forecast_captured_at": first[0].isoformat(),
        })
    def metrics(field):
        errors = [r[field] for r in evaluated]
        if not errors:
            return {"mae": None, "rmse": None, "bias": None}
        return {
            "mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "bias": sum(errors) / len(errors),
        }
    return {
        "status": "EVALUATED" if evaluated else "INSUFFICIENT_MATCHED_FINAL_SCORES",
        "sample_games": len(evaluated),
        "forecast_games_with_valid_snapshots": len(by_game),
        "final_games_with_verified_scores": len(finals),
        "conflicting_final_games": len(conflicted),
        "rejected": dict(sorted(rejected.items())),
        "margin": metrics("margin_error"),
        "total": metrics("total_error"),
        "game_errors": evaluated,
        "notes": "Diagnostic only. No model promotion, edge validation, or betting approval.",
    }


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", required=True, help="Frozen, truly pregame model forecast CSV")
    parser.add_argument("--finals", default="outputs/forward_edge_graded.csv")
    parser.add_argument("--output", default="reports/nfl_score_accuracy_diagnostic.json")
    args = parser.parse_args()
    report = analyze(read_csv(args.forecasts), read_csv(args.finals))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "game_errors"}, indent=2))


if __name__ == "__main__":
    main()
