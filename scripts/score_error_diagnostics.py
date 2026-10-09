"""Deterministic historical score-error decomposition. No model fitting or bets.

Use only with archived predictions and settled results. A retrospective row without
its original forecast timestamp is NOT verified prospective evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

SCHEMAS = {
    "nfl": {
        "margin": "projected_home_margin",
        "total": "projected_total",
        "actual_margin": "actual_home_margin",
        "actual_total": "actual_total",
        "total_low": 42.0,
        "total_high": 52.0,
    },
    "cfb": {
        "margin": "pred_margin_home",
        "total": "pred_total",
        "actual_margin": "actual_margin_home",
        "actual_total": "actual_total",
        "total_low": 45.0,
        "total_high": 65.0,
    },
}


def _num(record: dict[str, str], name: str) -> float:
    try:
        number = float(record[name])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid {name} for {record.get('game_id')}") from error
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {name} for {record.get('game_id')}")
    return number


def load_games(filename: str | Path, sport: str) -> list[dict]:
    """Enforce one row per game; derive home/away errors algebraically."""
    config = SCHEMAS[sport]
    required = {
        "game_id", "season", "week", "home_team", "away_team",
        config["margin"], config["total"], config["actual_margin"],
        config["actual_total"],
    }
    with Path(filename).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns: " + ", ".join(
                sorted(required - set(reader.fieldnames or []))
            ))
        data = list(reader)
    if not data:
        raise ValueError("No historical games supplied")
    output, seen = [], set()
    for row in data:
        key = str(row["game_id"]).strip()
        if not key or key in seen:
            raise ValueError(f"Duplicate or missing game identity: {key!r}")
        seen.add(key)
        if not row["home_team"] or not row["away_team"]:
            raise ValueError("Home and away team identities are required")
        try:
            season = int(row["season"])
            week = int(row["week"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid season/week for {key}") from error
        if season < 2000 or not 1 <= week <= 25:
            raise ValueError(f"Invalid season/week for {key}")
        pm = _num(row, config["margin"])
        pt = _num(row, config["total"])
        am = _num(row, config["actual_margin"])
        at = _num(row, config["actual_total"])
        if pt < 0 or at < 0 or abs(am) > at:
            raise ValueError(f"Impossible total/margin score pair for {key}")
        ph, pa = (pt + pm) / 2, (pt - pm) / 2
        ah, aa = (at + am) / 2, (at - am) / 2
        margin_err, total_err = pm - am, pt - at
        output.append({
            "game_id": key, "season": season, "week": week,
            "home_team": row["home_team"], "away_team": row["away_team"],
            "projected_margin": pm, "actual_margin": am,
            "projected_total": pt, "actual_total": at,
            "projected_home": ph, "projected_away": pa,
            "actual_home": ah, "actual_away": aa,
            "margin_error": margin_err, "total_error": total_err,
            "home_error": ph - ah, "away_error": pa - aa,
            "error_pattern": (
                "TOTAL_DOMINANT" if abs(total_err) >= abs(margin_err)
                else "MARGIN_DOMINANT"
            ),
            "predicted_total_band": (
                "LOW" if pt <= config["total_low"] else
                "MID" if pt <= config["total_high"] else "HIGH"
            ),
            "predicted_margin_band": (
                "CLOSE_0_3" if abs(pm) <= 3 else
                "MODERATE_3_7" if abs(pm) <= 7 else "LARGE_7_PLUS"
            ),
        })
    return output


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"games": 0}
    m = [r["margin_error"] for r in rows]
    t = [r["total_error"] for r in rows]
    h = [r["home_error"] for r in rows]
    a = [r["away_error"] for r in rows]
    winner = [
        int(r["projected_margin"] * r["actual_margin"] > 0)
        for r in rows if r["projected_margin"] and r["actual_margin"]
    ]
    return {
        "games": len(rows),
        "margin_mae": _mean([abs(v) for v in m]),
        "total_mae": _mean([abs(v) for v in t]),
        "home_points_mae": _mean([abs(v) for v in h]),
        "away_points_mae": _mean([abs(v) for v in a]),
        "margin_bias": _mean(m),
        "total_bias": _mean(t),
        "home_points_bias": _mean(h),
        "away_points_bias": _mean(a),
        "margin_rmse": math.sqrt(_mean([v * v for v in m])),
        "total_rmse": math.sqrt(_mean([v * v for v in t])),
        "margin_error_14_plus_rate": _mean([abs(v) >= 14 for v in m]),
        "margin_error_21_plus_rate": _mean([abs(v) >= 21 for v in m]),
        "total_error_20_plus_rate": _mean([abs(v) >= 20 for v in t]),
        "total_error_30_plus_rate": _mean([abs(v) >= 30 for v in t]),
        "winner_accuracy_excluding_ties": _mean(winner),
        "winner_sample": len(winner),
    }


def report(rows: list[dict], *, sport: str, source_sha256: str) -> dict:
    by_season = defaultdict(list)
    by_total = defaultdict(list)
    by_margin = defaultdict(list)
    for row in rows:
        by_season[str(row["season"])].append(row)
        by_total[row["predicted_total_band"]].append(row)
        by_margin[row["predicted_margin_band"]].append(row)
    def summarize(groups: dict) -> dict:
        return {name: metrics(games) for name, games in sorted(groups.items())}
    return {
        "spec": "score_error_decomposition_v1",
        "sport": sport,
        "evidence_class": "RETROSPECTIVE_HISTORICAL_REPORT_NOT_VERIFIED_FORWARD",
        "source_sha256": source_sha256,
        "overall": metrics(rows),
        "by_season": summarize(by_season),
        "by_predicted_total_band": summarize(by_total),
        "by_predicted_margin_band": summarize(by_margin),
        "diagnostic_limits": [
            "These rows lack verified pregame forecast capture timestamps.",
            "Errors alone do not establish causal injuries, pace, play calling, or QB impacts.",
            "Do not fit coefficients to this report or treat these metrics as forward betting performance.",
            "Do not mix these games with duplicate forward snapshots or claim newly approved bets.",
        ],
        "production_model_adjustment_points": 0.0,
    }


def extreme_games(rows: list[dict], top_n: int = 20) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda r: (
            -max(abs(r["margin_error"]), abs(r["total_error"])),
            -abs(r["total_error"]),
            r["game_id"],
        ),
    )
    return ranked[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", required=True, choices=sorted(SCHEMAS))
    parser.add_argument("--source", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--tails-output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    rows = load_games(source, args.sport)
    output = report(
        rows,
        sport=args.sport,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    jpath = Path(args.json_output)
    cpath = Path(args.tails_output)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    tails = extreme_games(rows)
    with cpath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tails[0]))
        writer.writeheader()
        writer.writerows(tails)
    print(json.dumps({
        "sport": args.sport,
        "games": output["overall"]["games"],
        "margin_mae": output["overall"]["margin_mae"],
        "total_mae": output["overall"]["total_mae"],
        "status": output["evidence_class"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
