"""Retrospective NFL raw-EV calibration: a diagnostic, never a wagering policy.

Records are ESPN archived provider-labelled spread selections, not
independently timestamped, source-book-executable entry odds. 2025 was
previously inspected; it is NOT a pristine untouched holdout.
Compare each settled spread on the selected side to its paired no-vig
market probability. Tune market/log-odds-model blend using 2024 only.
All output is research only, no stakes, no odds collection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
EV_BINS = (
    ("below_0", float("-inf"), 0.0),
    ("0_to_5_pct", 0.0, 0.05),
    ("5_to_10_pct", 0.05, 0.10),
    ("10_to_20_pct", 0.10, 0.20),
    ("20_to_30_pct", 0.20, 0.30),
    ("30_pct_and_up", 0.30, float("inf")),
)
REQUIRED = {
    "market_type", "game_id", "season", "week", "result",
    "model_probability", "no_vig_probability", "expected_value_per_unit",
    "net_units", "historical_provenance", "entry_timestamp_verified",
    "probability_model_family",
}


def _float(value):
    if isinstance(value, bool):
        raise ValueError("Boolean numeric field")
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Missing or malformed numeric field") from exc
    if not math.isfinite(x):
        raise ValueError("Nonfinite probability, EV or payout")
    return x


def load(path):
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Missing archived spread research columns")
        for rec in reader:
            if rec["market_type"] != "spread":
                continue
            season, week = int(rec["season"]), int(rec["week"])
            if season not in (2024, 2025) or week < 1 or week > 22:
                raise ValueError("Spread archive must be 2024/2025 only")
            gid = str(rec["game_id"])
            identity = (season, gid)
            if not gid or identity in seen:
                raise ValueError("Missing or duplicate archived game entry")
            seen.add(identity)
            if rec["result"] not in {"win", "loss", "push"}:
                raise ValueError("Unknown settlement")
            m, b = _float(rec["model_probability"]), _float(
                rec["no_vig_probability"]
            )
            if not 0 < m < 1 or not 0 < b < 1:
                raise ValueError("Probability outside open interval (0,1)")
            ev, pnl = _float(rec["expected_value_per_unit"]), _float(
                rec["net_units"]
            )
            if not -1 <= ev <= 5 or not -1 <= pnl <= 10:
                raise ValueError("Out-of-range historical economic observation")
            if rec["result"] == "loss" and pnl != -1:
                raise ValueError("Loss did not settle at -1u")
            if rec["result"] == "push" and pnl != 0:
                raise ValueError("Push did not settle at 0u")
            if rec["result"] == "win" and pnl <= 0:
                raise ValueError("Winning bet payout was nonpositive")
            rows.append({
                "season": season, "week": week, "game_id": gid,
                "result": rec["result"], "model_p": m, "market_p": b,
                "raw_ev": ev, "net_units": pnl,
                "entry_timestamp_verified": rec["entry_timestamp_verified"] == "true",
                "provenance": rec["historical_provenance"],
                "family": rec["probability_model_family"],
            })
    if not rows:
        raise ValueError("No archived NFL spread selections")
    return rows


def blend(row, alpha):
    """Market log odds plus a 2024-selected fraction of model disagreement."""
    m, b = row["model_p"], row["market_p"]
    def logit(p):
        return math.log(p / (1 - p))
    x = logit(b) + alpha * (logit(m) - logit(b))
    return 1.0 / (1.0 + math.exp(-x))


def proper_scores(rows, alpha):
    resolved = [r for r in rows if r["result"] != "push"]
    if not resolved:
        raise ValueError("Cannot score an all-push or empty segment")
    n, brier, logloss, predicted, winners = len(resolved), 0., 0., 0., 0
    for r in resolved:
        p = blend(r, alpha)
        y = int(r["result"] == "win")
        brier += (p - y) ** 2
        logloss -= y * math.log(p) + (1 - y) * math.log(1 - p)
        predicted += p
        winners += y
    return {
        "resolved": n, "pushes_excluded": len(rows) - n,
        "brier": brier / n, "log_loss": logloss / n,
        "mean_probability": predicted / n,
        "empirical_win_rate": winners / n,
    }


def economic(rows):
    if not rows:
        return {"observations": 0, "raw_mean_ev": None, "archive_roi": None,
                "archive_profit_units": None, "wins": 0, "losses": 0, "pushes": 0}
    n = len(rows)
    return {
        "observations": n,
        "raw_mean_ev": sum(r["raw_ev"] for r in rows) / n,
        "archive_roi": sum(r["net_units"] for r in rows) / n,
        "archive_profit_units": sum(r["net_units"] for r in rows),
        "wins": sum(r["result"] == "win" for r in rows),
        "losses": sum(r["result"] == "loss" for r in rows),
        "pushes": sum(r["result"] == "push" for r in rows),
    }


def bootstrap(rows, *, repetitions=2000):
    """Paired week-cluster 95% simultaneous (two metrics), market minus model."""
    blocks = defaultdict(list)
    for r in rows:
        if r["result"] == "push":
            continue
        blocks[(r["season"], r["week"])].append(r)
    weeks = [blocks[k] for k in sorted(blocks)]
    if len(weeks) < 8:
        return {"week_clusters": len(weeks), "simultaneous_95_ci": None}
    rng = random.Random(20261010)
    diffs = {"brier": [], "log_loss": []}
    for _ in range(repetitions):
        sample = [r for _ in weeks for r in weeks[rng.randrange(len(weeks))]]
        market = proper_scores(sample, 0.0)
        model = proper_scores(sample, 1.0)
        for key in diffs:
            diffs[key].append(market[key] - model[key])
    for values in diffs.values():
        values.sort()
    out = {
        name: [value[int(0.0125 * (len(value) - 1))],
               value[int(0.9875 * (len(value) - 1))]]
        for name, value in diffs.items()
    }
    return {"week_clusters": len(weeks), "simultaneous_95_ci": out}


def report(rows, *, file_sha256=None):
    tune = [r for r in rows if r["season"] == 2024]
    test = [r for r in rows if r["season"] == 2025]
    if not tune or not test:
        raise ValueError("Must have both 2024 tune and 2025 evaluation records")
    if any(not r["provenance"] or r["entry_timestamp_verified"] for r in rows):
        raise ValueError("Expected provider-labelled, unverified historical snapshots")
    # Fit/tune exclusively 2024; ties prefer market (lower alpha).
    objective = {str(alpha): proper_scores(tune, alpha)["brier"] for alpha in ALPHAS}
    chosen = min(ALPHAS, key=lambda a: (objective[str(a)], a))
    comparisons = {
        "market_only": proper_scores(test, 0.0),
        "full_raw_model": proper_scores(test, 1.0),
        "locked_2024_blend": proper_scores(test, chosen),
    }
    bins = {
        name: economic([r for r in test if start <= r["raw_ev"] < end])
        for name, start, end in EV_BINS
    }
    # Overlapping tail buckets are explicit diagnostics, not extra wagers.
    tails = {
        "raw_ev_at_least_20_pct": economic(
            [r for r in test if r["raw_ev"] >= 0.20]
        ),
        "raw_ev_at_least_30_pct": economic(
            [r for r in test if r["raw_ev"] >= 0.30]
        ),
    }
    return {
        "spec": "walters_nfl_raw_ev_calibration_audit_v1",
        "status": "RETROSPECTIVE_DIAGNOSTIC_NO_VALIDATED_EDGE",
        "sport": "nfl", "selection": "one archived spread per game",
        "input_sha256": file_sha256,
        "input_model_families": sorted({r["family"] for r in rows}),
        "records": len(rows),
        "tuning_year": 2024, "evaluation_year": 2025,
        "was_2025_previously_inspected": True,
        "archive_entry_quote_time_independently_verified": False,
        "economic_backtest_actionable": False,
        "raw_model_is_current_2026_key_number_model": False,
        "alpha_grid": list(ALPHAS),
        "tuning_brier_by_alpha": objective,
        "chosen_alpha_2024_only": chosen,
        "2025_probability_scores": comparisons,
        "2025_spread_archive_roi_by_raw_ev_band": bins,
        "2025_spread_archive_large_ev_tails": tails,
        "2025_market_minus_full_model": {
            "brier": comparisons["market_only"]["brier"] - comparisons[
                "full_raw_model"
            ]["brier"],
            "log_loss": comparisons["market_only"]["log_loss"] - comparisons[
                "full_raw_model"
            ]["log_loss"],
        },
        "2025_paired_week_bootstrap": bootstrap(test),
        "2025_probability_selection_warning": (
            "Archive uses model-selected side, not a randomized side; published odds"
            " are provider-labelled historical records and not verified executable."
        ),
        "model_promotion_allowed": False,
        "betting_authorized": False,
        "wagers_placed": 0,
        "required_next_proof": (
            "Validate first-published 2026 exact-line quotes with independent"
            " bookmaker evidence, prospectively scored picks, closing reference"
            " and enough settled games to estimate economic performance."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("reports/verified_market_bets.csv")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/walters_nfl_raw_ev_calibration.json")
    )
    args = parser.parse_args()
    result = report(
        load(args.source),
        file_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "2024_selected_model_log_odds_weight": result["chosen_alpha_2024_only"],
        "2025_market_brier": result["2025_probability_scores"]["market_only"]["brier"],
        "2025_full_model_brier": result["2025_probability_scores"]["full_raw_model"]["brier"],
        "2025_20pct_archive_roi": result[
            "2025_spread_archive_large_ev_tails"
        ]["raw_ev_at_least_20_pct"]["archive_roi"],
        "wager_authorized": False,
    }))


if __name__ == "__main__":
    main()
