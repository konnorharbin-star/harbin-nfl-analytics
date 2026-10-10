"""Exploratory Walters-style independent score calibration; never a wagering model.

Fits simple score-only residual corrections on past seasons, selects on a separate
season, evaluates on a later season. Historical archived projections do NOT
establish when original forecasts were created and are NOT pristine holdouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

RIDGES = (25.0, 100.0, 400.0)
TARGETS = {
    "nfl": (("margin", "projected_home_margin", "actual_home_margin"),
            ("total", "projected_total", "actual_total")),
    "cfb": (("margin", "pred_margin_home", "actual_margin_home"),
            ("total", "pred_total", "actual_total")),
}


def load(source: Path, sport: str):
    rows, ids = [], set()
    with source.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"game_id", "season", "week"}
        for _, pred, actual in TARGETS[sport]:
            required.update((pred, actual))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing mandatory historical score columns")
        for r in reader:
            game = str(r["game_id"]).strip()
            if not game or game in ids:
                raise ValueError("Missing or duplicate game_id")
            ids.add(game)
            try:
                r["season"], r["week"] = int(r["season"]), int(r["week"])
                if r["week"] < 1:
                    raise ValueError("Invalid week")
                for _, pred, actual in TARGETS[sport]:
                    r[pred], r[actual] = float(r[pred]), float(r[actual])
                    if not math.isfinite(r[pred]) or not math.isfinite(r[actual]):
                        raise ValueError("Nonfinite archived score")
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid season, week, or finite score") from exc
            rows.append(r)
    return rows


def grade(rows, pred, actual, weights=None):
    errors = []
    for r in rows:
        x = r[pred]
        correction = 0.0
        if weights is not None:
            correction = weights["intercept"] + weights["slope_z"] * (
                (x - weights["train_mean"]) / weights["train_sd"]
            )
        errors.append(x + correction - r[actual])
    if not errors:
        raise ValueError("Cannot grade an empty dataset")
    return {"games": len(errors),
            "mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors))}


def fit(train, pred, actual, family, ridge):
    x = [r[pred] for r in train]
    residual = [r[actual] - r[pred] for r in train]
    mu = sum(x) / len(x)
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))
    if sd <= 1e-12:
        sd = 1.0
    z = [(v - mu) / sd for v in x]
    # Shrink both corrections towards zero; forecast remains independent of lines.
    alpha = sum(residual) / (len(x) + ridge)
    beta = (sum(a * b for a, b in zip(z, residual)) /
            (sum(a * a for a in z) + ridge)) if family == "affine" else 0.0
    return {"intercept": alpha, "slope_z": beta,
            "train_mean": mu, "train_sd": sd}


def evaluate(rows, sport: str, tune_season: int, evaluation_season: int):
    if tune_season >= evaluation_season:
        raise ValueError("Evaluation must follow tuning")
    # Never use evaluation data to fit parameters or choose their family.
    train = [r for r in rows if r["season"] < tune_season]
    tune = [r for r in rows if r["season"] == tune_season]
    holdout = [r for r in rows if r["season"] == evaluation_season]
    for label, part in (("train", train), ("tune", tune), ("evaluation", holdout)):
        if len(part) < 100 or len({(r["season"], r["week"]) for r in part}) < 8:
            raise ValueError("Insufficient chronological " + label + " observations")
    result = {
        "spec": "walters_independent_score_calibration_v1",
        "sport": sport,
        "train_seasons": sorted({r["season"] for r in train}),
        "tuning_season": tune_season,
        "evaluation_season": evaluation_season,
        "status": "RETROSPECTIVE_EXPLORATORY_NOT_PROSPECTIVE",
        "prediction_capture_timestamps_verified": False,
        "pristine_untouched_holdout": False,
        "economic_edge_verified": False,
        "betting_authorized": False,
        "automatic_promotion": False,
        "limits": [
            "Archived score predictions do not prove original pregame provenance",
            "Evaluation seasons have been reviewed in prior research",
            "No bookmaker entry lines, executable odds, ROI or CLV evaluated",
            "Never deploy coefficients without fresh prospective comparison",
        ],
        "targets": {},
    }
    for label, pred, actual in TARGETS[sport]:
        baseline_tune = grade(tune, pred, actual)
        candidates = []
        for family in ("intercept", "affine"):
            for ridge in RIDGES:
                weights = fit(train, pred, actual, family, ridge)
                tune_score = grade(tune, pred, actual, weights)
                candidates.append({
                    "family": family,
                    "ridge": ridge,
                    "weights": weights,
                    "tuning": tune_score,
                    "beats_baseline_on_both_metrics": (
                        tune_score["mae"] < baseline_tune["mae"] and
                        tune_score["rmse"] < baseline_tune["rmse"]
                    ),
                })
        eligible = [r for r in candidates if r["beats_baseline_on_both_metrics"]]
        # Selection is 100% based on tuning; baseline with zero correction
        # always wins when no candidate improves both prespecified endpoints.
        selected = min(eligible, key=lambda r: (r["tuning"]["rmse"],
                                                 r["tuning"]["mae"])) if eligible else None
        result["targets"][label] = {
            "baseline_tuning": baseline_tune,
            "candidate_tuning": candidates,
            "selection": ({
                "family": selected["family"],
                "ridge": selected["ridge"],
                "weights": selected["weights"],
            } if selected else {"family": "baseline", "ridge": None,
                                "weights": None}),
            "evaluation_baseline": grade(holdout, pred, actual),
            "evaluation_selected": grade(
                holdout, pred, actual, selected["weights"] if selected else None
            ),
            "eligible_for_live_model": False,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=tuple(TARGETS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tune-season", type=int, default=2024)
    parser.add_argument("--evaluation-season", type=int, default=2025)
    args = parser.parse_args()
    report = evaluate(load(args.source, args.sport), args.sport,
                      args.tune_season, args.evaluation_season)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "sport": args.sport, "tune": args.tune_season,
        "evaluation": args.evaluation_season,
        "targets": {
            k: {"selected": v["selection"]["family"],
                "baseline": v["evaluation_baseline"],
                "corrected": v["evaluation_selected"]}
            for k, v in report["targets"].items()
        }, "economic_edge_verified": False
    }))


if __name__ == "__main__":
    main()
