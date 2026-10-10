"""Walters-inspired exact-margin, key-number score distribution research.

Research-only: learn discrete final-score margin probabilities from earlier
historical forecasts, validate on later seasons, no sportsbook odds in fit.
Historical forecast capture times are NOT verified. Never authorizes wagering.
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

MARGINS = tuple(range(-100, 101))
KEYS = (3, 7)
ALPHAS = (0.0, 0.5, 1.0)
LINES = (-7.0, -6.5, -3.0, -2.5, 0.0, 2.5, 3.0, 6.5, 7.0)
PRIOR_EQUIVALENT_GAMES = 120.0
COLUMNS = {
    "nfl": ("projected_home_margin", "actual_home_margin"),
    "cfb": ("pred_margin_home", "actual_margin_home"),
}


def load(path, sport):
    pred_col, actual_col = COLUMNS[sport]
    required = {"season", "week", "game_id", pred_col, actual_col}
    rows, ids = [], set()
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Historical projected/actual margin fields missing")
        for rec in reader:
            gid = str(rec["game_id"]).strip()
            if not gid or gid in ids:
                raise ValueError("Duplicate/missing game ID")
            ids.add(gid)
            try:
                season, week = int(rec["season"]), int(rec["week"])
                pred, actual = float(rec[pred_col]), float(rec[actual_col])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid numeric game observation") from exc
            if season < 2000 or week < 1 or week > 30:
                raise ValueError("Invalid season or week")
            if not all(math.isfinite(v) for v in (pred, actual)):
                raise ValueError("Nonfinite score")
            if not actual.is_integer() or abs(actual) > 100:
                raise ValueError("Observed score margin must be an integer [-100,100]")
            rows.append({
                "game_id": gid, "season": season, "week": week,
                "pred": pred, "actual": int(actual),
            })
    return sorted(rows, key=lambda r: (r["season"], r["week"], r["game_id"]))


def _cdf(z):
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def gaussian_bins(center, sigma):
    if not math.isfinite(center) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Bad Gaussian center/scale")
    p = [
        max(0.0, _cdf((m + .5 - center) / sigma)
            - _cdf((m - .5 - center) / sigma))
        for m in MARGINS
    ]
    # Fold any floating-point/tail loss into the outermost bins.
    p[0] += _cdf((-100.5 - center) / sigma)
    p[-1] += 1.0 - _cdf((100.5 - center) / sigma)
    total = sum(p)
    return [x / total for x in p]


def fit_baseline(train):
    if len(train) < 100:
        raise ValueError("Insufficient earlier-season training games")
    residual = [r["actual"] - r["pred"] for r in train]
    mean = sum(residual) / len(residual)
    sd = math.sqrt(
        sum((x - mean) ** 2 for x in residual) / (len(residual) - 1)
    )
    if sd < 1:
        raise ValueError("Degenerate historical margin variance")
    return mean, sd


def fit_key_multipliers(train, mean, sigma):
    expected = {k: 0.0 for k in KEYS}
    observed = {k: 0 for k in KEYS}
    for r in train:
        pmf = gaussian_bins(r["pred"] + mean, sigma)
        for k in KEYS:
            expected[k] += pmf[100 + k] + pmf[100 - k]
            observed[k] += abs(r["actual"]) == k
    multipliers = {}
    for k in KEYS:
        base_frequency = expected[k] / len(train)
        prior = PRIOR_EQUIVALENT_GAMES * base_frequency
        ratio = (observed[k] + prior) / (expected[k] + prior)
        multipliers[k] = max(0.60, min(2.20, ratio))
    return multipliers, observed, expected


def fit(train):
    mean, sigma = fit_baseline(train)
    multipliers, observed, expected = fit_key_multipliers(train, mean, sigma)
    return {
        "mean": mean, "sigma": sigma, "multipliers": multipliers,
        "observed_key_counts": observed, "expected_key_counts": expected,
        "fit_games": len(train),
    }


def distribution(model, projected_margin, alpha):
    if alpha not in ALPHAS:
        raise ValueError("Unregistered tilt strength")
    p = gaussian_bins(projected_margin + model["mean"], model["sigma"])
    if alpha:
        for k in KEYS:
            factor = model["multipliers"][k] ** alpha
            p[100 - k] *= factor
            p[100 + k] *= factor
        mass = sum(p)
        p = [x / mass for x in p]
    return p


def outcomes(pmf, *, home_spread):
    if home_spread not in LINES:
        raise ValueError("Spread must be one of prespecified audit lines")
    win = push = loss = 0.0
    for m, p in zip(MARGINS, pmf, strict=True):
        value = m + home_spread
        if value > 0:
            win += p
        elif value == 0:
            push += p
        else:
            loss += p
    return win, push, loss


def scores(rows, model, alpha):
    result = []
    for row in rows:
        pmf = distribution(model, row["pred"], alpha)
        logloss = brier = 0.0
        for line in LINES:
            probs = outcomes(pmf, home_spread=line)
            resolved = row["actual"] + line
            outcome = 0 if resolved > 0 else 1 if resolved == 0 else 2
            logloss -= math.log(max(1e-12, probs[outcome]))
            brier += sum(
                (p - int(i == outcome)) ** 2 for i, p in enumerate(probs)
            )
        result.append({
            "season": row["season"], "week": row["week"],
            "game_id": row["game_id"],
            "log_loss": logloss / len(LINES),
            "brier": brier / len(LINES),
        })
    return result


def metrics(rows):
    if not rows:
        raise ValueError("Cannot grade empty games")
    return {
        "games": len(rows),
        "mean_3way_log_loss": sum(r["log_loss"] for r in rows) / len(rows),
        "mean_3way_brier": sum(r["brier"] for r in rows) / len(rows),
    }


def bootstrap(baseline, candidate, *, repetitions=3000):
    if len(baseline) != len(candidate):
        raise ValueError("Paired games required")
    blocks = defaultdict(list)
    for a, b in zip(baseline, candidate, strict=True):
        if a["game_id"] != b["game_id"]:
            raise ValueError("Pair alignment differs")
        blocks[(a["season"], a["week"])].append((
            a["log_loss"] - b["log_loss"],
            a["brier"] - b["brier"],
        ))
    ordered = [blocks[k] for k in sorted(blocks)]
    if len(ordered) < 8:
        return {"clusters": len(ordered), "familywise_95_ci": None}
    rng = random.Random(20261010)
    sums = [[sum(t[j] for t in block) for j in (0, 1)] for block in ordered]
    counts = [len(block) for block in ordered]
    draws = [[], []]
    for _ in range(repetitions):
        indices = [rng.randrange(len(ordered)) for _ in ordered]
        n = sum(counts[i] for i in indices)
        for j in (0, 1):
            draws[j].append(sum(sums[i][j] for i in indices) / n)
    # Conservative simultaneous intervals for 2 sports x 2 metrics.
    bounds = {}
    for metric, values in zip(("log_loss", "brier"), draws, strict=True):
        values.sort()
        bounds[metric] = [
            values[int(.00625 * (len(values) - 1))],
            values[int(.99375 * (len(values) - 1))],
        ]
    return {"clusters": len(ordered), "familywise_95_ci": bounds}


def evaluate(rows, sport):
    seasons = sorted({r["season"] for r in rows})
    if sport == "nfl":
        train_seasons, tune_year, test_year = (2022, 2023), 2024, 2025
    else:
        train_seasons, tune_year, test_year = (2023,), 2024, 2025
    train = [r for r in rows if r["season"] in train_seasons]
    tune = [r for r in rows if r["season"] == tune_year]
    refit = [r for r in rows if r["season"] < test_year]
    test = [r for r in rows if r["season"] == test_year]
    if not seasons or min(seasons) != min(train_seasons):
        raise ValueError("Insufficient chronological seasons")
    if len(tune) < 100 or len(test) < 100:
        raise ValueError("Insufficient tuning/evaluation games")
    original = fit(train)
    tune_base = metrics(scores(tune, original, 0.0))
    tuned = {}
    eligible = []
    for alpha in ALPHAS:
        outcome = metrics(scores(tune, original, alpha))
        tuned[str(alpha)] = outcome
        if alpha > 0 and all(
            outcome[key] < tune_base[key]
            for key in ("mean_3way_log_loss", "mean_3way_brier")
        ):
            eligible.append((outcome["mean_3way_log_loss"], alpha))
    selected = min(eligible)[1] if eligible else 0.0
    # Selection is frozen before test labels can be examined. Refit on all
    # earlier seasons, including tuning outcomes; never fit to evaluation.
    refitted = fit(refit)
    holdout_baseline = scores(test, refitted, 0.0)
    holdout_candidate = scores(test, refitted, selected)
    base = metrics(holdout_baseline)
    candidate = metrics(holdout_candidate)
    ci = bootstrap(holdout_baseline, holdout_candidate)
    pass_ci = bool(
        ci["familywise_95_ci"]
        and all(x[0] > 0 for x in ci["familywise_95_ci"].values())
    )
    return {
        "spec": "walters_exact_margin_key_number_research_v1",
        "sport": sport,
        "selection_train_seasons": list(train_seasons),
        "tuning_season": tune_year,
        "evaluation_season": test_year,
        "train_games": len(train),
        "tune_games": len(tune),
        "evaluation_games": len(test),
        "discrete_support": [-100, 100],
        "special_margins": list(KEYS),
        "tested_home_spread_lines": list(LINES),
        "price_source": "NO_SPORTSBOOK_PRICE_USED",
        "baseline": "discretized historical Gaussian margin residual",
        "candidate": "training-only smoothed symmetric 3/7 margin multipliers",
        "candidate_strengths": list(ALPHAS),
        "initial_training_multipliers": original["multipliers"],
        "tuning_baseline": tune_base,
        "tuning_candidates": tuned,
        "selected_strength_on_tuning": selected,
        "refit_prior_season_multipliers": refitted["multipliers"],
        "evaluation_baseline": base,
        "evaluation_candidate": candidate,
        "evaluation_improvement_baseline_minus_candidate": {
            k: base[k] - candidate[k]
            for k in ("mean_3way_log_loss", "mean_3way_brier")
        },
        "week_paired_bootstrap": ci,
        "retrospective_statistical_screen_pass": pass_ci and selected > 0,
        "economic_edge_verified": False,
        "historical_prediction_pregame_timestamps_verified": False,
        "pristine_holdout": False,
        "betting_authorized": False,
        "promotion_authorized": False,
        "limitations": [
            "Historical 2025 outcomes were explored by earlier unrelated research",
            "Archived prediction timestamps have not been verified pregame",
            "Reference spreads are prespecified synthetic diagnostics, not sportsbook offers",
            "Multiclass scores are repeated dependent observations within each game",
            "No real-book prices, ROI, CLV or executable wager data",
            "Do not promote adjustments without genuinely forward independent evidence",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=tuple(COLUMNS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load(args.source, args.sport), args.sport)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "sport": args.sport, "tuning_selected_strength": report["selected_strength_on_tuning"],
        "evaluation_games": report["evaluation_games"],
        "evaluation_baseline": report["evaluation_baseline"],
        "evaluation_candidate": report["evaluation_candidate"],
        "screen_pass": report["retrospective_statistical_screen_pass"],
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
