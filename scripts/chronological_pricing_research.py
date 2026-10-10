"""Chronological pricing-layer research; never authorizes economic betting claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.market_first_experiment import nfl_history

RIDGES = (10.0, 100.0, 1000.0)
FAMILIES = ("market_calibration", "football_residual")


def arrays(rows):
    ids = [r["game_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game observations")
    p = np.array(
        [[r["predictions"]["market"], r["predictions"]["independent_model"]] for r in rows]
    )
    y = np.array([r["outcome"] for r in rows], dtype=float)
    if (
        not len(rows)
        or not np.isfinite(p).all()
        or not ((p > 0) & (p < 1)).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Finite paired probabilities and binary outcomes required")
    z = np.log(p / (1 - p))
    return z[:, 0], np.column_stack((z[:, 0], z[:, 1] - z[:, 0])), y


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def fit(rows, ridge, *, residual):
    """Ridge logistic offset; model weight [0,1], market temperature [0.5,1.5]."""
    offset, x, y = arrays(rows)
    if not residual:
        x[:, 1] = 0
    weights = np.zeros(2)

    def objective(w):
        z = offset + x @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + ridge * np.sum(w * w) / 2)

    for _ in range(100):
        p = sigmoid(offset + x @ weights)
        gradient = x.T @ (p - y) + ridge * weights
        hessian = (x.T * (p * (1 - p))) @ x + ridge * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        old = objective(weights)
        accepted = weights
        for scale in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            proposed = np.clip(weights - scale * step, [-0.5, 0], [0.5, 1])
            if objective(proposed) <= old:
                accepted = proposed
                break
        if np.max(np.abs(accepted - weights)) < 1e-9:
            break
        weights = accepted
    return [float(v) for v in weights]


def predict(rows, weights):
    offset, x, _ = arrays(rows)
    return sigmoid(offset + x @ np.array(weights))


def metrics(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p)))),
    }


def adequate(rows):
    return len(rows) >= 100 and len({(r["season"], r["week"]) for r in rows}) >= 8


def select(train, tune, family):
    if not adequate(train) or not adequate(tune):
        return {"weights": [0.0, 0.0], "reason": "INSUFFICIENT_DEVELOPMENT_SAMPLE", "ridge": None}
    _, _, y = arrays(tune)
    baseline = metrics(y, predict(tune, [0, 0]))
    candidates = []
    for ridge in RIDGES:
        w = fit(train, ridge, residual=family == "football_residual")
        score = metrics(y, predict(tune, w))
        # Zero weight remains eligible; both prespecified endpoints must improve.
        if all(score[k] < baseline[k] for k in baseline):
            candidates.append((score["log_loss"], ridge, w, score))
    if not candidates:
        return {
            "weights": [0.0, 0.0],
            "reason": "REJECTED_ON_TUNING",
            "ridge": None,
            "tuning_market": baseline,
        }
    _, ridge, w, score = min(candidates)
    return {
        "weights": w,
        "ridge": ridge,
        "reason": "SELECTED_WITHOUT_EVALUATION_OUTCOMES",
        "tuning_market": baseline,
        "tuning_candidate": score,
    }


def evaluate(rows):
    arrays(rows)
    seasons = sorted({int(r["season"]) for r in rows})
    folds, pooled = [], []
    for season in seasons[2:]:
        tuning = max(s for s in seasons if s < season)
        train = [r for r in rows if r["season"] < tuning]
        tune = [r for r in rows if r["season"] == tuning]
        test = [r for r in rows if r["season"] == season]
        selections = {f: select(train, tune, f) for f in FAMILIES}
        _, _, y = arrays(test)
        probabilities = {
            "market": predict(test, [0, 0]),
            **{f: predict(test, selections[f]["weights"]) for f in FAMILIES},
        }
        folds.append(
            {
                "evaluation_season": season,
                "training_seasons": [s for s in seasons if s < tuning],
                "tuning_season": tuning,
                "games": len(test),
                "selections": selections,
                "metrics": {k: metrics(y, p) for k, p in probabilities.items()},
            }
        )
        for i, r in enumerate(test):
            pooled.append(
                {**r, "pricing_predictions": {k: float(p[i]) for k, p in probabilities.items()}}
            )
    report = {
        "folds": folds,
        "evaluation_games": len(pooled),
        "status": "EXPLORATORY_CHRONOLOGICAL_PRICING_RESEARCH",
        "betting_authorized": False,
        "promotion_eligible": False,
        "historical_untouched_holdout": False,
        "economic_evidence_eligible": False,
        "spec": "chronological_pricing_v1",
        "limits": [
            "Previously inspected archives are not a pristine holdout",
            "Archive entry timestamps and executable prices remain unverified",
            "No thresholds, wagers, ROI or CLV inferred from prediction scores",
            "Frozen prospective assessment and external release review required",
        ],
    }
    if not pooled:
        return report
    _, _, y = arrays(pooled)
    p = {k: np.array([r["pricing_predictions"][k] for r in pooled]) for k in ("market", *FAMILIES)}
    report["metrics"] = {k: metrics(y, v) for k, v in p.items()}
    groups = sorted({(r["season"], r["week"]) for r in pooled})
    report["week_clusters"] = len(groups)
    blocks = [
        np.array([i for i, r in enumerate(pooled) if (r["season"], r["week"]) == g]) for g in groups
    ]
    samples = np.random.default_rng(20261010).integers(0, len(blocks), (10000, len(blocks)))
    counts = np.array([len(b) for b in blocks])
    report["paired_improvement"] = {}
    for family in FAMILIES:
        result = {}
        for metric in ("brier", "log_loss"):

            def losses(v, endpoint=metric):
                v = np.clip(v, 1e-12, 1 - 1e-12)
                return (
                    (v - y) ** 2
                    if endpoint == "brier"
                    else -(y * np.log(v) + (1 - y) * np.log1p(-v))
                )

            diff = losses(p["market"]) - losses(p[family])
            ci = None
            if adequate(pooled):
                sums = np.array([diff[b].sum() for b in blocks])
                estimate = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
                # Two candidate families x two endpoints, week-block paired CI.
                ci = [float(v) for v in np.quantile(estimate, [0.05 / 8, 1 - 0.05 / 8])]
            result[metric] = {"market_minus_candidate": float(diff.mean()), "familywise_95_ci": ci}
        report["paired_improvement"][family] = result
    report["candidate_decisions"] = {
        family: "ARCHIVE_SCORE_IMPROVEMENT_ONLY"
        if all(
            v["familywise_95_ci"] is not None and v["familywise_95_ci"][0] > 0
            for v in report["paired_improvement"][family].values()
        )
        else "NO_SUPPORTED_MARKET_IMPROVEMENT"
        for family in FAMILIES
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path)
    args = parser.parse_args()
    report = evaluate(nfl_history(args.historical))
    report.update(
        sport=args.sport, input_sha256=hashlib.sha256(args.historical.read_bytes()).hexdigest()
    )
    if args.source_metadata:
        report["source_metadata"] = json.loads(args.source_metadata.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "sport": args.sport,
                "games": report["evaluation_games"],
                "metrics": report.get("metrics"),
                "paired_improvement": report.get("paired_improvement"),
            }
        )
    )


if __name__ == "__main__":
    main()
