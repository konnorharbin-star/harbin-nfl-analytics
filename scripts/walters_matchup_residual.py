"""Retrospective, opponent-aware scoring residuals. RESEARCH ONLY, no betting.

For each target season/week, the offense and defense corrections are constructed
only from completed games in STRICTLY PRIOR weeks of that same season. Training
and evaluation outcomes never enter contemporaneous feature construction.
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

COLUMNS = {
    "nfl": ("projected_home_margin", "projected_total",
            "actual_home_margin", "actual_total"),
    "cfb": ("pred_margin_home", "pred_total",
            "actual_margin_home", "actual_total"),
}
LOOKBACKS = (3, 6, 10)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
SHRINK_GAMES = 4.0


def load_rows(path: Path, sport: str):
    margin, total, actual_margin, actual_total = COLUMNS[sport]
    required = {"game_id", "season", "week", "home_team", "away_team",
                margin, total, actual_margin, actual_total}
    rows, seen = [], set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required historical prediction fields")
        for rec in reader:
            gid = str(rec["game_id"]).strip()
            if not gid or gid in seen:
                raise ValueError("Duplicate or missing game_id")
            seen.add(gid)
            home, away = (str(rec[k]).strip() for k in ("home_team", "away_team"))
            if not home or not away or home.casefold() == away.casefold():
                raise ValueError("Invalid home and away teams")
            try:
                season, week = int(rec["season"]), int(rec["week"])
                values = [float(rec[k]) for k in (margin, total, actual_margin, actual_total)]
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid historical score observation") from exc
            if week < 1 or season < 2000 or not all(map(math.isfinite, values)):
                raise ValueError("Out-of-range historical score observation")
            rows.append({"game_id": gid, "season": season, "week": week,
                         "home_team": home, "away_team": away,
                         "pred_margin": values[0], "pred_total": values[1],
                         "actual_margin": values[2], "actual_total": values[3]})
    return rows


def pregame_features(rows, lookback: int):
    """Return new rows with total/margin corrections from PRIOR weeks only.

    Results for all games in the current week are ingested only after all
    features for that week are frozen. Season resets avoid stale roster effects.
    """
    if lookback not in LOOKBACKS:
        raise ValueError("Unregistered lookback")
    weeks = defaultdict(list)
    for row in rows:
        weeks[(row["season"], row["week"])].append(row)
    state = defaultdict(lambda: {"offense": [], "defense": []})
    prior_season = None
    output = []
    for (season, _week), games in sorted(weeks.items()):
        # Stable tie-breaking even when two games share a team and week.
        games = sorted(games, key=lambda r: r["game_id"])
        if season != prior_season:
            state.clear()
            prior_season = season

        def mean(team, side):
            past = state[team][side][-lookback:]
            return sum(past) / (len(past) + SHRINK_GAMES)

        for r in games:
            home_delta = (mean(r["home_team"], "offense")
                          + mean(r["away_team"], "defense")) / 2.0
            away_delta = (mean(r["away_team"], "offense")
                          + mean(r["home_team"], "defense")) / 2.0
            output.append({**r, "delta_margin": home_delta - away_delta,
                           "delta_total": home_delta + away_delta})

        for r in games:
            ph = (r["pred_total"] + r["pred_margin"]) / 2.0
            pa = (r["pred_total"] - r["pred_margin"]) / 2.0
            ah = (r["actual_total"] + r["actual_margin"]) / 2.0
            aa = (r["actual_total"] - r["actual_margin"]) / 2.0
            state[r["home_team"]]["offense"].append(ah - ph)
            state[r["home_team"]]["defense"].append(aa - pa)
            state[r["away_team"]]["offense"].append(aa - pa)
            state[r["away_team"]]["defense"].append(ah - ph)
    return output


def metrics(rows, target, weight):
    errors = [
        r["pred_" + target] + weight * r["delta_" + target] - r["actual_" + target]
        for r in rows
    ]
    if not errors:
        raise ValueError("Empty evaluation group")
    return {"games": len(errors), "mae": sum(abs(x) for x in errors) / len(errors),
            "rmse": math.sqrt(sum(x * x for x in errors) / len(errors))}


def bootstrap(rows, target, weight, *, draws=2500):
    """Paired whole-week bootstrap, baseline improvement in score points."""
    if not weight:
        return {"week_clusters": len({(r["season"], r["week"]) for r in rows}),
                "mae_familywise_95_ci": [0.0, 0.0],
                "rmse_familywise_95_ci": [0.0, 0.0],
                "both_lower_bounds_positive": False}
    groups = defaultdict(list)
    for r in rows:
        b = r["pred_" + target] - r["actual_" + target]
        c = b + weight * r["delta_" + target]
        groups[(r["season"], r["week"])].append((b, c))
    blocks = [
        (len(v), sum(abs(b) for b, c in v), sum(abs(c) for b, c in v),
         sum(b*b for b, c in v), sum(c*c for b, c in v))
        for v in groups.values()
    ]
    rng = random.Random(20261010)
    maes, rmses = [], []
    for _ in range(draws):
        n = a = b = s = t = 0
        for _ in blocks:
            nn, aa, bb, ss, tt = blocks[rng.randrange(len(blocks))]
            n += nn
            a += aa
            b += bb
            s += ss
            t += tt
        maes.append((a-b)/n)
        rmses.append(math.sqrt(s/n)-math.sqrt(t/n))

    def ci(v):
        ordered = sorted(v)
        def quantile(q):
            pos = q*(len(ordered)-1)
            lo = int(pos)
            return ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(pos-lo)
        # Conservative Bonferroni family of 2 sports x 2 targets x 2 endpoints.
        return [quantile(0.003125), quantile(0.996875)]

    mc, rc = ci(maes), ci(rmses)
    return {"week_clusters": len(blocks), "mae_familywise_95_ci": mc,
            "rmse_familywise_95_ci": rc,
            "both_lower_bounds_positive": mc[0] > 0 and rc[0] > 0}


def evaluate(rows, sport, tune_season=2024, evaluation_season=2025):
    if tune_season >= evaluation_season:
        raise ValueError("Evaluation must follow tuning")
    years = {s: [r for r in rows if r["season"] == s]
             for s in {r["season"] for r in rows}}
    for label, partition in (
        ("development", [r for r in rows if r["season"] < tune_season]),
        ("tuning", years.get(tune_season, [])),
        ("evaluation", years.get(evaluation_season, [])),
    ):
        if len(partition) < 100 or len({(r["season"],r["week"]) for r in partition}) < 8:
            raise ValueError("Insufficient chronological " + label + " data")
    by_lookback = {k: pregame_features(rows, k) for k in LOOKBACKS}
    report = {
        "spec": "walters_matchup_residual_v1", "sport": sport,
        "tuning_season": tune_season, "evaluation_season": evaluation_season,
        "development_seasons": sorted(s for s in years if s < tune_season),
        "status": "RETROSPECTIVE_UNVERIFIED_ARCHIVE",
        "source_prediction_capture_times_verified": False,
        "economic_edge_verified": False, "pristine_untouched_holdout": False,
        "betting_authorized": False, "model_promotion_authorized": False,
        "frozen_grid": {"lookbacks": LOOKBACKS, "weights": WEIGHTS,
                        "shrink_games": SHRINK_GAMES},
        "limitations": [
            "Historical model projections are not verified as pregame immutable forecasts",
            "Archives and 2025 outcomes were previously reviewed; not a pristine holdout",
            "No sportsbook price, line stage, CLV, ROI or expected return is inferred",
            "Live deployment requires new, timestamped prospective validation",
        ],
        "targets": {},
    }
    for target in ("margin", "total"):
        base_rows = [r for r in by_lookback[LOOKBACKS[0]]
                     if r["season"] == tune_season]
        baseline = metrics(base_rows, target, 0.0)
        candidates = []
        for lookback, enriched in by_lookback.items():
            tuning = [r for r in enriched if r["season"] == tune_season]
            for weight in WEIGHTS:
                grade = metrics(tuning, target, weight)
                candidates.append({"lookback": lookback, "weight": weight,
                                   "tuning": grade,
                                   "beats_both": grade["mae"] < baseline["mae"] - 1e-12
                                   and grade["rmse"] < baseline["rmse"] - 1e-12})
        eligible = [x for x in candidates if x["beats_both"]]
        selected = min(eligible, key=lambda x: (
            x["tuning"]["rmse"], x["tuning"]["mae"],
            x["lookback"], x["weight"])) if eligible else None
        lb = selected["lookback"] if selected else LOOKBACKS[0]
        w = selected["weight"] if selected else 0.0
        future = [r for r in by_lookback[lb] if r["season"] == evaluation_season]
        report["targets"][target] = {
            "tuning_baseline": baseline, "all_prespecified_candidates": candidates,
            "selected": {"lookback": lb if selected else None, "weight": w,
                         "reason": "TUNING_BOTH_METRICS" if selected else "ZERO_CORRECTION"},
            "evaluation_baseline": metrics(future, target, 0.0),
            "evaluation_candidate": metrics(future, target, w),
            "evaluation_uncertainty": bootstrap(future, target, w),
            "production_eligible": False,
        }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sport", required=True, choices=tuple(COLUMNS))
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    report = evaluate(load_rows(args.source, args.sport), args.sport)
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"sport": args.sport, "targets": {
        k: {"selection": t["selected"], "baseline": t["evaluation_baseline"],
            "candidate": t["evaluation_candidate"],
            "confidence": t["evaluation_uncertainty"]["both_lower_bounds_positive"]}
        for k, t in report["targets"].items()},
        "betting_authorized": False}))


if __name__ == "__main__":
    main()
