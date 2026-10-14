"""Independent, past-only clean-play efficiency residual challenger.

The video motivates research, not coefficients. These filters and candidates are
our declared experiment. No prices, target-game plays or current news are features.
"""

from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np

from nfl.coupled_nflverse_asof import kickoff_utc
from nfl.ratings import fit_pregame_fair_score

SPEC = "NFL_CLEAN_EFFICIENCY_RESEARCH_V1"
METRICS = ("pass_epa", "rush_epa", "success_rate")
REQUIRED = {
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "epa",
    "qb_dropback",
    "rush_attempt",
    "qb_kneel",
    "qb_spike",
    "play_type",
    "down",
    "qtr",
    "half_seconds_remaining",
    "score_differential",
}


def aggregate(schedules, plays):
    """Keep real regulation pass/run plays; retain sacks, scrambles and turnovers.

    Neutral state uses football score/time, never spread-adjusted win probability.
    Missing required fields exclude the play; success is explicitly EPA > 0.
    """
    if not REQUIRED <= set(plays.columns):
        raise ValueError("MISSING_CLEAN_PLAY_COLUMNS")
    fixtures = {g["game_id"]: g for g in schedules.to_dicts() if g["game_type"] == "REG"}
    totals = defaultdict(lambda: {m: [0.0, 0] for m in METRICS})
    exclusions, seen = Counter(), set()
    for r in plays.select(sorted(REQUIRED)).iter_rows(named=True):
        g = fixtures.get(r["game_id"])
        if g is None or (r["posteam"], r["defteam"]) not in (
            (g["home_team"], g["away_team"]),
            (g["away_team"], g["home_team"]),
        ):
            exclusions["unmatched_game_team"] += 1
            continue
        if any(
            r[k] is None
            for k in (
                "epa",
                "play_id",
                "qb_dropback",
                "rush_attempt",
                "qb_kneel",
                "qb_spike",
                "score_differential",
                "half_seconds_remaining",
                "down",
                "qtr",
            )
        ):
            exclusions["missing_required_value"] += 1
            continue
        if not all(
            np.isfinite(float(r[k]))
            for k in ("epa", "score_differential", "half_seconds_remaining")
        ):
            exclusions["nonfinite"] += 1
            continue
        if any(r[k] not in (0, 1) for k in ("qb_dropback", "rush_attempt", "qb_kneel", "qb_spike")):
            exclusions["invalid_play_flag"] += 1
            continue
        if (
            r["play_type"] not in ("pass", "run")
            or r["down"] not in (1, 2, 3, 4)
            or r["qtr"] not in (1, 2, 3, 4)
            or r["qb_kneel"] != 0
            or r["qb_spike"] != 0
        ):
            exclusions["non_scrimmage_or_clock_play"] += 1
            continue
        if abs(r["score_differential"]) > 14 or r["half_seconds_remaining"] <= 120:
            exclusions["non_neutral_state"] += 1
            continue
        if r["qb_dropback"] != 1 and r["rush_attempt"] != 1:
            exclusions["unclassified_play"] += 1
            continue
        key = (r["game_id"], r["play_id"])
        if key in seen:
            raise ValueError("DUPLICATE_ELIGIBLE_PLAY")
        seen.add(key)
        metrics = [
            ("success_rate", float(r["epa"] > 0)),
            ("pass_epa" if r["qb_dropback"] == 1 else "rush_epa", float(r["epa"])),
        ]
        for role, team in (("off", r["posteam"]), ("def", r["defteam"])):
            for m, value in metrics:
                t = totals[(r["game_id"], team, role)][m]
                t[0] += value
                t[1] += 1
    return dict(totals), dict(exclusions)


def profile(prior_games, team, totals, league):
    """Last eight mature games, shrink each observed metric by 100 league plays."""
    games = [g for g in prior_games if team in (g["home_team"], g["away_team"])][-8:]
    if len(games) < 4:
        raise ValueError("INSUFFICIENT_PRIOR_TEAM_GAMES")
    result = {"prior_game_ids": [g["game_id"] for g in games]}
    for role in ("off", "def"):
        for m in METRICS:
            values = [totals.get((g["game_id"], team, role), {}).get(m, [0.0, 0]) for g in games]
            n = sum(v[1] for v in values)
            if n < 40 or sum(v[1] > 0 for v in values) < 4:
                raise ValueError("INSUFFICIENT_PRIOR_METRIC_PLAYS")
            result[role + "_" + m] = (sum(v[0] for v in values) + 100 * league[m]) / (n + 100)
            result[role + "_" + m + "_plays"] = n
    return result


def examples(schedules, totals, seasons=(2023, 2024, 2025), at=None):
    import polars as pl

    games = sorted([g for g in schedules.to_dicts() if g["game_type"] == "REG"], key=kickoff_utc)
    rows, errors = [], []
    for year, week in sorted(
        {(g["season"], g["week"]) for g in games if g["season"] in seasons and g["week"] >= 5}
    ):
        targets = [g for g in games if g["season"] == year and g["week"] == week]
        targets = (
            [g for g in targets if at < kickoff_utc(g) <= at + timedelta(days=8)]
            if at is not None
            else [g for g in targets if g["home_score"] is not None and g["away_score"] is not None]
        )
        if not targets:
            continue
        cutoff = at if at is not None else min(kickoff_utc(g) for g in targets)
        prior = [
            g
            for g in games
            if g["season"] in (year - 1, year)
            and (g["season"] < year or g["week"] < week)
            and g["home_score"] is not None
            and g["away_score"] is not None
            and kickoff_utc(g) + timedelta(hours=72) <= cutoff
        ]
        ids = {g["game_id"] for g in prior}
        league = {}
        for m in METRICS:
            vals = [v[m] for (gid, _, role), v in totals.items() if gid in ids and role == "off"]
            n = sum(v[1] for v in vals)
            if n == 0:
                raise ValueError("MISSING_PRIOR_LEAGUE_PLAYS")
            league[m] = sum(v[0] for v in vals) / n
        baseline = fit_pregame_fair_score(
            schedules.filter(pl.col("game_id").is_in(list(ids))), year, week
        )
        for g in targets:
            try:
                home = profile(prior, g["home_team"], totals, league)
                away = profile(prior, g["away_team"], totals, league)
                fair = baseline.project(g["home_team"], g["away_team"])
                signals = {}
                for m in METRICS:
                    # Defense stores EPA/success allowed; larger is weaker.
                    h = home["off_" + m] + away["def_" + m] - 2 * league[m]
                    a = away["off_" + m] + home["def_" + m] - 2 * league[m]
                    signals[m + "_margin"] = h - a
                    signals[m + "_total"] = h + a
                rows.append(
                    dict(
                        game_id=g["game_id"],
                        season=year,
                        week=week,
                        kickoff_utc=kickoff_utc(g).isoformat(),
                        home=home,
                        away=away,
                        baseline_margin=fair.home_margin,
                        baseline_total=fair.total,
                        actual_margin=g["home_score"] - g["away_score"] if at is None else None,
                        actual_total=g["home_score"] + g["away_score"] if at is None else None,
                        signals=signals,
                    )
                )
            except ValueError as exc:
                errors.append(dict(game_id=g["game_id"], reason=str(exc)))
    return rows, errors


def fit(rows, target, alpha):
    if len(rows) < 100 or target not in ("margin", "total"):
        raise ValueError("INSUFFICIENT_OR_INVALID_TRAINING")
    x = np.array([[r["signals"][m + "_" + target] for m in METRICS] for r in rows])
    y = np.array([r["actual_" + target] - r["baseline_" + target] for r in rows])
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = np.column_stack([np.ones(len(rows)), (x - center) / scale])
    penalty = np.diag([0.0, alpha, alpha, alpha])
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return dict(
        target=target,
        alpha=alpha,
        center=center.tolist(),
        scale=scale.tolist(),
        beta=beta.tolist(),
        metrics=list(METRICS),
    )


def predict(row, model, blend):
    if not 0 <= blend <= 1 or model["metrics"] != list(METRICS):
        raise ValueError("INVALID_FROZEN_SPEC")
    t = model["target"]
    x = np.array([row["signals"][m + "_" + t] for m in METRICS])
    z = np.r_[1.0, (x - model["center"]) / model["scale"]]
    return float(row["baseline_" + t] + blend * (z @ model["beta"]))


def study(rows):
    train = [r for r in rows if r["season"] == 2023]
    tune = [r for r in rows if r["season"] == 2024]
    diagnostic = [r for r in rows if r["season"] == 2025]
    if min(map(len, (train, tune, diagnostic))) < 100:
        raise ValueError("INSUFFICIENT_CHRONOLOGICAL_BLOCKS")
    results, frozen = {}, {}
    for t in ("margin", "total"):
        choices = []
        for alpha in (10.0, 100.0):
            model = fit(train, t, alpha)
            for blend in (0.0, 0.25, 0.5, 1.0):
                mae = np.mean([abs(predict(r, model, blend) - r["actual_" + t]) for r in tune])
                choices.append(dict(alpha=alpha, blend=blend, tuning_mae=float(mae)))
        chosen = min(choices, key=lambda c: (c["tuning_mae"], c["blend"], c["alpha"]))
        model = fit(train + tune, t, chosen["alpha"])
        truth = np.array([r["actual_" + t] for r in diagnostic])
        base = np.array([r["baseline_" + t] for r in diagnostic])
        candidate = np.array([predict(r, model, chosen["blend"]) for r in diagnostic])
        delta = abs(candidate - truth) - abs(base - truth)
        # Two score targets: Bonferroni 97.5% marginal intervals, game units.
        rng = np.random.default_rng(20261014)
        boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(4000)])
        week_values = [
            [float(delta[i]) for i, r in enumerate(diagnostic) if r["week"] == w]
            for w in sorted({r["week"] for r in diagnostic})
        ]
        week_boots = []
        for _ in range(4000):
            sampled = [week_values[i] for i in rng.integers(0, len(week_values), len(week_values))]
            week_boots.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
        results[t] = dict(
            selected=chosen,
            tuning_candidates=choices,
            baseline_mae=float(abs(base - truth).mean()),
            challenger_mae=float(abs(candidate - truth).mean()),
            baseline_rmse=float(np.sqrt(((base - truth) ** 2).mean())),
            challenger_rmse=float(np.sqrt(((candidate - truth) ** 2).mean())),
            mae_delta=float(delta.mean()),
            week_cluster_mae_delta_interval_975=np.quantile(week_boots, [0.0125, 0.9875]).tolist(),
            game_bootstrap_mae_delta_interval_975=np.quantile(boots, [0.0125, 0.9875]).tolist(),
        )
        frozen[t] = dict(model=model, blend=chosen["blend"])
    return dict(
        spec=SPEC,
        train_games=len(train),
        tune_games=len(tune),
        diagnostic_games=len(diagnostic),
        results=results,
        diagnostic_is_untouched=False,
        source_vintages_verified=False,
        market_accuracy_proven=False,
        validated_edges=0,
        probability_calibration=None,
        roi=None,
        clv=None,
        production_action="PASS",
        used_for_bet_ranking=False,
    ), frozen
