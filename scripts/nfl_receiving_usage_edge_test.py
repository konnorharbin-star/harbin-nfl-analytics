"""Fixed receiving-workload hypothesis: conditional, chronological, unpriced.

2024 selects workload windows and fits empirical errors; 2025 is previously
inspected diagnostics. Missing stat rows stay unknown. A single pre-outcome,
baseline-only median-proximal synthetic line is used per player/game/market.
No actual market profitability is established by this experiment.
"""
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import random

import polars as pl
from nfl.prop_challenger import source_frames, forecast_week
from nfl.prop_probability import PROP_THRESHOLD_GRID, _scale, over_probability
from scripts.usage_prop_research import _usage_forecast, add_opportunity_columns, OPTIONS, WINDOWS

SPEC = "NFL_RECEIVING_USAGE_CONDITIONAL_FIXED_TEST_V1"
MARKETS = ("receptions", "receiving_yards")


def normalize_players(schedules, players):
    fixtures = {}
    for g in schedules.to_dicts():
        if g["game_type"] != "REG":
            continue
        for team in (g["home_team"], g["away_team"]):
            key = (g["season"], g["week"], team)
            if key in fixtures:
                raise ValueError("AMBIGUOUS_TEAM_WEEK")
            fixtures[key] = g
    result = []
    seen = set()
    for row in players.to_dicts():
        if row.get("season_type") != "REG":
            continue
        team = row.get("team") or row.get("recent_team")
        g = fixtures.get((row.get("season"), row.get("week"), team))
        if g is None:
            raise ValueError("UNMATCHED_PLAYER_TEAM_WEEK")
        if row.get("game_id") not in (None, g["game_id"]):
            raise ValueError("PLAYER_GAME_ID_CONFLICT")
        opponent = g["away_team"] if team == g["home_team"] else g["home_team"]
        if row.get("opponent_team") not in (None, opponent):
            raise ValueError("PLAYER_OPPONENT_CONFLICT")
        columns=("player_id","player_name","position","season","week","season_type",
                 "receptions","receiving_yards","rushing_yards","passing_yards",
                 "targets","carries","attempts")
        item = {key:row.get(key) for key in columns}
        item.update(team=team, game_id=g["game_id"])
        key = (g["game_id"], row["player_id"])
        if key in seen:
            raise ValueError("DUPLICATE_PLAYER_GAME")
        seen.add(key)
        result.append(item)
    return pl.DataFrame(result)


def examples(games, stats, year):
    actual = {(r["game_id"], r["player_id"]): r for r in stats}
    complete = {g["game_id"] for g in games if g["season"] == year and g["completed"]}
    rows, unknown = [], 0
    weeks = sorted({g["week"] for g in games if g["season"] == year and g["completed"] and g["week"] >= 4})
    for week in weeks:
        for row in forecast_week(games, stats, season=year, week=week, options=OPTIONS):
            if row["market"] not in MARKETS or row["game_id"] not in complete:
                continue
            target = actual.get((row["game_id"], row["player_id"]))
            if target is None or target.get(row["market"]) is None:
                unknown += 1
                continue
            if target["team"] != row["team"]:
                raise ValueError("TARGET_TEAM_CONFLICT")
            rows.append((row, float(target[row["market"]])))
    return rows, unknown


def fit_errors(rows, method):
    vectors = defaultdict(list)
    for row, actual in rows:
        mean = row[method]
        error = (actual-mean)/_scale(row["market"], mean)
        vectors[(row["market"], "ALL")].append(error)
        vectors[(row["market"], row["position"])].append(error)
    return vectors


def vector(vectors, row):
    specific = vectors.get((row["market"], row["position"]), [])
    return specific if len(specific) >= 80 else vectors[(row["market"], "ALL")]


def clustered(records, field, reps=3000, by_week=False):
    by_game = defaultdict(list)
    for row in records:
        key="_".join(row["game_id"].split("_")[:2]) if by_week else row["game_id"]
        by_game[key].append(row[field])
    groups = list(by_game.values())
    if len(groups) < (8 if by_week else 20):
        return None
    rng = random.Random(20261014)
    draws = []
    for _ in range(reps):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(sum(sum(g) for g in sample)/sum(len(g) for g in sample))
    draws.sort()
    # Bonferroni over the two registered receiving market families.
    return [draws[int(.0125*reps)], draws[int(.9875*reps)-1]]


def run_study(games, stats):
    training, missing_train = examples(games, stats, 2024)
    testing, missing_test = examples(games, stats, 2025)
    histories = defaultdict(list)
    for row in stats:
        histories[(row["season"], row["team"], row["player_id"])].append(row)
    report = {"spec": SPEC, "status": "UNPRICED_RESEARCH_ONLY", "training": 2024,
              "diagnostic": 2025, "diagnostic_previously_inspected": True,
              "missing_train_stat_groups": missing_train, "missing_test_stat_groups": missing_test,
              "real_market_prices_available": False, "betting_authorized": False,
              "action": "PASS", "markets": {}}
    catalogs, options = {}, {}
    for market in MARKETS:
        train = [(r, y) for r, y in training if r["market"] == market]
        candidates = []
        for window in WINDOWS:
            rows = [(r, y, _usage_forecast(r, histories, window)) for r, y in train]
            valid = [(r, y, p) for r, y, p in rows if p is not None]
            if len(valid) >= 160 and len(valid) >= .9*len(train):
                candidates.append((sum(abs(p-y) for r, y, p in valid)/len(valid), window))
        if not candidates:
            raise ValueError("INSUFFICIENT_TRAINING_USAGE")
        window = min(candidates)[1]
        options[market] = window
        transformed = []
        for r, y in train:
            p = _usage_forecast(r, histories, window)
            if p is not None:
                transformed.append((dict(r, usage_mean=p), y))
        usage_errors = fit_errors(transformed, "usage_mean")
        baseline_errors = fit_errors(transformed, "challenger_mean")
        catalogs[market] = {"usage": usage_errors, "baseline": baseline_errors}
        scored = []
        for r, y in testing:
            if r["market"] != market:
                continue
            p = _usage_forecast(r, histories, window)
            if p is None:
                continue
            base = r["challenger_mean"]
            be = vector(baseline_errors, r)
            ue = vector(usage_errors, r)
            # Primary anchor depends only on baseline's pre-outcome probability.
            line = min(PROP_THRESHOLD_GRID[market], key=lambda l: (
                round(abs(over_probability(base, l, market, be)-.5), 10), l))
            pb = over_probability(base, line, market, be)
            pu = over_probability(p, line, market, ue)
            outcome = int(y > line)
            scored.append({"game_id": r["game_id"], "brier_usage": (pu-outcome)**2,
                           "brier_baseline": (pb-outcome)**2,
                           "brier_delta": (pu-outcome)**2-(pb-outcome)**2,
                           "mae_delta": abs(p-y)-abs(base-y),
                           "mae_usage": abs(p-y), "mae_baseline": abs(base-y)})
        if len(scored) < 160:
            raise ValueError("INSUFFICIENT_DIAGNOSTIC")
        n = len(scored)
        report["markets"][market] = {
            "selected_training_window": window, "training_groups": len(transformed),
            "diagnostic_groups": n, "distinct_games": len({r["game_id"] for r in scored}),
            **{field: sum(r[field] for r in scored)/n for field in (
                "mae_usage", "mae_baseline", "mae_delta", "brier_usage", "brier_baseline", "brier_delta")},
            "game_cluster_brier_delta_interval_97_5": clustered(scored, "brier_delta"),
            "week_cluster_brier_delta_interval_97_5": clustered(scored, "brier_delta", by_week=True),
            "game_cluster_mae_delta_interval_97_5": clustered(scored, "mae_delta"),
            "matched_stat_row_conditional_only": True,
            "single_synthetic_primary_line_not_actual_book_line": True,
            "no_market_edge_certified": True}
    return report, catalogs, options


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-prefix", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/receiving_usage_conditional_edge_test.json"))
    args = parser.parse_args()
    if args.cache_prefix:
        schedules = pl.read_parquet(args.cache_prefix/"schedules_research.parquet")
        raw = pl.read_parquet(args.cache_prefix/"player_research.parquet")
    else:
        import nflreadpy as nfl
        schedules = nfl.load_schedules([2024, 2025, 2026])
        raw = nfl.load_player_stats([2024, 2025, 2026])
    players = normalize_players(schedules, raw)
    games, stats, quality = source_frames(schedules, players)
    stats = add_opportunity_columns(stats, players)
    result, catalogs, options = run_study(games, stats)
    result["captured_utc"] = datetime.now(UTC).isoformat()
    result["normalized_source_sha256"] = hashlib.sha256(json.dumps(
        stats, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["source_quality"] = quality
    result["sources"] = ["https://github.com/nflverse/nflverse-data/releases",
                         "https://nflreadr.nflverse.com/articles/dictionary_player_stats.html"]
    result["calibration_catalog"] = {
        market+"|"+method+"|"+position:errors
        for market, methods in catalogs.items() for method, groups in methods.items()
        for (unused, position), errors in groups.items()}
    histories = defaultdict(list)
    for row in stats:
        histories[(row["season"], row["team"], row["player_id"])].append(row)
    now = datetime.now(UTC)
    future = []
    for week in sorted({g["week"] for g in games if g["season"] == 2026 and g["kickoff"]
                        and now < g["kickoff"] and (g["kickoff"]-now).days <= 10}):
        for r in forecast_week(games, stats, season=2026, week=week, options=OPTIONS, future_only_at=now):
            if r["market"] not in MARKETS:
                continue
            if datetime.fromisoformat(r["kickoff_utc"]) > now+timedelta(days=10):
                continue
            mean = _usage_forecast(r, histories, options[r["market"]])
            if mean is None:
                continue
            position=r["position"] if len(catalogs[r["market"]]["usage"].get((r["market"],r["position"]),[]))>=80 else "ALL"
            future.append({**r, "usage_mean": mean, "usage_window": options[r["market"]],
                           "residual_group": r["market"]+"|usage|"+position,
                           "captured_utc": now.isoformat(), "spec": SPEC,
                           "betting_authorized": False, "estimated_ev": None})
    result["forward_forecasts"] = future
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "forward_forecasts"}, sort_keys=True))
    print(json.dumps({"new_unpriced_forward_stat_groups": len(future)}))


if __name__ == "__main__":
    main()
