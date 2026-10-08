"""Grade frozen prior-game efficiency signals against NFL prediction residuals.

All snapshots are built by the existing walk-forward PBP pipeline; no target
game observations are included in the feature computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_factors import (
    PREGAME_FEATURES,
    attach_pregame_factors,
    evaluate_factor_diagnostics,
)
from nfl.recent_form_dataset import build_recent_form_walkforward_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--graded", default="reports/game_intelligence.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    graded = pl.read_csv(args.graded, try_parse_dates=True).filter(
        pl.col("season") == args.season
    )
    client = NFLDataClient()
    schedules = client.load_schedules([args.season - 1, args.season], refresh=args.refresh)
    pbp = client.load_pbp([args.season], refresh=args.refresh)
    snapshot = build_recent_form_walkforward_dataset(
        schedules, pbp, args.season, start_week=args.start_week
    )
    combined = attach_pregame_factors(
        graded, snapshot, feature_columns=PREGAME_FEATURES
    )
    report = evaluate_factor_diagnostics(combined, PREGAME_FEATURES)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output / "game_intelligence_factors.csv")
    (output / "game_intelligence_factors.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({
        "season": args.season,
        "graded_games": graded.height,
        "joined_rows": combined.height,
        "status": report["status"],
    }))


if __name__ == "__main__":
    main()
