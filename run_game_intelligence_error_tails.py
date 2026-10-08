"""Evaluate frozen pregame efficiency against large NFL forecast errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_error_tails import analyze_error_tails
from nfl.recent_form_dataset import build_recent_form_walkforward_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument(
        "--output", default="reports/game_intelligence_error_tails.json"
    )
    args = parser.parse_args()
    if args.start_season > args.end_season:
        raise ValueError("start season cannot follow end season")
    seasons = list(range(args.start_season, args.end_season + 1))
    client = NFLDataClient()
    schedules = client.load_schedules(
        list(range(args.start_season - 1, args.end_season + 1))
    )
    pbp = client.load_pbp(seasons)
    snapshots = [
        build_recent_form_walkforward_dataset(
            schedules, pbp, season, start_week=5
        )
        for season in seasons
    ]
    report = analyze_error_tails(pl.concat(snapshots, how="vertical_relaxed"))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "status": report["status"],
        "margin_extremes": report["targets"]["margin"]["extreme_errors"],
        "total_extremes": report["targets"]["total"]["extreme_errors"],
    }))


if __name__ == "__main__":
    main()
