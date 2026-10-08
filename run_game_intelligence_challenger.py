"""Run fixed chronological NFL efficiency challenger; no automatic promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_challenger import evaluate_intelligence_challenger
from nfl.recent_form_dataset import build_recent_form_walkforward_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--output", default="reports/game_intelligence_challenger.json")
    args = parser.parse_args()
    if args.start_season > 2023 or args.end_season < 2025:
        raise ValueError("requires development before 2024 and both fixed test seasons")
    seasons = list(range(args.start_season, args.end_season + 1))
    client = NFLDataClient()
    schedules = client.load_schedules(
        list(range(args.start_season - 1, args.end_season + 1))
    )
    pbp = client.load_pbp(seasons)
    datasets = [
        build_recent_form_walkforward_dataset(
            schedules, pbp, season, start_week=5
        )
        for season in seasons
    ]
    frame = pl.concat(datasets, how="vertical_relaxed")
    frame = frame.rename({
        "baseline_home_margin": "projected_home_margin",
        "baseline_total": "projected_total",
    })
    report = evaluate_intelligence_challenger(frame)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "status": report["status"],
        "margin_consistent": report["targets"]["margin"]["consistent_improvement"],
        "total_consistent": report["targets"]["total"]["consistent_improvement"],
    }))


if __name__ == "__main__":
    main()
