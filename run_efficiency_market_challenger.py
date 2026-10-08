"""Evaluate fixed efficiency residual challenger against market archive."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.efficiency_market_challenger import evaluate_efficiency_market
from nfl.recent_form_dataset import build_recent_form_walkforward_dataset


def main() -> None:
    client = NFLDataClient()
    seasons = [2022, 2023, 2024, 2025]
    schedules = client.load_schedules([2021, *seasons])
    pbp = client.load_pbp(seasons)
    datasets = [
        build_recent_form_walkforward_dataset(
            schedules, pbp, season, start_week=5
        ) for season in seasons
    ]
    frame = pl.concat(datasets, how="vertical_relaxed")
    frame = frame.join(
        schedules.select("season", "week", "game_id", "spread_line", "total_line"),
        on=["season", "week", "game_id"], how="left", validate="1:1",
    )
    report = evaluate_efficiency_market(frame)
    out = Path("reports/efficiency_market_challenger.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["targets"], sort_keys=True))


if __name__ == "__main__":
    main()
