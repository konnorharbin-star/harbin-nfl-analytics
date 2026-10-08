"""Audit independent NFL fair scores against archived final market lines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_market_relative import (
    build_market_relative_games,
    evaluate_market_relative_games,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/free_market_predictions.csv")
    parser.add_argument("--output", default="reports/game_intelligence_market_relative.json")
    args = parser.parse_args()
    predictions = pl.read_csv(args.predictions)
    years = sorted(int(year) for year in predictions["season"].unique().to_list())
    schedules = NFLDataClient().load_schedules(years)
    games = build_market_relative_games(predictions, schedules)
    report = evaluate_market_relative_games(games)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    games.write_csv(output.with_suffix(".csv"))
    print(json.dumps(report["targets"], sort_keys=True))


if __name__ == "__main__":
    main()
