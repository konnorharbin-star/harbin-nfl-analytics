"""Audit fixed NFL model-market disagreement bands."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_market_relative import build_market_relative_games
from nfl.market_disagreement_diagnostics import audit_disagreements


def main() -> None:
    projections = pl.read_csv("reports/free_market_predictions.csv")
    seasons = sorted(projections["season"].unique().to_list())
    schedule = NFLDataClient().load_schedules(seasons)
    paired = build_market_relative_games(projections, schedule)
    results = audit_disagreements(paired)
    output = Path("reports/market_disagreement_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "margin_games": results["targets"]["margin"]["covered_games"],
        "total_games": results["targets"]["total"]["covered_games"],
        "status": results["status"],
    }))


if __name__ == "__main__":
    main()
