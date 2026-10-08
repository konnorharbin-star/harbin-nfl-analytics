"""Run predeclared possessions × efficiency NFL market benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.factorized_market_benchmark import evaluate_factorized_market


def main() -> None:
    client = NFLDataClient()
    schedules = client.load_schedules([2023, 2024, 2025])
    pbp = client.load_pbp([2023, 2024, 2025])
    result = evaluate_factorized_market(schedules, pbp)
    output = Path("reports/factorized_market_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["by_season"], sort_keys=True))


if __name__ == "__main__":
    main()
