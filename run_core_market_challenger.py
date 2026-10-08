"""Reconstruct fixed core challenger, compared to archived final lines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.core_market_challenger import compare_core_challenger
from nfl.data import NFLDataClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/core_market_challenger.json")
    args = parser.parse_args()
    schedules = NFLDataClient().load_schedules([2023, 2024, 2025])
    result = compare_core_challenger(schedules)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["by_season"], sort_keys=True))


if __name__ == "__main__":
    main()
