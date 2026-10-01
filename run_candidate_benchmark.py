from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.candidate_benchmark import run_candidate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rolling-origin NFL predictive candidate benchmark."
    )
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/candidate_benchmark.json")
    args = parser.parse_args()

    report = run_candidate_benchmark(
        start_week=args.start_week,
        end_week=args.end_week,
        refresh=args.refresh,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
