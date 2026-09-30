"""Run the NFL chronological walk-forward reconstruction audit."""

from __future__ import annotations

import argparse
import json

from nfl.walkforward import run_walkforward_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int, help="NFL season, e.g. 2025")
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--end-week", type=int, default=10)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore local parquet cache and redownload source data",
    )
    args = parser.parse_args()

    audit = run_walkforward_audit(
        args.season,
        start_week=args.start_week,
        end_week=args.end_week,
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
