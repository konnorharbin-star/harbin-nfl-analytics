"""Run the Stage 2 NFL fair-score and PBP feature audit."""

from __future__ import annotations

import argparse
import json

from nfl.stage2 import run_stage2_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int, help="NFL season, e.g. 2025")
    parser.add_argument("week", type=int, help="Target week; only earlier data may be used")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore local parquet cache and redownload source data",
    )
    args = parser.parse_args()

    audit = run_stage2_audit(args.season, args.week, refresh=args.refresh)
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
