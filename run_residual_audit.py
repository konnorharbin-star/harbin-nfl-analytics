"""Run the multi-season nested residual-model audit."""

from __future__ import annotations

import argparse
import json

from nfl.residual_audit import run_residual_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024, 2025],
        help="Chronological seasons used to build the modeling dataset",
    )
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_residual_audit(
        seasons=tuple(args.seasons),
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
        start_week=args.start_week,
        end_week=args.end_week,
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
