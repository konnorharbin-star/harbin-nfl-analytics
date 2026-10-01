"""Run the validation/holdout audit for recency-weighted fair-score ratings."""

from __future__ import annotations

import argparse
import json

from nfl.recency_audit import run_recency_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=8.0)
    parser.add_argument(
        "--half-lives",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0, 8.0, 12.0],
        help="Candidate exponential half-lives in weeks; selected on validation only",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_recency_audit(
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
        start_week=args.start_week,
        ridge=args.ridge,
        half_life_grid=tuple(args.half_lives),
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
