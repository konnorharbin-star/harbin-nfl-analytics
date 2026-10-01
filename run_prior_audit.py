"""Run the validation/holdout audit for prior-season fair-score weighting."""

from __future__ import annotations

import argparse
import json

from nfl.prior_audit import run_prior_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=8.0)
    parser.add_argument(
        "--prior-weights",
        nargs="+",
        type=float,
        default=[0.02, 0.05, 0.10, 0.20, 0.35],
        help="Candidate prior-season sample weights selected on validation only",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_prior_audit(
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
        start_week=args.start_week,
        ridge=args.ridge,
        prior_weight_grid=tuple(args.prior_weights),
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
