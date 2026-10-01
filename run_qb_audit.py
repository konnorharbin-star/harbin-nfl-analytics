"""Run the nested holdout audit for last-observed quarterback-state features."""

from __future__ import annotations

import argparse
import json

from nfl.qb_audit import run_qb_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--score-ridge", type=float, default=8.0)
    parser.add_argument("--qb-prior-dropbacks", type=float, default=75.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_qb_audit(
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
        start_week=args.start_week,
        end_week=args.end_week,
        score_ridge=args.score_ridge,
        qb_prior_dropbacks=args.qb_prior_dropbacks,
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
