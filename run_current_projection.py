"""Generate independent baseline and QB shadow projections for upcoming NFL games."""

from __future__ import annotations

import argparse
import json

from nfl.contracts import DataContractError
from nfl.current import run_current_projection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--training-seasons",
        nargs="+",
        type=int,
        help="Historical seasons used to fit the frozen QB residual structures",
    )
    parser.add_argument("--score-ridge", type=float, default=8.0)
    parser.add_argument("--qb-prior-dropbacks", type=float, default=75.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--blocked-report",
        action="store_true",
        help=(
            "Emit a machine-readable BLOCKED report instead of a nonzero exit when "
            "live source data violates a projection contract"
        ),
    )
    args = parser.parse_args()

    training_seasons = (
        tuple(args.training_seasons) if args.training_seasons is not None else None
    )
    try:
        projection, audit = run_current_projection(
            args.season,
            args.week,
            training_seasons=training_seasons,
            score_ridge=args.score_ridge,
            qb_prior_dropbacks=args.qb_prior_dropbacks,
            refresh=args.refresh,
        )
    except DataContractError as exc:
        if not args.blocked_report:
            raise
        result = {
            "status": "BLOCKED",
            "reason": str(exc),
            "season": args.season,
            "requested_week": args.week,
            "games": [],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    result = {
        "status": "READY",
        "audit": audit.to_dict(),
        "games": projection.to_dicts(),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
