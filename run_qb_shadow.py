"""Run the frozen quarterback adjustment on completed 2026 shadow games."""

from __future__ import annotations

import argparse
import json

from nfl.qb_shadow import run_qb_shadow_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-season", type=int, default=2026)
    parser.add_argument("--shadow-start-week", type=int, default=3)
    parser.add_argument("--score-ridge", type=float, default=8.0)
    parser.add_argument("--qb-prior-dropbacks", type=float, default=75.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_qb_shadow_audit(
        shadow_season=args.shadow_season,
        shadow_start_week=args.shadow_start_week,
        score_ridge=args.score_ridge,
        qb_prior_dropbacks=args.qb_prior_dropbacks,
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
