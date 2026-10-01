"""Run the chronological NFL home-win probability audit."""

from __future__ import annotations

import argparse
import json

from nfl.win_probability_audit import run_win_probability_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    audit = run_win_probability_audit(
        validation_season=args.validation_season,
        holdout_season=args.holdout_season,
        refresh=args.refresh,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
