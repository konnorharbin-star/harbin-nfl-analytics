"""Capture the frozen QB-total shadow before kickoff."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from nfl.current import run_current_projection, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.qb_state import QB_PRIOR_DROPBACKS
from nfl.qb_total_forward import append_qb_total_forward_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", nargs="?", type=int, default=2026)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source = NFLDataClient()
    projection, audit = run_current_projection(
        args.season,
        args.week,
        client=source,
        refresh=args.refresh,
    )
    schedules = source.load_schedules([args.season], refresh=args.refresh)
    targets = unplayed_regular_games(schedules, args.season, audit.week)
    result = append_qb_total_forward_predictions(
        projection,
        targets,
        training_seasons=audit.training_seasons,
        qb_prior_dropbacks=QB_PRIOR_DROPBACKS,
        captured_at=datetime.now(UTC),
    )
    result["projection_audit"] = audit.to_dict()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
