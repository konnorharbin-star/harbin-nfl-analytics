from __future__ import annotations

import argparse
import json

from nfl.current import run_current_projection, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.free_market_backtest import build_archive_projection_dataset
from nfl.probability_forward import (
    PROBABILITY_TRAINING_SEASONS,
    append_probability_forward_predictions,
    attach_probability_shadow,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", nargs="?", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source = NFLDataClient()
    projection, audit = run_current_projection(
        args.season,
        client=source,
        refresh=args.refresh,
    )
    schedule_start = min(PROBABILITY_TRAINING_SEASONS) - 1
    schedules = source.load_schedules(
        list(range(schedule_start, args.season + 1)),
        refresh=False,
    )
    targets = unplayed_regular_games(schedules, args.season, audit.week)
    historical = build_archive_projection_dataset(
        schedules,
        start_season=min(PROBABILITY_TRAINING_SEASONS),
        end_season=max(PROBABILITY_TRAINING_SEASONS),
    )
    shadowed, shadow_meta = attach_probability_shadow(projection, historical)
    capture = append_probability_forward_predictions(shadowed, targets)

    print(
        json.dumps(
            {
                "projection_audit": audit.to_dict(),
                "probability_shadow": shadow_meta,
                "capture": capture,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
