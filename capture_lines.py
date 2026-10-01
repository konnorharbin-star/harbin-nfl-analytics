"""Capture verified pre-kickoff NFL market snapshots from canonical current sources."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl.current import next_unplayed_regular_week, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.line_history import append_market_snapshots
from nfl.pro_market import collect_current_markets


def _default_season() -> int:
    now = datetime.now(UTC)
    return now.year - 1 if now.month <= 2 else now.year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=_default_season())
    parser.add_argument("--week", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    client = NFLDataClient()
    schedules = client.load_schedules([args.season], refresh=args.refresh)
    week = args.week or next_unplayed_regular_week(schedules, args.season)
    targets = unplayed_regular_games(schedules, args.season, week)
    markets, source_meta = collect_current_markets(targets, week=week)
    report = append_market_snapshots(markets, targets)
    payload = {
        "status": "READY",
        "season": args.season,
        "week": week,
        "source": "canonical current NFL market aggregation",
        "source_breadth": source_meta,
        **report,
    }
    target = Path("outputs/line_capture_status.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
