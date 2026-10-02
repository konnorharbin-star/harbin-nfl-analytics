from __future__ import annotations

import argparse
import json

from nfl.data import NFLDataClient
from nfl.forward_ledger_health import write_forward_ledger_health


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit prospective NFL shadow ledger timing and capture coverage."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.season != 2026:
        raise ValueError("Stage 33 forward ledger health is frozen to the 2026 season")

    schedules = NFLDataClient().load_schedules([args.season], refresh=args.refresh)
    report = write_forward_ledger_health(schedules)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
