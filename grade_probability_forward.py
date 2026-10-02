from __future__ import annotations

import json

from nfl.data import NFLDataClient
from nfl.probability_forward import write_probability_forward_report


def main() -> None:
    source = NFLDataClient()
    schedules = source.load_schedules([2026], refresh=True)
    summary = write_probability_forward_report(schedules)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
