"""Grade the frozen recent-form totals prospective prediction ledger."""

from __future__ import annotations

import json

import polars as pl

from nfl.data import NFLDataClient
from nfl.recent_form_forward import (
    load_recent_form_forward_predictions,
    write_recent_form_forward_report,
)


def main() -> None:
    predictions = load_recent_form_forward_predictions()
    if predictions.is_empty():
        schedules = pl.DataFrame()
    else:
        seasons = sorted(
            {
                int(value)
                for value in predictions.get_column("season").drop_nulls().to_list()
            }
        )
        schedules = NFLDataClient().load_schedules(seasons, refresh=True)
    summary = write_recent_form_forward_report(schedules)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
