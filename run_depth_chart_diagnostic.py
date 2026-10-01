"""Inspect the live nflverse depth-chart schema without mutating model state."""

from __future__ import annotations

import argparse
import json

import nflreadpy as nfl
import polars as pl


def _sample_values(frame: pl.DataFrame, column: str, limit: int = 8) -> list[object]:
    values: list[object] = []
    for value in frame.get_column(column).drop_nulls().head(limit).to_list():
        if isinstance(value, (str, int, float, bool)) or value is None:
            values.append(value)
        else:
            values.append(str(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", nargs="?", type=int, default=2026)
    args = parser.parse_args()

    frame = nfl.load_depth_charts([args.season])
    if not isinstance(frame, pl.DataFrame):
        raise RuntimeError(
            f"nflreadpy depth-chart loader returned {type(frame).__name__}, not Polars"
        )

    interesting = [
        column
        for column in frame.columns
        if any(
            token in column.lower()
            for token in (
                "season",
                "week",
                "date",
                "dt",
                "team",
                "club",
                "player",
                "name",
                "gsis",
                "espn",
                "pos",
                "depth",
                "rank",
            )
        )
    ]
    meta = {
        column: {
            "dtype": str(frame.schema[column]),
            "non_null": frame.get_column(column).drop_nulls().len(),
            "sample_values": _sample_values(frame, column),
        }
        for column in interesting
    }
    output = {
        "requested_season": args.season,
        "rows": frame.height,
        "columns": frame.columns,
        "interesting_columns": meta,
        "has_season": "season" in frame.columns,
        "has_team": "team" in frame.columns,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
