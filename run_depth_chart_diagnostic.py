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


def _date_scope(frame: pl.DataFrame) -> dict[str, object]:
    if "dt" not in frame.columns or frame.is_empty():
        return {"min_dt": None, "max_dt": None, "dt_year_counts": {}}
    parsed = frame.select(
        pl.col("dt")
        .cast(pl.String)
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("parsed_dt")
    ).drop_nulls()
    if parsed.is_empty():
        return {"min_dt": None, "max_dt": None, "dt_year_counts": {}}
    counts = (
        parsed.with_columns(pl.col("parsed_dt").dt.year().alias("year"))
        .group_by("year")
        .len()
        .sort("year")
    )
    return {
        "min_dt": str(parsed.get_column("parsed_dt").min()),
        "max_dt": str(parsed.get_column("parsed_dt").max()),
        "dt_year_counts": {
            str(row["year"]): int(row["len"])
            for row in counts.iter_rows(named=True)
        },
    }


def _frame_summary(frame: pl.DataFrame, requested_season: int) -> dict[str, object]:
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
    return {
        "requested_season": requested_season,
        "rows": frame.height,
        "columns": frame.columns,
        "interesting_columns": meta,
        "has_season": "season" in frame.columns,
        "has_team": "team" in frame.columns,
        "date_scope": _date_scope(frame),
        "teams": (
            sorted(str(value) for value in frame.get_column("team").unique().to_list())
            if "team" in frame.columns
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", nargs="?", type=int, default=2026)
    args = parser.parse_args()

    seasons = sorted({args.season - 1, args.season})
    summaries: dict[str, object] = {}
    for season in seasons:
        summaries[str(season)] = _frame_summary(
            nfl.load_depth_charts([season]),
            season,
        )

    output = {
        "target_season": args.season,
        "single_season_requests": summaries,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
