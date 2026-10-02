"""Measure free ESPN archived open/close market coverage across completed NFL weeks."""

from __future__ import annotations

import argparse
import json

import polars as pl

from nfl.data import NFLDataClient, completed_games
from nfl.espn_historical import ESPNArchiveClient, fetch_espn_archive_quotes


def _projection_targets(
    schedules: pl.DataFrame,
    *,
    season: int,
    week: int,
) -> pl.DataFrame:
    games = completed_games(schedules).filter(
        (pl.col("game_type") == "REG")
        & (pl.col("season") == season)
        & (pl.col("week") == week)
    )
    rows: list[dict[str, object]] = []
    for row in games.iter_rows(named=True):
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])
        rows.append(
            {
                "season": season,
                "week": week,
                "game_id": str(row["game_id"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "projected_home_margin": 0.0,
                "projected_total": 0.0,
                "actual_home_margin": home_score - away_score,
                "actual_total": home_score + away_score,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    seasons = sorted(
        {
            int(value.strip())
            for value in str(args.seasons).split(",")
            if value.strip()
        }
    )
    if not seasons:
        raise SystemExit("--seasons must include at least one season")

    schedules = NFLDataClient().load_schedules(seasons, refresh=True)
    output: dict[str, object] = {
        "week": args.week,
        "seasons": seasons,
        "samples": {},
    }
    total_games = 0
    total_open = 0
    total_close = 0
    total_entry_pairs = 0
    total_closing_pairs = 0
    books: set[str] = set()

    for season in seasons:
        targets = _projection_targets(
            schedules,
            season=season,
            week=args.week,
        )
        if targets.is_empty():
            output["samples"][str(season)] = {
                "status": "NO_COMPLETED_GAMES",
            }
            continue
        entries, closings, coverage = fetch_espn_archive_quotes(
            targets,
            client=ESPNArchiveClient(max_workers=args.max_workers),
        )
        sample = coverage.to_dict()
        sample["open_game_coverage"] = (
            coverage.games_with_open / coverage.requested_games
            if coverage.requested_games
            else 0.0
        )
        sample["close_game_coverage"] = (
            coverage.games_with_close / coverage.requested_games
            if coverage.requested_games
            else 0.0
        )
        sample["qualified_for_open_close_evidence"] = (
            sample["open_game_coverage"] >= 0.90
            and sample["close_game_coverage"] >= 0.90
        )
        sample["entry_rows"] = entries.height
        sample["closing_rows"] = closings.height
        sample["entry_markets"] = (
            entries.group_by("market_type").len().to_dicts()
            if not entries.is_empty()
            else []
        )
        sample["closing_markets"] = (
            closings.group_by("market_type").len().to_dicts()
            if not closings.is_empty()
            else []
        )
        output["samples"][str(season)] = sample
        total_games += coverage.requested_games
        total_open += coverage.games_with_open
        total_close += coverage.games_with_close
        total_entry_pairs += coverage.entry_pairs
        total_closing_pairs += coverage.closing_pairs
        books.update(coverage.books)

    samples = output["samples"]
    qualified_seasons = sorted(
        int(season)
        for season, sample in samples.items()
        if isinstance(sample, dict)
        and sample.get("qualified_for_open_close_evidence") is True
    )
    output["summary"] = {
        "requested_games": total_games,
        "games_with_open": total_open,
        "games_with_close": total_close,
        "open_game_coverage": (
            total_open / total_games if total_games else 0.0
        ),
        "close_game_coverage": (
            total_close / total_games if total_games else 0.0
        ),
        "entry_pairs": total_entry_pairs,
        "closing_pairs": total_closing_pairs,
        "books": sorted(books),
        "qualified_seasons": qualified_seasons,
    }
    output["status"] = (
        "READY"
        if len(qualified_seasons) >= 2
        else "INSUFFICIENT_COVERAGE"
    )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
