"""Generate completed-game NFL prediction-error intelligence, including 2026."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.free_market_backtest import build_archive_projection_dataset
from nfl.game_intelligence import build_game_intelligence, summarize_game_intelligence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", default="reports/free_market_predictions.csv")
    parser.add_argument("--current-season", type=int, default=2026)
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    history = pl.read_csv(args.historical, try_parse_dates=True)
    frames = [history.filter(pl.col("season") < args.current_season)]
    if not args.skip_current:
        schedules = NFLDataClient().load_schedules(
            [args.current_season - 1, args.current_season], refresh=args.refresh
        )
        try:
            latest = build_archive_projection_dataset(
                schedules, start_season=args.current_season,
                end_season=args.current_season,
            )
            frames.append(latest)
        except Exception:
            # Fail rather than publishing stale or partial 2026 inference.
            raise
    combined = pl.concat(frames, how="diagonal_relaxed")
    graded = build_game_intelligence(combined)
    summary = summarize_game_intelligence(graded)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graded.write_csv(output / "game_intelligence.csv")
    (output / "game_intelligence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps({
        "status": summary["status"],
        "games": graded.height,
        "seasons": sorted(graded["season"].unique().to_list()),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
