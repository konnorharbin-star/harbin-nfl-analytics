"""Study or monitor the video-inspired clean efficiency experiment, free only."""

import argparse
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl.clean_efficiency_research import REQUIRED, SPEC, aggregate, examples, predict, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("study", "forecast"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("research/clean_efficiency_frozen.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/clean_efficiency_research.json"))
    args = parser.parse_args()
    years = [2022, 2023, 2024, 2025] if args.mode == "study" else [2025, 2026]
    schedules = (
        pl.read_parquet(args.source_dir / "schedules.parquet")
        if args.source_dir
        else nfl.load_schedules(years)
    )
    schedules = schedules.filter(pl.col("season").is_in(years)).select(
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "gameday",
        "gametime",
    )
    frames = [
        pl.read_parquet(args.source_dir / f"pbp_{y}.parquet")
        if args.source_dir
        else nfl.load_pbp([y]).select(sorted(REQUIRED))
        for y in years
    ]
    plays = pl.concat(frames, how="vertical_relaxed").select(sorted(REQUIRED))
    totals, exclusions = aggregate(schedules, plays)
    captured = datetime.now(UTC)
    source = {
        "captured_utc": captured.isoformat(),
        "source_urls": [
            f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
            for y in years
        ],
        "schedule_url": "https://github.com/nflverse/nfldata/blob/master/data/games.csv",
        "normalized_play_sha256": hashlib.sha256(plays.write_json().encode()).hexdigest(),
        "normalized_schedule_sha256": hashlib.sha256(schedules.write_json().encode()).hexdigest(),
        "source_publication_vintages_verified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    replay = json.dumps(
        {"schedules": schedules.to_dicts(), "plays": plays.to_dicts()},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode()
    snapshot = args.out.with_suffix(".source.json.gz")
    snapshot.write_bytes(gzip.compress(replay, mtime=0))
    source["replay_snapshot_sha256"] = hashlib.sha256(replay).hexdigest()
    source["replay_snapshot_file"] = str(snapshot)
    if args.mode == "study":
        rows, errors = examples(schedules, totals)
        report, models = study(rows)
        frozen = dict(
            spec=SPEC,
            training_seasons=[2023, 2024],
            models=models,
            source=source,
            filters="regulation pass/run; no kneels/spikes; abs score diff<=14; half time>120",
            prior_window_games=8,
            shrinkage_plays=100,
            weights_selected_without_2025=True,
        )
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.model.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
        report["blocks"] = {
            "fit": "2023 weeks5-18",
            "tune": "2024 weeks5-18",
            "diagnostic": "previously inspected 2025 weeks5-18",
        }
    else:
        rows, errors = examples(schedules, totals, seasons=(2026,), at=captured)
        frozen = json.loads(args.model.read_text())
        if frozen["spec"] != SPEC or frozen["training_seasons"] != [2023, 2024]:
            raise ValueError("INVALID_FROZEN_LINEAGE")
        forecasts = []
        for row in rows:
            forecasts.append(
                row
                | {
                    "challenger_" + t: predict(
                        row, frozen["models"][t]["model"], frozen["models"][t]["blend"]
                    )
                    for t in ("margin", "total")
                }
            )
        report = dict(
            spec=SPEC,
            forecasts=forecasts,
            model_file_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            market_accuracy_proven=False,
            validated_edges=0,
            production_action="PASS",
            used_for_bet_ranking=False,
            player_props_updated=False,
        )
    report.update(
        source=source,
        exclusions=exclusions,
        errors=errors,
        video_url="https://www.youtube.com/watch?v=LzcTB0lOF74",
        video_evidence="Publisher description and chapter outline; transcript unavailable",
        coefficients_from_video=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "spec": SPEC,
                "mode": args.mode,
                "forecast_games": len(report.get("forecasts", [])),
                "errors": len(errors),
                "results": report.get("results"),
                "validated_edges": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
