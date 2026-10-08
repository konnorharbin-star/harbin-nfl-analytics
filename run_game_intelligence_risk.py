"""Run NFL QB, OL and turnover forensics using historical frozen snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.data import NFLDataClient
from nfl.game_intelligence_risk import (
    assemble_pregame_risk,
    attach_postgame_turnovers,
    summarize_risk_attribution,
)
from nfl.personnel_context import build_personnel_walkforward_dataset
from nfl.qb_dataset import build_qb_walkforward_dataset
from nfl.recent_form_dataset import build_recent_form_walkforward_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output", default="reports/game_intelligence_risk.json"
    )
    args = parser.parse_args()
    if args.start_season > args.end_season:
        raise ValueError("invalid season range")
    seasons = list(range(args.start_season, args.end_season + 1))
    client = NFLDataClient()
    schedules = client.load_schedules(
        list(range(args.start_season - 1, args.end_season + 1)),
        refresh=args.refresh,
    )
    pbp = client.load_pbp(seasons, refresh=args.refresh)
    player_stats = client.load_player_stats(seasons, refresh=args.refresh)

    baseline_frames: list[pl.DataFrame] = []
    qb_frames: list[pl.DataFrame] = []
    personnel_frames: list[pl.DataFrame] = []
    source_issues: dict[str, str] = {}
    for season in seasons:
        baseline_frames.append(build_recent_form_walkforward_dataset(
            schedules, pbp, season, start_week=5
        ))
        try:
            qb_frames.append(build_qb_walkforward_dataset(
                schedules,
                player_stats.filter(pl.col("season") == season),
                season, start_week=5,
            ))
        except Exception as exc:
            # Missing QB source remains an explicit uncovered exposure,
            # not a zero-change QB, and is reported for inspection.
            source_issues[f"qb_{season}"] = f"{type(exc).__name__}: {exc}"
        try:
            injury = client.load_injuries([season], refresh=args.refresh)
            depth = client.load_depth_charts([season], refresh=args.refresh)
            personnel = build_personnel_walkforward_dataset(
                schedules, injury, depth, season, start_week=5
            )
            personnel_frames.append(personnel)
        except Exception as exc:
            source_issues[f"ol_{season}"] = f"{type(exc).__name__}: {exc}"

    if not qb_frames:
        raise RuntimeError("no QB sources available; cannot run risk audit")
    baseline = pl.concat(baseline_frames, how="vertical_relaxed")
    qb = pl.concat(qb_frames, how="vertical_relaxed")
    personnel = (
        pl.concat(personnel_frames, how="diagonal_relaxed")
        if personnel_frames else None
    )
    games = assemble_pregame_risk(baseline, qb, personnel)
    games, turnover_status = attach_postgame_turnovers(games, pbp)
    report = summarize_risk_attribution(games, turnover_status=turnover_status)
    report["source_errors"] = source_issues
    report["seasons"] = seasons
    report["known_qb_games"] = games.filter(
        pl.col("last_observed_qb_switch").is_not_null()
    ).height
    report["known_ol_games"] = games.filter(
        pl.col("pregame_ol_injury_stress").is_not_null()
    ).height

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    games.write_csv(output.with_suffix(".csv"))
    print(json.dumps({
        "status": report["status"],
        "games": report["games"],
        "known_qb_games": report["known_qb_games"],
        "known_ol_games": report["known_ol_games"],
        "turnover_source": report["turnover_source_status"],
        "source_errors": source_issues,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
