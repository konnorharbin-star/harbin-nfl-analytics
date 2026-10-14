"""Publish a separate score-only regulation/OT challenger from free football inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl

from nfl.coupled_nflverse_asof import kickoff_utc
from nfl.overtime_source import regulation_history, score_challenger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/overtime_score_challenger.json"))
    args = parser.parse_args()
    report = {
        "status": "RESEARCH_ONLY_NO_VALIDATED_EDGE",
        "games": [],
        "used_for_bet_ranking": False,
        "player_props_updated": False,
        "original_forecasts_unchanged": True,
        "promotion_earned": False,
    }
    try:
        if args.season not in (2025, 2026):
            raise ValueError("regular-season overtime rule lineage supports 2025/2026 only")
        schedules = nfl.load_schedules([args.season - 1, args.season])
        pbp = nfl.load_pbp([args.season - 1, args.season])
        captured = datetime.now(UTC)
        report["capture_utc"] = captured.isoformat()
        games, kernel, audit = regulation_history(schedules, pbp, at=captured, season=args.season)
        report["source_audit"] = audit
        report["source_snapshot"] = {"games": games, "kernel": [vars(event) for event in kernel]}
        future = [
            g
            for g in schedules.to_dicts()
            if g.get("season") == args.season
            and g.get("game_type") == "REG"
            and kickoff_utc(g) > captured
        ]
        week = args.week if args.week is not None else min((g["week"] for g in future), default=-1)
        report["season"], report["week"] = args.season, week
        for target in future:
            if target["week"] != week:
                continue
            try:
                report["games"].append(score_challenger(games, kernel, target, draws=args.draws))
            except (ValueError, RuntimeError) as error:
                report["games"].append(
                    {
                        "game_id": target["game_id"],
                        "status": "SOURCE_OR_MODEL_BLOCKED",
                        "reason": str(error),
                    }
                )
    except Exception as error:
        report.update(status="SOURCE_OR_MODEL_BLOCKED", reason=f"{type(error).__name__}: {error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "games": len(report["games"]),
                "artifact": str(args.out),
                "reason": report.get("reason"),
            }
        )
    )
    if report["status"] == "SOURCE_OR_MODEL_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
