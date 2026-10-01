"""Inspect live ESPN NFL event matching and odds response shape without changing model state."""

from __future__ import annotations

import argparse
import json
from typing import Any

from nfl.current import next_unplayed_regular_week, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.espn_market import (
    ESPN_SCOREBOARD_URL,
    ESPNMarketClient,
    _event_teams,
    parse_espn_odds,
)


def _odds_shape(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    output: dict[str, object] = {"keys": sorted(value.keys())}
    for key in ("provider", "homeTeamOdds", "awayTeamOdds"):
        nested = value.get(key)
        if isinstance(nested, dict):
            output[f"{key}_keys"] = sorted(nested.keys())
    return output


def _inspect_payload(
    payload: dict[str, object],
    *,
    targets: dict[tuple[str, str], dict[str, Any]],
    client: ESPNMarketClient,
) -> dict[str, object]:
    events = payload.get("events")
    if not isinstance(events, list):
        return {"event_count": 0, "error": "events missing or not a list"}

    scoreboard_pairs: list[str] = []
    matched: list[dict[str, object]] = []
    matched_pairs: set[tuple[str, str]] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        teams = _event_teams(event)
        if teams is None:
            continue
        scoreboard_pairs.append(f"{teams[1]}@{teams[0]}")
        target = targets.get(teams)
        if target is None:
            continue
        matched_pairs.add(teams)
        event_id = str(event.get("id") or "")
        competitions = event.get("competitions")
        competition = competitions[0] if isinstance(competitions, list) and competitions else {}
        scoreboard_odds = competition.get("odds") if isinstance(competition, dict) else None
        scoreboard_objects = [
            item for item in scoreboard_odds if isinstance(item, dict)
        ] if isinstance(scoreboard_odds, list) else []
        parsed_scoreboard = []
        for odds in scoreboard_objects:
            parsed_scoreboard.extend(
                parse_espn_odds(
                    odds,
                    game_id=str(target["game_id"]),
                    event_id=event_id,
                    home_team=teams[0],
                    away_team=teams[1],
                    captured_at=client_capture,
                )
            )

        core_objects = client._core_odds(event_id) if event_id else []
        parsed_core = []
        for odds in core_objects:
            parsed_core.extend(
                parse_espn_odds(
                    odds,
                    game_id=str(target["game_id"]),
                    event_id=event_id,
                    home_team=teams[0],
                    away_team=teams[1],
                    captured_at=client_capture,
                )
            )
        matched.append(
            {
                "game_id": str(target["game_id"]),
                "pair": f"{teams[1]}@{teams[0]}",
                "event_id": event_id,
                "scoreboard_odds_objects": len(scoreboard_objects),
                "scoreboard_market_types": sorted(
                    {item.market_type for item in parsed_scoreboard}
                ),
                "scoreboard_first_odds_shape": (
                    _odds_shape(scoreboard_objects[0]) if scoreboard_objects else None
                ),
                "core_odds_objects": len(core_objects),
                "core_market_types": sorted({item.market_type for item in parsed_core}),
                "core_first_odds_shape": _odds_shape(core_objects[0]) if core_objects else None,
            }
        )

    return {
        "event_count": len(events),
        "scoreboard_pairs": sorted(scoreboard_pairs),
        "matched_events": len(matched),
        "unmatched_targets": sorted(
            f"{away}@{home}"
            for home, away in targets
            if (home, away) not in matched_pairs
        ),
        "matched": matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, nargs="?", default=2026)
    parser.add_argument("--week", type=int)
    args = parser.parse_args()

    source = NFLDataClient()
    schedules = source.load_schedules([args.season], refresh=True)
    week = args.week or next_unplayed_regular_week(schedules, args.season)
    target_frame = unplayed_regular_games(schedules, args.season, week)
    targets = {
        (str(row["home_team"]), str(row["away_team"])): row
        for row in target_frame.iter_rows(named=True)
    }

    client = ESPNMarketClient(timeout_seconds=15.0)
    global client_capture
    from datetime import UTC, datetime

    client_capture = datetime.now(UTC)
    default_payload = client.scoreboard(week=week)
    pinned_payload = client._json(
        ESPN_SCOREBOARD_URL,
        {"limit": 100, "seasontype": 2, "week": week, "dates": str(args.season)},
    )

    output = {
        "season": args.season,
        "week": week,
        "target_games": target_frame.height,
        "target_pairs": sorted(
            f"{away}@{home}" for home, away in targets
        ),
        "default_scoreboard": _inspect_payload(
            default_payload,
            targets=targets,
            client=client,
        ),
        "season_pinned_scoreboard": _inspect_payload(
            pinned_payload,
            targets=targets,
            client=client,
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
