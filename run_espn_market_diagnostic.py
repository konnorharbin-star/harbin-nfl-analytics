"""Probe live NFL market-source behavior without changing model state."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from nfl.current import next_unplayed_regular_week, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.espn_market import (
    ESPN_CORE_ODDS_URL,
    ESPN_SCOREBOARD_URL,
    _event_teams,
    parse_espn_odds,
)

HEADER_PROFILES: dict[str, dict[str, str]] = {
    "current_client": {"User-Agent": "harbin-nfl-analytics/0.1"},
    "browser_minimal": {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
        )
    },
    "browser_espn": {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/nfl/",
    },
}
MARKET_FIELD_TOKENS = ("spread", "total", "money", "odds", "line")


def _request_json(
    url: str,
    *,
    params: dict[str, object] | None,
    headers: dict[str, str],
    timeout: float = 15.0,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    target = url if not params else f"{url}?{urlencode(params)}"
    request = Request(target, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = int(getattr(response, "status", 200))
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        return None, {"ok": False, "status": exc.code, "error": str(exc)}
    except (URLError, TimeoutError) as exc:
        return None, {"ok": False, "status": None, "error": str(exc)}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {
            "ok": False,
            "status": status,
            "error": f"JSONDecodeError: {exc}",
        }
    if not isinstance(payload, dict):
        return None, {
            "ok": False,
            "status": status,
            "error": f"unexpected payload type {type(payload).__name__}",
        }
    return payload, {"ok": True, "status": status, "error": None}


def _odds_shape(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    output: dict[str, object] = {"keys": sorted(value.keys())}
    for key in ("provider", "homeTeamOdds", "awayTeamOdds", "bettingOdds"):
        nested = value.get(key)
        if isinstance(nested, dict):
            output[f"{key}_keys"] = sorted(nested.keys())
    return output


def _snapshot_values(value: object) -> dict[str, object]:
    """Return compact historical odds snapshots with nested numeric values."""

    def compact(snapshot: object, *, depth: int = 0) -> object:
        if isinstance(snapshot, (str, int, float, bool)) or snapshot is None:
            return snapshot
        if isinstance(snapshot, list):
            if depth >= 3:
                return {"type": "list", "items": len(snapshot)}
            return [compact(item, depth=depth + 1) for item in snapshot[:8]]
        if not isinstance(snapshot, dict):
            return {"type": type(snapshot).__name__}
        if depth >= 4:
            return {"keys": sorted(snapshot.keys())}

        output: dict[str, object] = {}
        for key, raw in snapshot.items():
            if key in {"$ref", "links", "team", "provider"}:
                continue
            if isinstance(raw, (str, int, float, bool)) or raw is None:
                output[key] = raw
                continue
            nested = compact(raw, depth=depth + 1)
            if nested not in ({}, [], None):
                output[key] = nested
        return output or {"keys": sorted(snapshot.keys())}

    output: dict[str, object] = {}
    for label in ("open", "current", "close"):
        if label in value:
            output[label] = compact(value.get(label))

    for side_name in ("homeTeamOdds", "awayTeamOdds", "bettingOdds"):
        side = value.get(side_name)
        if isinstance(side, dict):
            output[side_name] = compact(side)
    return output


def _scoreboard_summary(
    payload: dict[str, object],
    *,
    targets: dict[tuple[str, str], dict[str, Any]],
    captured_at: datetime,
) -> dict[str, object]:
    events = payload.get("events")
    if not isinstance(events, list):
        return {"event_count": 0, "error": "events missing or not a list"}

    matched_pairs: set[tuple[str, str]] = set()
    scoreboard_pairs: list[str] = []
    matched: list[dict[str, object]] = []
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
        competitions = event.get("competitions")
        competition = competitions[0] if isinstance(competitions, list) and competitions else {}
        odds_list = competition.get("odds") if isinstance(competition, dict) else None
        odds_objects = [
            item for item in odds_list if isinstance(item, dict)
        ] if isinstance(odds_list, list) else []
        parsed = []
        for odds in odds_objects:
            parsed.extend(
                parse_espn_odds(
                    odds,
                    game_id=str(target["game_id"]),
                    event_id=str(event.get("id") or ""),
                    home_team=teams[0],
                    away_team=teams[1],
                    captured_at=captured_at,
                )
            )
        matched.append(
            {
                "game_id": str(target["game_id"]),
                "pair": f"{teams[1]}@{teams[0]}",
                "event_id": str(event.get("id") or ""),
                "scoreboard_odds_objects": len(odds_objects),
                "scoreboard_market_types": sorted({row.market_type for row in parsed}),
                "first_odds_shape": _odds_shape(odds_objects[0]) if odds_objects else None,
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


def _first_matched_event_id(summary: dict[str, object]) -> str | None:
    matched = summary.get("matched")
    if not isinstance(matched, list) or not matched:
        return None
    first = matched[0]
    if not isinstance(first, dict):
        return None
    value = str(first.get("event_id") or "").strip()
    return value or None


def _provider_identity(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    provider = value.get("provider")
    if isinstance(provider, dict):
        return str(provider.get("name") or provider.get("id") or "").strip()
    return str(provider or "").strip()


def _core_probe(
    event_id: str,
    *,
    headers: dict[str, str],
) -> dict[str, object]:
    payload, request_meta = _request_json(
        ESPN_CORE_ODDS_URL.format(event_id=event_id),
        params={"limit": 50},
        headers=headers,
    )
    output: dict[str, object] = {"request": request_meta}
    if payload is None:
        return output
    items = payload.get("items")
    output["top_level_keys"] = sorted(payload.keys())
    output["item_count"] = len(items) if isinstance(items, list) else 0
    if not isinstance(items, list) or not items:
        return output

    resolved_items: list[dict[str, object]] = []
    resolution_errors: list[dict[str, object]] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        item = raw
        reference = raw.get("$ref")
        if reference:
            resolved, resolved_meta = _request_json(
                str(reference).replace("http://", "https://"),
                params=None,
                headers=headers,
            )
            if resolved is None:
                resolution_errors.append(
                    {"index": index, "request": resolved_meta}
                )
                continue
            item = resolved
        resolved_items.append(item)

    output["resolved_item_count"] = len(resolved_items)
    output["resolution_errors"] = resolution_errors
    output["provider_identities"] = sorted(
        {
            identity
            for identity in (_provider_identity(item) for item in resolved_items)
            if identity
        }
    )
    output["resolved_shapes"] = [
        _odds_shape(item)
        for item in resolved_items[:5]
    ]
    output["resolved_snapshot_values"] = [
        {
            "provider": _provider_identity(item),
            "snapshots": _snapshot_values(item),
        }
        for item in resolved_items[:5]
    ]
    return output


def _schedule_market_summary(target_frame: pl.DataFrame) -> dict[str, object]:
    market_columns = sorted(
        column
        for column in target_frame.columns
        if any(token in column.lower() for token in MARKET_FIELD_TOKENS)
    )
    column_meta: dict[str, object] = {}
    for column in market_columns:
        series = target_frame.get_column(column)
        non_null = series.drop_nulls()
        values: list[object] = []
        for value in non_null.head(8).to_list():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values.append(value)
            else:
                values.append(str(value))
        column_meta[column] = {
            "dtype": str(target_frame.schema[column]),
            "non_null": non_null.len(),
            "sample_values": values,
        }

    selected = [
        column
        for column in (
            "game_id",
            "away_team",
            "home_team",
            "gameday",
            "gametime",
            *market_columns,
        )
        if column in target_frame.columns
    ]
    rows = target_frame.select(selected).to_dicts() if selected else []
    return {
        "all_schedule_columns": sorted(target_frame.columns),
        "market_like_columns": market_columns,
        "market_column_meta": column_meta,
        "target_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, nargs="?", default=2026)
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Probe completed regular-season games instead of only unplayed targets.",
    )
    args = parser.parse_args()

    source = NFLDataClient()
    schedules = source.load_schedules([args.season], refresh=True)
    week = args.week or next_unplayed_regular_week(schedules, args.season)
    if args.historical:
        target_frame = schedules.filter(
            (pl.col("season") == args.season)
            & (pl.col("week") == week)
            & (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        ).sort("game_id")
    else:
        target_frame = unplayed_regular_games(schedules, args.season, week)
    targets = {
        (str(row["home_team"]), str(row["away_team"])): row
        for row in target_frame.iter_rows(named=True)
    }
    captured_at = datetime.now(UTC)

    profiles: dict[str, object] = {}
    for name, headers in HEADER_PROFILES.items():
        default_payload, default_request = _request_json(
            ESPN_SCOREBOARD_URL,
            params={"limit": 100, "seasontype": 2, "week": week},
            headers=headers,
        )
        pinned_payload, pinned_request = _request_json(
            ESPN_SCOREBOARD_URL,
            params={
                "limit": 100,
                "seasontype": 2,
                "week": week,
                "dates": str(args.season),
            },
            headers=headers,
        )
        default_summary = (
            _scoreboard_summary(default_payload, targets=targets, captured_at=captured_at)
            if default_payload is not None
            else None
        )
        pinned_summary = (
            _scoreboard_summary(pinned_payload, targets=targets, captured_at=captured_at)
            if pinned_payload is not None
            else None
        )
        event_id = None
        if isinstance(pinned_summary, dict):
            event_id = _first_matched_event_id(pinned_summary)
        if event_id is None and isinstance(default_summary, dict):
            event_id = _first_matched_event_id(default_summary)
        profiles[name] = {
            "headers": sorted(headers.keys()),
            "default_request": default_request,
            "default_scoreboard": default_summary,
            "season_pinned_request": pinned_request,
            "season_pinned_scoreboard": pinned_summary,
            "core_probe": _core_probe(event_id, headers=headers) if event_id else None,
        }

    output = {
        "season": args.season,
        "week": week,
        "historical_mode": bool(args.historical),
        "target_games": target_frame.height,
        "target_pairs": sorted(f"{away}@{home}" for home, away in targets),
        "nflverse_schedule": _schedule_market_summary(target_frame),
        "espn_profiles": profiles,
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
