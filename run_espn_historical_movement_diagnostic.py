"""Probe ESPN's public historical NFL odds movement endpoint.

This diagnostic is intentionally read-only. It discovers a completed NFL event from
the ESPN scoreboard, resolves the Core odds provider, then inspects the provider's
movement history shape for timestamped line/price data.
"""

from __future__ import annotations

import argparse
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
ESPN_CORE_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/"
    "{event_id}/competitions/{event_id}/odds"
)
ESPN_MOVEMENT_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/"
    "{event_id}/competitions/{event_id}/odds/{provider_id}/history/0/movement"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HarbinNFLAnalytics/0.1)",
    "Accept": "application/json,text/plain,*/*",
}


def _request_json(
    url: str,
    *,
    params: dict[str, object] | None = None,
    timeout: float = 20.0,
) -> tuple[object | None, dict[str, object]]:
    target = url if not params else f"{url}?{urlencode(params)}"
    request = Request(target, headers=HEADERS)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = int(getattr(response, "status", 200))
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        return None, {"ok": False, "status": exc.code, "error": str(exc)}
    except (URLError, TimeoutError) as exc:
        return None, {"ok": False, "status": None, "error": str(exc)}
    try:
        return json.loads(raw), {"ok": True, "status": status, "error": None}
    except json.JSONDecodeError as exc:
        return None, {
            "ok": False,
            "status": status,
            "error": f"JSONDecodeError: {exc}",
        }


def _shape(value: object, *, depth: int = 0) -> object:
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        output: dict[str, object] = {"keys": sorted(value.keys())}
        for key, nested in value.items():
            if isinstance(nested, (dict, list)):
                output[key] = _shape(nested, depth=depth + 1)
            elif key.lower() in {
                "id",
                "name",
                "date",
                "timestamp",
                "updated",
                "lastupdated",
                "spread",
                "overunder",
                "moneyline",
                "homemoneyline",
                "awaymoneyline",
                "odds",
                "price",
                "value",
            }:
                output[key] = nested
        return output
    if isinstance(value, list):
        return {
            "count": len(value),
            "first": _shape(value[0], depth=depth + 1) if value else None,
        }
    return type(value).__name__


def _events(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    values = payload.get("events")
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _resolve_item(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    reference = value.get("$ref")
    if not reference:
        return value
    payload, meta = _request_json(str(reference).replace("http://", "https://"))
    if not meta["ok"] or not isinstance(payload, dict):
        return None
    return payload


def _provider_id(value: dict[str, object]) -> str | None:
    provider = value.get("provider")
    if isinstance(provider, dict):
        for key in ("id", "uid"):
            candidate = str(provider.get(key) or "").strip()
            if candidate:
                match = re.search(r"(\d+)$", candidate)
                return match.group(1) if match else candidate
        reference = str(provider.get("$ref") or "")
        match = re.search(r"/providers/(\d+)", reference)
        if match:
            return match.group(1)

    for key in ("providerId", "provider_id"):
        candidate = str(value.get(key) or "").strip()
        if candidate:
            return candidate

    reference = str(value.get("$ref") or "")
    match = re.search(r"/odds/(\d+)", reference)
    return match.group(1) if match else None


def _provider_name(value: dict[str, object]) -> str:
    provider = value.get("provider")
    if isinstance(provider, dict):
        return str(provider.get("name") or provider.get("id") or "")
    return str(provider or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, default=1)
    args = parser.parse_args()

    scoreboard, scoreboard_meta = _request_json(
        ESPN_SCOREBOARD_URL,
        params={
            "limit": 100,
            "dates": str(args.season),
            "seasontype": 2,
            "week": args.week,
        },
    )
    events = _events(scoreboard)
    output: dict[str, object] = {
        "season": args.season,
        "week": args.week,
        "scoreboard_request": scoreboard_meta,
        "scoreboard_events": len(events),
    }
    if not events:
        output["status"] = "NO_EVENTS"
        print(json.dumps(output, indent=2, sort_keys=True, default=str))
        return

    event = events[0]
    event_id = str(event.get("id") or "")
    output["event_id"] = event_id
    output["event_shape"] = _shape(event)

    core, core_meta = _request_json(
        ESPN_CORE_ODDS_URL.format(event_id=event_id),
        params={"limit": 50},
    )
    output["core_request"] = core_meta
    output["core_shape"] = _shape(core)
    items = core.get("items") if isinstance(core, dict) else None
    raw_items = items if isinstance(items, list) else []
    resolved = [
        item
        for raw in raw_items
        if (item := _resolve_item(raw)) is not None
    ]
    output["core_items"] = len(raw_items)
    output["resolved_items"] = len(resolved)
    output["providers"] = [
        {
            "id": _provider_id(item),
            "name": _provider_name(item),
            "shape": _shape(item),
        }
        for item in resolved[:10]
    ]

    probes: list[dict[str, object]] = []
    for item in resolved[:10]:
        provider_id = _provider_id(item)
        if not provider_id:
            continue
        movement, movement_meta = _request_json(
            ESPN_MOVEMENT_URL.format(
                event_id=event_id,
                provider_id=provider_id,
            ),
            params={"limit": 100},
        )
        probes.append(
            {
                "provider_id": provider_id,
                "provider_name": _provider_name(item),
                "request": movement_meta,
                "shape": _shape(movement),
                "payload": movement,
            }
        )
    output["movement_probes"] = probes
    output["status"] = (
        "READY"
        if any(
            isinstance(probe.get("request"), dict)
            and probe["request"].get("ok")
            for probe in probes
        )
        else "NO_MOVEMENT"
    )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
