"""Free TheRundown v2 3-book main-spread watch on frozen football forecasts.

The $0 pre-match feed is five minutes delayed. This is THIRD-PARTY reported
book prices, NOT independently verified sportsbook execution. Emit indicative
numbers only. No paid API, no historical endpoint, no betting actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from scripts.forward_price_validation import decimal, first_commit_time, utc
from scripts.walters_key_market_gate import compare, half_point

SPORT_IDS = {"nfl": 2, "cfb": 1}
BOOKS = {"19": "DraftKings", "22": "BetMGM", "23": "FanDuel"}
BASE_URL = "https://therundown.io/api/v2"
MAX_AGE = timedelta(minutes=15)
MAX_SIDE_SKEW = timedelta(minutes=3)
MAX_BYTES = 3_000_000
HEX = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
NFL_ALIASES = {"LA": "LAR", "JAC": "JAX", "WSH": "WAS"}


def _normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _team_match(forecast_team, provider_team, sport):
    if not isinstance(provider_team, dict):
        return False
    if sport == "nfl":
        lhs = NFL_ALIASES.get(str(forecast_team).upper(), str(forecast_team).upper())
        rhs = str(provider_team.get("abbreviation") or "").upper()
        return bool(lhs) and lhs == NFL_ALIASES.get(rhs, rhs)
    # CFB API appends school mascot; no fuzzy similarity/ranking allowed.
    school = re.sub(r"\s+", " ", str(forecast_team).strip()).lower()
    full = re.sub(r"\s+", " ", str(provider_team.get("name") or "").strip()).lower()
    return bool(school) and (
        _normalized(school) == _normalized(full)
        or full.startswith(school + " ")
    )


def _event_matching(f, ev):
    sport = f["sport"]
    if ev.get("sport_id") != SPORT_IDS[sport]:
        return False
    teams = ev.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        return False
    try:
        correct_time = abs(
            (utc(ev["event_date"]) - utc(f["kickoff"])).total_seconds()
        ) <= 300
    except (ValueError, KeyError, TypeError):
        return False
    return (
        correct_time
        and _team_match(f["away_team"], teams[0], sport)
        and _team_match(f["home_team"], teams[1], sport)
        and teams[0].get("team_id") != teams[1].get("team_id")
        and all(t.get("team_id") is not None for t in teams)
    )


def _price_record(line, book, now):
    price = (line.get("prices") or {}).get(book)
    if not isinstance(price, dict) or price.get("is_main_line") is not True:
        raise ValueError("BOOK_LINE_NOT_MAIN")
    val = price.get("price")
    odds = float(val)
    if odds != int(odds) or abs(odds) < 100 or abs(odds) > 100000:
        raise ValueError("INVALID_OR_OFFBOARD_PRICE")
    decimal(val)
    at = utc(price["updated_at"])
    if at > now or now - at > MAX_AGE:
        raise ValueError("STALE_OR_FUTURE_PROVIDER_PRICE_TIME")
    return {
        "line": half_point(line["value"]),
        "odds": int(odds),
        "reported_updated_at": at,
        "provider_price_id": str(price.get("id") or ""),
        "provider_line_id": str(line.get("id") or ""),
    }


def extract_pair(ev, f, book, now):
    """Fail closed on ambiguous same-book lines or non-opposing handicaps."""
    if not _event_matching(f, ev):
        raise ValueError("EVENT_AND_TEAM_MISMATCH")
    if not now < utc(f["kickoff"]):
        raise ValueError("KICKOFF_NOT_IN_FUTURE")
    selected = []
    for market in ev.get("markets") or []:
        if market.get("market_id") != 2 or market.get("period_id") != 0:
            continue
        participants = market.get("participants") or []
        if len(participants) != 2:
            raise ValueError("SPREAD_MARKET_NOT_TWO_SIDES")
        picked = {}
        for p in participants:
            if p.get("type") != "TYPE_TEAM":
                raise ValueError("NON_TEAM_SPREAD_SIDE")
            team_id = p.get("id")
            for side, team in (
                ("home", ev["teams"][1]), ("away", ev["teams"][0])
            ):
                if team_id != team.get("team_id"):
                    continue
                if side in picked:
                    raise ValueError("DUPLICATE_SPREAD_SIDE")
                main = []
                for line in p.get("lines") or []:
                    try:
                        main.append(_price_record(line, book, now))
                    except (KeyError, ValueError, TypeError):
                        continue
                if len(main) != 1:
                    raise ValueError("MISSING_OR_MULTIPLE_MAIN_LINES")
                picked[side] = main[0]
                break
        if set(picked) != {"home", "away"}:
            raise ValueError("MISSING_OPPOSITE_TEAM")
        if picked["home"]["line"] != -picked["away"]["line"]:
            raise ValueError("DIFFERING_OPPOSITE_HANDICAP")
        timestamps = [v["reported_updated_at"] for v in picked.values()]
        if max(timestamps) - min(timestamps) > MAX_SIDE_SKEW:
            raise ValueError("ASYNCHRONOUS_OPPOSING_QUOTES")
        selected.append(picked)
    if len(selected) != 1:
        raise ValueError("MISSING_OR_MULTIPLE_SPREAD_MARKETS")
    return selected[0]


def _eligible_forecasts(root, sport, now, commit_lookup):
    included, blockers = [], Counter()
    for path in sorted(Path(root).glob("*.json")):
        try:
            f = json.loads(path.read_text(encoding="utf-8"))
            if (
                f.get("spec") != "walters_key_number_forward_v1"
                or f.get("release_state") != "SHADOW"
                or f.get("sport") != sport
                or f.get("game_id") != path.stem
                or not HEX.fullmatch(str(f.get("game_id")))
            ):
                raise ValueError("INVALID_FROZEN_FORECAST")
            kickoff, frozen = utc(f["kickoff"]), utc(f["captured_at"])
            published = commit_lookup(path)
            if published is None or not frozen <= published < now < kickoff:
                raise ValueError("NOT_PRIOR_PUBLISHED_BEFORE_FETCH")
            if kickoff - now > timedelta(days=7):
                raise ValueError("FORECAST_TOO_FAR")
            included.append(f)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers[str(exc)] += 1
    return included, blockers


def _url(sport, date):
    qs = urlencode({
        "market_ids": "2", "affiliate_ids": "19,22,23",
        "main_line": "true", "offset": "300",
    })
    return f"{BASE_URL}/sports/{SPORT_IDS[sport]}/events/{date}?{qs}"


def _fetch(url, key):
    if not key:
        raise ValueError("MISSING_FREE_API_KEY")
    # Avoid any API key in the URL, saved artifacts, stderr or reports.
    with urlopen(
        Request(url, headers={
            "X-TheRundown-Key": key,
            "User-Agent": "Harbin-Walters-Free-Research/1.0",
        }), timeout=14
    ) as response:
        blob = response.read(MAX_BYTES + 1)
    if len(blob) > MAX_BYTES:
        raise ValueError("OVERSIZED_MARKET_RESPONSE")
    result = json.loads(blob)
    if not isinstance(result, dict) or not isinstance(result.get("events"), list):
        raise ValueError("BAD_PROVIDER_EVENT_SHAPE")
    return result, hashlib.sha256(blob).hexdigest()


def scan(sport, forecast_root, evidence_root, *, now=None,
         key=None, request_fn=_fetch, commit_lookup=first_commit_time,
         fetch_enabled=True):
    if sport not in SPORT_IDS:
        raise ValueError("Unsupported sport")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Naive capture clock")
    now = now.astimezone(UTC)
    eligible, exclusions = _eligible_forecasts(
        forecast_root, sport, now, commit_lookup
    )
    summary = {
        "spec": "walters_free_three_book_watch_v1",
        "sport": sport, "observed_at": now.isoformat(),
        "forecasts_with_immutable_pregame_publication": len(eligible),
        "forecast_exclusions": dict(exclusions),
        "provider": "TheRundown_v2_free_pre_match",
        "free_plan_expected_five_minute_delay": True,
        "official_booksite_execution_verified": False,
        "provider_price_update_is_not_independent_sportsbook_certification": True,
        "source_urls_requested": [],
        "source_failures": {},
        "observations_written": 0,
        "research_price_comparisons": [],
        "mode": "NO_KEY_NO_PRICE"
        if not key or not fetch_enabled else "INDICATIVE_SOURCE_OBSERVATIONS",
        "betting_authorized": False,
        "profitability_validated": False,
        "measured_clv": None,
        "measured_roi": None,
    }
    if not key or not fetch_enabled or not eligible:
        return summary
    dates = sorted({
        f["kickoff"][:10] for f in eligible
    } | {
        utc(f["kickoff"]).astimezone(
            ZoneInfo("America/Chicago")
        ).strftime("%Y-%m-%d") for f in eligible
    })
    evidence_root = Path(evidence_root)
    parsed = []
    responses = {}
    for date in dates[:10]:
        url = _url(sport, date)
        summary["source_urls_requested"].append(url)
        try:
            responses[date] = request_fn(url, key)
        except (OSError, ValueError, TimeoutError) as exc:
            summary["source_failures"][date] = type(exc).__name__
    for f in eligible:
        matches = []
        for date, (response, source_hash) in responses.items():
            for ev in response.get("events", []):
                if _event_matching(f, ev):
                    matches.append((ev, date, source_hash))
        unique = {m[0].get("event_id"): m for m in matches}
        if len(unique) != 1:
            exclusions["UNMATCHED_OR_AMBIGUOUS_PROVIDER_EVENT"] += 1
            continue
        ev, date, source_hash = next(iter(unique.values()))
        for book in BOOKS:
            try:
                paired = extract_pair(ev, f, book, now)
                quote = {
                    "quote_id": "indicative",
                    "game_id": f["game_id"], "sport": sport,
                    "source_url": _url(sport, date),
                    "book": BOOKS[book],
                    "home_spread": paired["home"]["line"],
                    "away_spread": paired["away"]["line"],
                    "home_american_odds": paired["home"]["odds"],
                    "away_american_odds": paired["away"]["odds"],
                }
                view = compare(f, quote)
                view["status"] = "INDICATIVE_AGGREGATOR_WATCH_NOT_BET"
                view["provider_event_id"] = str(ev["event_id"])
                view["publisher"] = "TheRundown"
                view["observed_at"] = now.isoformat()
                view["side_provider_updated_at"] = {
                    s: paired[s]["reported_updated_at"].isoformat()
                    for s in ("home", "away")
                }
                view["source_url"] = _url(sport, date)
                view["raw_response_sha256"] = source_hash
                view["provider_book_id"] = book
                view["source_is_not_book_executability_proof"] = True
                view["wager_authorized"] = False
                fingerprint = hashlib.sha256(json.dumps({
                    "event": ev["event_id"], "book": book,
                    "game_id": f["game_id"],
                    "lines": [paired[s]["line"] for s in ("home", "away")],
                    "odds": [paired[s]["odds"] for s in ("home", "away")],
                    "updated": [
                        paired[s]["reported_updated_at"].isoformat()
                        for s in ("home", "away")
                    ],
                }, sort_keys=True).encode()).hexdigest()[:24]
                view["quote_id"] = f["game_id"] + "_" + book + "_" + fingerprint
                view["indicative_quote_only"] = True
                # Skip writing this price until the score forecast is already
                # published; that condition was required above.
                evidence_root.mkdir(parents=True, exist_ok=True)
                target = evidence_root / (view["quote_id"] + ".json")
                if not target.exists():
                    target.write_text(
                        json.dumps(view, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    summary["observations_written"] += 1
                parsed.append(view)
            except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                exclusions[type(exc).__name__ + ":" + str(exc)] += 1
    summary["forecast_exclusions"] = dict(exclusions)
    summary["research_price_comparisons"] = parsed
    summary["indicative_paired_quotes"] = len(parsed)
    summary["book_price_capture_is_not_independent_execution"] = True
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=tuple(SPORT_IDS), required=True)
    parser.add_argument(
        "--forecast-root", type=Path,
        default=Path("history/walters_key_forward_v1/forecasts"),
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=Path("history/walters_key_forward_v1/indicative_quotes"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/walters_free_three_book_watch.json"),
    )
    args = parser.parse_args()
    key = os.environ.get("THERUNDOWN_API_KEY", "")
    report = scan(
        args.sport, args.forecast_root, args.evidence_root, key=key,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sport": args.sport,
        "mode": report["mode"],
        "pregame_forecasts": report["forecasts_with_immutable_pregame_publication"],
        "indicative_two_sided_quotes": len(report["research_price_comparisons"]),
        "newly_recorded": report["observations_written"],
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
