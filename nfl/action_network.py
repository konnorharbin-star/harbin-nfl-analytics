"""Free multi-book NFL market enrichment from Action Network's public scoreboard.

This mirrors the NCAA market-intelligence source hierarchy. Action Network is treated
as optional public enrichment behind ESPN and remains downstream of the independent
football projection. Malformed, ambiguous, stale-without-timestamp, or incomplete
two-way rows fail closed instead of being synthesized.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from .contracts import DataContractError, require_columns
from .espn_market import ESPNTwoWayMarket, _canon_team
from .odds_api import TEAM_NAME_TO_NFLVERSE

ACTION_NETWORK_URLS = (
    "https://api.actionnetwork.com/web/v2/scoreboard/nfl",
    "https://api.actionnetwork.com/web/v1/scoreboard/nfl",
)

ACTION_NETWORK_BOOKS = {
    "15": "DraftKings",
    "30": "FanDuel",
    "75": "BetMGM",
    "123": "Caesars",
    "68": "BetRivers",
    "247": "BetUS",
    "972": "Unibet",
    "71": "WynnBET",
    "79": "FoxBet",
    "69": "PointsBet",
}

KNOWN_TEAM_CODES = set(TEAM_NAME_TO_NFLVERSE.values())


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _american(value: object) -> int | None:
    number = _number(value)
    if number is None:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9:
        return None
    if rounded == 0 or -100 < rounded < 100:
        return None
    return rounded


def _timestamp(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _team_codes(team: dict[str, object]) -> set[str]:
    output: set[str] = set()
    for key in (
        "display_name",
        "full_name",
        "short_name",
        "name",
        "location",
        "abbr",
        "abbreviation",
    ):
        raw = team.get(key)
        if raw in {None, ""}:
            continue
        text = str(raw).strip()
        mapped = TEAM_NAME_TO_NFLVERSE.get(text)
        if mapped:
            output.add(mapped)
        compact = _canon_team(text)
        if compact in KNOWN_TEAM_CODES:
            output.add(compact)
    return output


def _game_teams(game: dict[str, object]) -> tuple[set[str], set[str]]:
    teams = game.get("teams")
    if not isinstance(teams, list):
        return set(), set()
    home_id = str(game.get("home_team_id") or "")
    away_id = str(game.get("away_team_id") or "")
    home = next(
        (
            team
            for team in teams
            if isinstance(team, dict) and str(team.get("id") or "") == home_id
        ),
        {},
    )
    away = next(
        (
            team
            for team in teams
            if isinstance(team, dict) and str(team.get("id") or "") == away_id
        ),
        {},
    )
    return _team_codes(home), _team_codes(away)


def _book_name(book_id: object) -> str:
    key = str(book_id or "").strip()
    return ACTION_NETWORK_BOOKS.get(key, f"ActionNetwork book {key or 'unknown'}")


def _row_timestamp(
    row: dict[str, object],
    market: dict[str, object],
    event: dict[str, object],
) -> datetime | None:
    for value in (
        row.get("last_update"),
        row.get("updated_at"),
        row.get("timestamp"),
        market.get("last_update"),
        market.get("updated_at"),
        event.get("last_update"),
        event.get("updated_at"),
    ):
        parsed = _timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _offer_origin_time(row: dict[str, object]) -> datetime | None:
    # Only timestamps tied to the actual sportsbook-side quote are proof.
    # Event/market update times and request capture times do not qualify.
    for name in ("last_update", "updated_at", "timestamp"):
        stamp = _timestamp(row.get(name))
        if stamp is not None:
            return stamp
    return None


def _paired_origin_time(
    quote: dict[str, object], kind: str, first: str, second: str,
) -> datetime | None:
    stamps = quote.get("origin_stamps")
    if not isinstance(stamps, dict):
        return None
    a, b = stamps.get(f"{kind}:{first}"), stamps.get(f"{kind}:{second}")
    if not isinstance(a, datetime) or not isinstance(b, datetime):
        return None
    from datetime import timedelta
    if abs(a - b) > timedelta(minutes=2):
        return None
    return max(a, b)


def parse_action_network_game(
    game: dict[str, object],
    target_map: dict[tuple[str, str], str],
    *,
    observed_at: datetime | None = None,
) -> list[ESPNTwoWayMarket]:
    """Normalize one Action Network game into complete per-book NFL markets."""

    home_codes, away_codes = _game_teams(game)
    matches = [
        (home, away, game_id)
        for (home, away), game_id in target_map.items()
        if home in home_codes and away in away_codes
    ]
    if len(matches) != 1:
        return []
    home_code, away_code, game_id = matches[0]

    markets = game.get("markets")
    if isinstance(markets, list):
        markets = {
            str(index): value
            for index, value in enumerate(markets)
            if isinstance(value, dict)
        }
    if not isinstance(markets, dict):
        return []

    grouped: dict[str, dict[str, object]] = {}
    for market_id, market in markets.items():
        if not isinstance(market, dict):
            continue
        event = market.get("event")
        if not isinstance(event, dict):
            continue
        for kind in ("moneyline", "spread", "total"):
            rows = event.get(kind)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                book_id = str(
                    row.get("book_id")
                    if row.get("book_id") is not None
                    else market_id
                )
                book = grouped.setdefault(
                    book_id,
                    {
                        "book": _book_name(book_id),
                        "timestamps": [],
                        "origin_stamps": {},
                        "home_ml": None,
                        "away_ml": None,
                        "home_spread": None,
                        "away_spread": None,
                        "home_spread_odds": None,
                        "away_spread_odds": None,
                        "total_over": None,
                        "total_under": None,
                        "over_odds": None,
                        "under_odds": None,
                    },
                )
                stamp = _row_timestamp(row, market, event)
                if stamp is not None:
                    timestamps = book["timestamps"]
                    if isinstance(timestamps, list):
                        timestamps.append(stamp)

                side = str(row.get("side") or "").lower()
                value = _number(row.get("value"))
                odds = _american(row.get("odds"))
                origin = _offer_origin_time(row)
                if odds is not None and side in (
                    ("home", "away") if kind != "total" else ("over", "under")
                ):
                    origin_stamps = book["origin_stamps"]
                    if isinstance(origin_stamps, dict):
                        # No prior source stamp survives a later undated side
                        # update. Otherwise an updated price could inherit an
                        # old, unrelated timestamp and create fake arbitrage.
                        origin_stamps[f"{kind}:{side}"] = origin
                if kind == "moneyline":
                    if side == "home":
                        book["home_ml"] = odds
                    elif side == "away":
                        book["away_ml"] = odds
                elif kind == "spread":
                    if side == "home":
                        book["home_spread"] = value
                        book["home_spread_odds"] = odds
                    elif side == "away":
                        book["away_spread"] = value
                        book["away_spread_odds"] = odds
                elif kind == "total":
                    if side == "over":
                        book["total_over"] = value
                        book["over_odds"] = odds
                    elif side == "under":
                        book["total_under"] = value
                        book["under_odds"] = odds

    source_event_id = str(game.get("id") or game.get("game_id") or "")
    output: list[ESPNTwoWayMarket] = []
    for quote in grouped.values():
        timestamps = quote.get("timestamps")
        if isinstance(timestamps, list) and timestamps:
            captured_at = max(timestamps)
        elif observed_at is not None:
            if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                raise DataContractError(
                    "Action Network observed_at must be timezone-aware"
                )
            captured_at = observed_at.astimezone(UTC)
        else:
            continue
        book_name = str(quote["book"])

        home_ml = quote.get("home_ml")
        away_ml = quote.get("away_ml")
        if isinstance(home_ml, int) and isinstance(away_ml, int):
            output.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="moneyline",
                    provider="action_network",
                    book=book_name,
                    source_event_id=source_event_id,
                    captured_at=captured_at,
                    source_quote_at=_paired_origin_time(quote, "moneyline", "home", "away"),
                    source_quote_time_verified=(
                        _paired_origin_time(quote, "moneyline", "home", "away")
                        is not None
                    ),
                    first_side="home",
                    first_line=None,
                    first_american_odds=home_ml,
                    second_side="away",
                    second_line=None,
                    second_american_odds=away_ml,
                )
            )

        home_spread = quote.get("home_spread")
        away_spread = quote.get("away_spread")
        home_spread_odds = quote.get("home_spread_odds")
        away_spread_odds = quote.get("away_spread_odds")
        if home_spread is None and away_spread is not None:
            home_spread = -float(away_spread)
        if away_spread is None and home_spread is not None:
            away_spread = -float(home_spread)
        if (
            home_spread is not None
            and away_spread is not None
            and isinstance(home_spread_odds, int)
            and isinstance(away_spread_odds, int)
            and abs(float(home_spread) + float(away_spread)) <= 1e-6
        ):
            output.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="spread",
                    provider="action_network",
                    book=book_name,
                    source_event_id=source_event_id,
                    captured_at=captured_at,
                    source_quote_at=_paired_origin_time(quote, "spread", "home", "away"),
                    source_quote_time_verified=(
                        _paired_origin_time(quote, "spread", "home", "away")
                        is not None
                    ),
                    first_side="home",
                    first_line=float(home_spread),
                    first_american_odds=home_spread_odds,
                    second_side="away",
                    second_line=float(away_spread),
                    second_american_odds=away_spread_odds,
                )
            )

        total_over = quote.get("total_over")
        total_under = quote.get("total_under")
        over_odds = quote.get("over_odds")
        under_odds = quote.get("under_odds")
        if (
            total_over is not None
            and total_under is not None
            and isinstance(over_odds, int)
            and isinstance(under_odds, int)
            and abs(float(total_over) - float(total_under)) <= 1e-6
        ):
            output.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="total",
                    provider="action_network",
                    book=book_name,
                    source_event_id=source_event_id,
                    captured_at=captured_at,
                    source_quote_at=_paired_origin_time(quote, "total", "over", "under"),
                    source_quote_time_verified=(
                        _paired_origin_time(quote, "total", "over", "under")
                        is not None
                    ),
                    first_side="over",
                    first_line=float(total_over),
                    first_american_odds=over_odds,
                    second_side="under",
                    second_line=float(total_under),
                    second_american_odds=under_odds,
                )
            )

    return sorted(
        output,
        key=lambda row: (row.game_id, row.market_type, row.book),
    )


class ActionNetworkNFLClient:
    """No-key current NFL market client for Action Network's public scoreboard."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        fetch_json: Callable[[str], object] | None = None,
        enabled: bool | None = None,
    ) -> None:
        env_enabled = (
            os.environ.get("HARBIN_ACTION_NETWORK", "1").strip().lower()
            not in {"0", "false", "off", "no"}
        )
        self.enabled = env_enabled if enabled is None else bool(enabled)
        self.timeout_seconds = float(timeout_seconds)
        self._fetch_json = fetch_json or self._http_json
        self.last_diagnostic: dict[str, object] = {
            "status": "NOT_RUN",
            "endpoint": None,
            "payload_games": 0,
            "team_matches": 0,
            "parsed_rows": 0,
        }

    def _http_json(self, url: str) -> object:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HarbinNFLAnalytics/0.1)",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataContractError(
                f"Action Network public scoreboard request failed: {exc}"
            ) from exc

    def current_markets(
        self,
        targets: pl.DataFrame,
        *,
        week: int,
    ) -> list[ESPNTwoWayMarket]:
        if not self.enabled:
            return []
        require_columns(
            targets,
            {"season", "game_id", "home_team", "away_team"},
            "action_network_targets",
        )
        if targets.is_empty():
            return []
        seasons = {
            int(value)
            for value in targets.get_column("season").drop_nulls().to_list()
        }
        if len(seasons) != 1:
            raise DataContractError(
                "Action Network targets must belong to exactly one season"
            )
        season = next(iter(seasons))
        target_map = {
            (str(row["home_team"]), str(row["away_team"])): str(row["game_id"])
            for row in targets.iter_rows(named=True)
        }
        params = urlencode(
            {
                "season": season,
                "week": int(week),
                "seasonType": "reg",
                "periods": "event",
            }
        )

        errors: list[str] = []
        diagnostics: list[dict[str, object]] = []
        for base_url in ACTION_NETWORK_URLS:
            url = f"{base_url}?{params}"
            try:
                payload = self._fetch_json(url)
            except DataContractError as exc:
                errors.append(str(exc))
                diagnostics.append(
                    {
                        "endpoint": base_url,
                        "status": "ERROR",
                        "error": str(exc),
                    }
                )
                continue
            if not isinstance(payload, dict):
                errors.append("Action Network response is not an object")
                diagnostics.append(
                    {
                        "endpoint": base_url,
                        "status": "INVALID",
                        "payload_type": type(payload).__name__,
                    }
                )
                continue
            games = payload.get("games")
            if not isinstance(games, list):
                errors.append("Action Network response is missing games")
                diagnostics.append(
                    {
                        "endpoint": base_url,
                        "status": "INVALID",
                        "payload_keys": sorted(str(key) for key in payload)[:30],
                    }
                )
                continue

            observed_at = datetime.now(UTC)
            rows: list[ESPNTwoWayMarket] = []
            team_matches = 0
            games_with_markets = 0
            sample: dict[str, object] = {}
            for game in games:
                if not isinstance(game, dict):
                    continue
                home_codes, away_codes = _game_teams(game)
                if any(
                    home in home_codes and away in away_codes
                    for home, away in target_map
                ):
                    team_matches += 1
                markets = game.get("markets")
                if isinstance(markets, (dict, list)) and markets:
                    games_with_markets += 1
                if not sample:
                    sample = {
                        "game_keys": sorted(str(key) for key in game)[:40],
                        "home_codes": sorted(home_codes),
                        "away_codes": sorted(away_codes),
                        "home_team_id": game.get("home_team_id"),
                        "away_team_id": game.get("away_team_id"),
                    }
                    if isinstance(markets, dict):
                        sample["market_keys"] = sorted(
                            str(key) for key in markets
                        )[:20]
                        first_market = next(
                            (
                                value
                                for value in markets.values()
                                if isinstance(value, dict)
                            ),
                            None,
                        )
                        if isinstance(first_market, dict):
                            sample["first_market_keys"] = sorted(
                                str(key) for key in first_market
                            )[:30]
                            sample["market_timestamp_fields"] = {
                                key: first_market.get(key)
                                for key in (
                                    "last_update",
                                    "updated_at",
                                    "timestamp",
                                )
                                if first_market.get(key) not in {None, ""}
                            }
                            event = first_market.get("event")
                            if isinstance(event, dict):
                                sample["event_keys"] = sorted(
                                    str(key) for key in event
                                )[:30]
                                sample["event_timestamp_fields"] = {
                                    key: event.get(key)
                                    for key in (
                                        "last_update",
                                        "updated_at",
                                        "timestamp",
                                    )
                                    if event.get(key) not in {None, ""}
                                }
                                row_samples: dict[str, object] = {}
                                for kind in ("moneyline", "spread", "total"):
                                    values = event.get(kind)
                                    first_row = (
                                        next(
                                            (
                                                row
                                                for row in values
                                                if isinstance(row, dict)
                                            ),
                                            None,
                                        )
                                        if isinstance(values, list)
                                        else None
                                    )
                                    if isinstance(first_row, dict):
                                        row_samples[kind] = {
                                            "keys": sorted(
                                                str(key) for key in first_row
                                            )[:30],
                                            "book_id": first_row.get("book_id"),
                                            "side": first_row.get("side"),
                                            "value": first_row.get("value"),
                                            "odds": first_row.get("odds"),
                                            "last_update": first_row.get(
                                                "last_update"
                                            ),
                                            "updated_at": first_row.get(
                                                "updated_at"
                                            ),
                                            "timestamp": first_row.get(
                                                "timestamp"
                                            ),
                                        }
                                sample["row_samples"] = row_samples
                    elif isinstance(markets, list):
                        sample["market_list_length"] = len(markets)
                        first_market = next(
                            (
                                value
                                for value in markets
                                if isinstance(value, dict)
                            ),
                            None,
                        )
                        if isinstance(first_market, dict):
                            sample["first_market_keys"] = sorted(
                                str(key) for key in first_market
                            )[:30]

                rows.extend(
                    parse_action_network_game(
                        game,
                        target_map,
                        observed_at=observed_at,
                    )
                )

            diagnostic = {
                "endpoint": base_url,
                "status": "READY" if rows else "NO_ROWS",
                "payload_games": len(games),
                "team_matches": team_matches,
                "games_with_markets": games_with_markets,
                "parsed_rows": len(rows),
                "collector_observed_at": observed_at.isoformat(),
                "provider_timestamp_available": bool(
                    sample.get("market_timestamp_fields")
                    or sample.get("event_timestamp_fields")
                    or any(
                        isinstance(value, dict)
                        and any(
                            value.get(field) not in {None, ""}
                            for field in (
                                "last_update",
                                "updated_at",
                                "timestamp",
                            )
                        )
                        for value in (
                            sample.get("row_samples", {})
                            if isinstance(sample.get("row_samples"), dict)
                            else {}
                        ).values()
                    )
                ),
                "sample": sample,
            }
            diagnostics.append(diagnostic)
            self.last_diagnostic = {
                **diagnostic,
                "attempts": diagnostics,
            }
            if rows:
                return rows

        self.last_diagnostic = {
            "status": "ERROR" if errors else "NO_ROWS",
            "payload_games": 0,
            "team_matches": 0,
            "parsed_rows": 0,
            "attempts": diagnostics,
        }
        if errors and all(
            item.get("status") in {"ERROR", "INVALID"}
            for item in diagnostics
        ):
            raise DataContractError("; ".join(errors[-2:]))
        return []
