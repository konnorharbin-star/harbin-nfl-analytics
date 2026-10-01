"""Professional/current NFL market aggregation with free ESPN as the primary source.

The NCAA model treats richer paid/professional market data as optional enrichment rather
than a dependency. This module gives the NFL model the same source hierarchy: ESPN is
always attempted first; The Odds API is used only when a key is configured. Every quote
stays downstream of the independent football projection.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from .contracts import DataContractError, require_columns
from .espn_market import ESPNMarketClient, ESPNTwoWayMarket
from .odds_api import NFL_SPORT_KEY, TEAM_NAME_TO_NFLVERSE

CURRENT_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def _american(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9 or rounded == 0 or -100 < rounded < 100:
        return None
    return rounded


def _quote_time(value: object, fallback: datetime) -> datetime:
    if value in {None, ""}:
        return fallback
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _market_pair(
    market: dict[str, Any],
    *,
    home_name: str,
    away_name: str,
) -> tuple[str, tuple[str, float | None, int], tuple[str, float | None, int]] | None:
    key = str(market.get("key") or "")
    outcomes = market.get("outcomes")
    if not isinstance(outcomes, list):
        return None
    by_name = {
        str(item.get("name") or ""): item
        for item in outcomes
        if isinstance(item, dict)
    }
    if key == "h2h":
        if home_name not in by_name or away_name not in by_name:
            return None
        home_price = _american(by_name[home_name].get("price"))
        away_price = _american(by_name[away_name].get("price"))
        if home_price is None or away_price is None:
            return None
        return "moneyline", ("home", None, home_price), ("away", None, away_price)
    if key == "spreads":
        if home_name not in by_name or away_name not in by_name:
            return None
        home = by_name[home_name]
        away = by_name[away_name]
        home_price = _american(home.get("price"))
        away_price = _american(away.get("price"))
        if (
            home.get("point") is None
            or away.get("point") is None
            or home_price is None
            or away_price is None
        ):
            return None
        home_line = float(home["point"])
        away_line = float(away["point"])
        if abs(home_line + away_line) > 1e-6:
            return None
        return (
            "spread",
            ("home", home_line, home_price),
            ("away", away_line, away_price),
        )
    if key == "totals":
        lower = {name.lower(): item for name, item in by_name.items()}
        if "over" not in lower or "under" not in lower:
            return None
        over = lower["over"]
        under = lower["under"]
        over_price = _american(over.get("price"))
        under_price = _american(under.get("price"))
        if (
            over.get("point") is None
            or under.get("point") is None
            or over_price is None
            or under_price is None
        ):
            return None
        over_line = float(over["point"])
        under_line = float(under["point"])
        if abs(over_line - under_line) > 1e-6:
            return None
        return (
            "total",
            ("over", over_line, over_price),
            ("under", under_line, under_price),
        )
    return None


class CurrentOddsAPIClient:
    """Optional no-state current-market client for The Odds API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 20.0,
        fetch_json: Callable[[str], object] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        self._fetch_json = fetch_json or self._http_json

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _http_json(self, url: str) -> object:
        request = Request(url, headers={"User-Agent": "harbin-nfl-analytics/0.1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataContractError(f"The Odds API current request failed: {exc}") from exc

    def current_markets(self, targets: pl.DataFrame) -> list[ESPNTwoWayMarket]:
        if not self.api_key:
            return []
        require_columns(targets, {"game_id", "home_team", "away_team"}, "live_targets")
        target_map = {
            (str(row["home_team"]), str(row["away_team"])): str(row["game_id"])
            for row in targets.iter_rows(named=True)
        }
        url = CURRENT_ODDS_API_URL.format(sport=NFL_SPORT_KEY) + "?" + urlencode(
            {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
        )
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            raise DataContractError("The Odds API current response is not a list")

        fetched_at = datetime.now(UTC)
        rows: list[ESPNTwoWayMarket] = []
        for event in payload:
            if not isinstance(event, dict):
                continue
            home_name = str(event.get("home_team") or "")
            away_name = str(event.get("away_team") or "")
            home = TEAM_NAME_TO_NFLVERSE.get(home_name)
            away = TEAM_NAME_TO_NFLVERSE.get(away_name)
            if home is None or away is None:
                continue
            game_id = target_map.get((home, away))
            if game_id is None:
                continue
            source_event = str(event.get("id") or "")
            bookmakers = event.get("bookmakers")
            if not isinstance(bookmakers, list):
                continue
            for bookmaker in bookmakers:
                if not isinstance(bookmaker, dict):
                    continue
                book = str(
                    bookmaker.get("title") or bookmaker.get("key") or "the_odds_api"
                )
                captured_at = _quote_time(bookmaker.get("last_update"), fetched_at)
                markets = bookmaker.get("markets")
                if not isinstance(markets, list):
                    continue
                for market in markets:
                    if not isinstance(market, dict):
                        continue
                    pair = _market_pair(
                        market,
                        home_name=home_name,
                        away_name=away_name,
                    )
                    if pair is None:
                        continue
                    market_type, first, second = pair
                    rows.append(
                        ESPNTwoWayMarket(
                            game_id=game_id,
                            market_type=market_type,
                            provider="the_odds_api",
                            book=book,
                            source_event_id=source_event,
                            captured_at=captured_at,
                            first_side=first[0],
                            first_line=first[1],
                            first_american_odds=first[2],
                            second_side=second[0],
                            second_line=second[1],
                            second_american_odds=second[2],
                        )
                    )
        return rows


def collect_current_markets(
    targets: pl.DataFrame,
    *,
    week: int,
    espn_client: ESPNMarketClient | None = None,
    pro_client: CurrentOddsAPIClient | None = None,
) -> tuple[list[ESPNTwoWayMarket], dict[str, object]]:
    """Return deduplicated current markets and source/breadth metadata."""

    espn = espn_client or ESPNMarketClient()
    optional = pro_client or CurrentOddsAPIClient()
    source_errors: list[str] = []
    markets: list[ESPNTwoWayMarket] = []
    try:
        markets.extend(espn.current_markets(targets, week=week))
    except DataContractError as exc:
        source_errors.append(f"espn: {exc}")

    pro_rows: list[ESPNTwoWayMarket] = []
    if optional.configured:
        try:
            pro_rows = optional.current_markets(targets)
            markets.extend(pro_rows)
        except DataContractError as exc:
            source_errors.append(f"the_odds_api: {exc}")

    unique: dict[tuple[object, ...], ESPNTwoWayMarket] = {}
    for market in markets:
        key = (
            market.game_id,
            market.market_type,
            market.provider,
            market.book,
            market.first_side,
            market.first_line,
            market.first_american_odds,
            market.second_side,
            market.second_line,
            market.second_american_odds,
        )
        previous = unique.get(key)
        if previous is None or market.captured_at > previous.captured_at:
            unique[key] = market
    values = sorted(
        unique.values(),
        key=lambda item: (item.game_id, item.market_type, item.book, item.provider),
    )
    if not values:
        raise DataContractError(
            "no usable current NFL markets from ESPN or optional professional source"
        )

    games = targets.height
    book_counts: dict[str, set[str]] = {}
    for row in values:
        book_counts.setdefault(row.game_id, set()).add(str(row.book).strip().lower())
    multi_book_games = sum(len(books) >= 2 for books in book_counts.values())
    return values, {
        "primary_source": "ESPN public endpoints",
        "optional_source_configured": optional.configured,
        "optional_source_rows": len(pro_rows),
        "sources": sorted({row.provider for row in values}),
        "books": len({str(row.book).strip().lower() for row in values}),
        "games": games,
        "games_with_any_market": len(book_counts),
        "multi_book_games": multi_book_games,
        "multi_book_coverage": multi_book_games / games if games else 0.0,
        "source_errors": source_errors,
    }
