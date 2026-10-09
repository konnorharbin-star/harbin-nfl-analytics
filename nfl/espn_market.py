"""Free current NFL market retrieval from ESPN public endpoints.

This mirrors the NCAA live-market path: ESPN is the primary no-key source for the
current slate, while paid/multi-book providers remain optional enrichment. Sportsbook
prices are consumed only after the independent football projection already exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from .book_identity import canonical_book_identity
from .contracts import DataContractError, require_columns

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
ESPN_CORE_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/"
    "{event_id}/competitions/{event_id}/odds"
)

TEAM_ALIASES = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LA": "LA",
    "LAR": "LA",
    "LAC": "LAC",
    "LV": "LV",
    "LVR": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}


@dataclass(frozen=True)
class ESPNTwoWayMarket:
    game_id: str
    market_type: str
    provider: str
    book: str
    source_event_id: str
    captured_at: datetime
    first_side: str
    first_line: float | None
    first_american_odds: int
    second_side: str
    second_line: float | None
    second_american_odds: int
    # Strictly optional, source-origin book update time. Captured_at alone may
    # be collector retrieval time, NOT evidence of sportsbook quote freshness.
    source_quote_at: datetime | None = None
    source_quote_time_verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canon_team(value: object) -> str:
    key = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return TEAM_ALIASES.get(key, key)


def _number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if numeric != numeric else numeric


def _american(value: object, *, fallback: int | None = None) -> int | None:
    numeric = _number(value)
    if numeric is None:
        return fallback
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        return fallback
    if rounded == 0 or -100 < rounded < 100:
        return fallback
    return rounded


def _provider_name(odds: dict[str, object]) -> str:
    provider = odds.get("provider")
    if isinstance(provider, dict):
        return str(provider.get("name") or provider.get("id") or "ESPN")
    return str(provider or "ESPN")


def _moneyline(side: object) -> int | None:
    if not isinstance(side, dict):
        return None
    return _american(side.get("moneyLine", side.get("moneyline")))


def _spread_price(side: object) -> int:
    if not isinstance(side, dict):
        return -110
    return _american(
        side.get("spreadOdds", side.get("spread_odds")),
        fallback=-110,
    ) or -110


def _event_teams(event: dict[str, object]) -> tuple[str, str] | None:
    competitions = event.get("competitions")
    if not isinstance(competitions, list) or not competitions:
        return None
    competition = competitions[0]
    if not isinstance(competition, dict):
        return None
    home: str | None = None
    away: str | None = None
    competitors = competition.get("competitors")
    if not isinstance(competitors, list):
        return None
    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue
        team = competitor.get("team")
        if not isinstance(team, dict):
            continue
        abbreviation = _canon_team(team.get("abbreviation", ""))
        if competitor.get("homeAway") == "home":
            home = abbreviation
        elif competitor.get("homeAway") == "away":
            away = abbreviation
    if not home or not away:
        return None
    return home, away


def _parse_home_spread(
    odds: dict[str, object],
    *,
    home_team: str,
    away_team: str,
) -> float | None:
    details = str(odds.get("details") or "").strip()
    match = re.match(r"^(.*?)\s+([+-]?\d+(?:\.\d+)?)$", details)
    if match:
        favorite = _canon_team(match.group(1))
        favorite_line = _number(match.group(2))
        if favorite_line is not None:
            if favorite == home_team:
                return favorite_line
            if favorite == away_team:
                return -favorite_line

    spread = _number(odds.get("spread"))
    if spread is None:
        return None
    home_odds = odds.get("homeTeamOdds")
    away_odds = odds.get("awayTeamOdds")
    if isinstance(home_odds, dict) and home_odds.get("favorite") is True:
        return -abs(spread)
    if isinstance(away_odds, dict) and away_odds.get("favorite") is True:
        return abs(spread)
    return float(spread)


def parse_espn_odds(
    odds: dict[str, object],
    *,
    game_id: str,
    event_id: str,
    home_team: str,
    away_team: str,
    captured_at: datetime,
) -> list[ESPNTwoWayMarket]:
    """Normalize one ESPN odds object into complete two-way NFL markets."""

    provider = _provider_name(odds)
    home_side = odds.get("homeTeamOdds")
    away_side = odds.get("awayTeamOdds")
    output: list[ESPNTwoWayMarket] = []

    home_ml = _moneyline(home_side)
    away_ml = _moneyline(away_side)
    if home_ml is not None and away_ml is not None:
        output.append(
            ESPNTwoWayMarket(
                game_id=game_id,
                market_type="moneyline",
                provider="espn",
                book=provider,
                source_event_id=event_id,
                captured_at=captured_at,
                first_side="home",
                first_line=None,
                first_american_odds=home_ml,
                second_side="away",
                second_line=None,
                second_american_odds=away_ml,
            )
        )

    home_spread = _parse_home_spread(
        odds,
        home_team=home_team,
        away_team=away_team,
    )
    if home_spread is not None:
        output.append(
            ESPNTwoWayMarket(
                game_id=game_id,
                market_type="spread",
                provider="espn",
                book=provider,
                source_event_id=event_id,
                captured_at=captured_at,
                first_side="home",
                first_line=home_spread,
                first_american_odds=_spread_price(home_side),
                second_side="away",
                second_line=-home_spread,
                second_american_odds=_spread_price(away_side),
            )
        )

    total = _number(odds.get("overUnder", odds.get("overunder")))
    if total is not None:
        over_odds = _american(
            odds.get("overOdds", odds.get("over_odds")),
            fallback=-110,
        ) or -110
        under_odds = _american(
            odds.get("underOdds", odds.get("under_odds")),
            fallback=-110,
        ) or -110
        output.append(
            ESPNTwoWayMarket(
                game_id=game_id,
                market_type="total",
                provider="espn",
                book=provider,
                source_event_id=event_id,
                captured_at=captured_at,
                first_side="over",
                first_line=total,
                first_american_odds=over_odds,
                second_side="under",
                second_line=total,
                second_american_odds=under_odds,
            )
        )
    return output


class ESPNMarketClient:
    """No-key current NFL odds client using ESPN scoreboard/Core endpoints."""

    def __init__(self, *, timeout_seconds: float = 12.0) -> None:
        self.timeout_seconds = float(timeout_seconds)

    def _json(self, url: str, params: dict[str, object] | None = None) -> dict[str, object]:
        target = url if not params else f"{url}?{urlencode(params)}"
        request = Request(
            target,
            headers={"User-Agent": "harbin-nfl-analytics/0.1"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataContractError(f"ESPN NFL odds request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise DataContractError("ESPN NFL odds response is not a JSON object")
        return payload

    def scoreboard(self, *, week: int) -> dict[str, object]:
        return self._json(
            ESPN_SCOREBOARD_URL,
            {"limit": 100, "seasontype": 2, "week": int(week)},
        )

    def _core_odds(self, event_id: str) -> list[dict[str, object]]:
        payload = self._json(ESPN_CORE_ODDS_URL.format(event_id=event_id), {"limit": 50})
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        resolved: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            reference = item.get("$ref")
            if reference:
                try:
                    item = self._json(str(reference).replace("http://", "https://"))
                except DataContractError:
                    continue
            resolved.append(item)
        return resolved

    def current_markets(
        self,
        targets: pl.DataFrame,
        *,
        week: int,
    ) -> list[ESPNTwoWayMarket]:
        """Fetch ESPN markets for target nflverse games, with Core supplementation."""

        require_columns(targets, {"game_id", "home_team", "away_team"}, "live_targets")
        target_map = {
            (_canon_team(row["home_team"]), _canon_team(row["away_team"])): row
            for row in targets.iter_rows(named=True)
        }
        captured_at = datetime.now(UTC)
        payload = self.scoreboard(week=week)
        events = payload.get("events")
        if not isinstance(events, list):
            raise DataContractError("ESPN NFL scoreboard response is missing events")

        markets: list[ESPNTwoWayMarket] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            teams = _event_teams(event)
            if teams is None or teams not in target_map:
                continue
            target = target_map[teams]
            event_id = str(event.get("id") or "").strip()
            if not event_id:
                continue
            competitions = event.get("competitions")
            competition = (
                competitions[0] if isinstance(competitions, list) and competitions else {}
            )
            odds_list = competition.get("odds") if isinstance(competition, dict) else None
            normalized: list[ESPNTwoWayMarket] = []
            if isinstance(odds_list, list):
                for odds in odds_list:
                    if isinstance(odds, dict):
                        normalized.extend(
                            parse_espn_odds(
                                odds,
                                game_id=str(target["game_id"]),
                                event_id=event_id,
                                home_team=teams[0],
                                away_team=teams[1],
                                captured_at=captured_at,
                            )
                        )
            try:
                core_odds = self._core_odds(event_id)
            except DataContractError:
                core_odds = []
            for odds in core_odds:
                normalized.extend(
                    parse_espn_odds(
                        odds,
                        game_id=str(target["game_id"]),
                        event_id=event_id,
                        home_team=teams[0],
                        away_team=teams[1],
                        captured_at=captured_at,
                    )
                )
            if normalized:
                by_market_book: dict[
                    tuple[str, str],
                    ESPNTwoWayMarket,
                ] = {}
                for market in normalized:
                    book = canonical_book_identity(
                        market.book or market.provider
                    )
                    key = (market.market_type, book)
                    by_market_book.setdefault(key, market)
                markets.extend(by_market_book.values())

        if not markets:
            raise DataContractError("ESPN returned no usable NFL markets for the target slate")
        return sorted(
            markets,
            key=lambda item: (
                item.game_id,
                item.market_type,
                canonical_book_identity(item.book or item.provider),
            ),
        )
