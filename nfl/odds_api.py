"""Historical NFL odds adapter for The Odds API.

The provider is optional and strictly downstream of the independent football model.
Historical snapshots are fetched only for explicit UTC decision timestamps, cached
without the API key, mapped to nflverse game ids, and normalized into the platform's
point-in-time market-history contract.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from .contracts import DataContractError, require_columns
from .market_history import MARKET_HISTORY_REQUIRED, validate_market_history

THE_ODDS_API_PROVIDER = "the_odds_api"
NFL_SPORT_KEY = "americanfootball_nfl"
FEATURED_MARKETS = ("h2h", "spreads", "totals")

TEAM_NAME_TO_NFLVERSE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def _parse_iso_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DataContractError(f"provider timestamp is not timezone-aware: {value}")
    return parsed.astimezone(UTC)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataContractError("historical odds timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return _require_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _game_date(value: object) -> date:
    text = str(value)
    return date.fromisoformat(text[:10])


def _american_price(value: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"invalid American price from provider: {value!r}") from exc
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        raise DataContractError(f"non-integral American price from provider: {numeric}")
    if rounded == 0 or -100 < rounded < 100:
        raise DataContractError(f"invalid American price from provider: {rounded}")
    return rounded


def _normalize_markets(markets: Iterable[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(value) for value in markets))
    if not values:
        raise ValueError("at least one odds market is required")
    unsupported = sorted(set(values).difference(FEATURED_MARKETS))
    if unsupported:
        raise ValueError(f"unsupported featured market(s): {', '.join(unsupported)}")
    return values


class TheOddsAPIClient:
    """Small cached client for the provider's historical featured-market endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path = "data/odds_api_cache",
        base_url: str = "https://api.the-odds-api.com/v4",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def _cache_path(self, endpoint: str, params_without_key: dict[str, str]) -> Path:
        stable = endpoint + "?" + urlencode(sorted(params_without_key.items()))
        digest = sha256(stable.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"historical_{digest}.json"

    def historical_snapshot(
        self,
        at: datetime,
        *,
        sport: str = NFL_SPORT_KEY,
        regions: tuple[str, ...] = ("us",),
        markets: tuple[str, ...] = FEATURED_MARKETS,
        bookmakers: tuple[str, ...] | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch the closest provider snapshot equal to or earlier than ``at``."""

        if not self.api_key:
            raise DataContractError(
                "THE_ODDS_API_KEY is required for historical odds retrieval; "
                "historical provider data is not synthesized"
            )
        normalized_markets = _normalize_markets(markets)
        endpoint = f"/historical/sports/{sport}/odds"
        params: dict[str, str] = {
            "markets": ",".join(normalized_markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date": _iso_z(at),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(dict.fromkeys(bookmakers))
        else:
            if not regions:
                raise ValueError("regions cannot be empty when bookmakers are not specified")
            params["regions"] = ",".join(dict.fromkeys(regions))

        cache_path = self._cache_path(endpoint, params)
        if cache_path.exists() and not refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            query = {**params, "apiKey": self.api_key}
            url = f"{self.base_url}{endpoint}?{urlencode(query)}"
            request = Request(url, headers={"User-Agent": "harbin-nfl-analytics/0.1"})
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                raise DataContractError(
                    f"The Odds API historical request failed with HTTP {exc.code}: {body}"
                ) from exc
            except (URLError, TimeoutError) as exc:
                raise DataContractError(f"The Odds API historical request failed: {exc}") from exc
            cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise DataContractError("The Odds API historical response has an unexpected schema")
        if not isinstance(payload.get("timestamp"), str):
            raise DataContractError("The Odds API historical response is missing timestamp")
        return payload


def _match_schedule_game(
    schedules: pl.DataFrame,
    *,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> dict[str, object] | None:
    candidates = schedules.filter(
        (pl.col("home_team") == home_team) & (pl.col("away_team") == away_team)
    )
    if candidates.is_empty():
        return None

    event_date = commence_time.date()
    ranked: list[tuple[int, str, dict[str, object]]] = []
    for row in candidates.iter_rows(named=True):
        day_distance = abs((_game_date(row["gameday"]) - event_date).days)
        if day_distance <= 1:
            ranked.append((day_distance, str(row["game_id"]), row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise DataContractError(
            f"ambiguous schedule match for {away_team} at {home_team} on {event_date}"
        )
    return ranked[0][2]


def _market_pair(
    market: dict[str, Any],
    *,
    event_home_name: str,
    event_away_name: str,
) -> tuple[str, list[tuple[str, float | None, int]]] | None:
    key = str(market.get("key", ""))
    outcomes = market.get("outcomes")
    if key not in FEATURED_MARKETS or not isinstance(outcomes, list):
        return None

    if key == "h2h":
        by_name = {str(item.get("name")): item for item in outcomes if isinstance(item, dict)}
        if event_home_name not in by_name or event_away_name not in by_name:
            return None
        return (
            "moneyline",
            [
                ("home", None, _american_price(by_name[event_home_name].get("price"))),
                ("away", None, _american_price(by_name[event_away_name].get("price"))),
            ],
        )

    if key == "spreads":
        by_name = {str(item.get("name")): item for item in outcomes if isinstance(item, dict)}
        if event_home_name not in by_name or event_away_name not in by_name:
            return None
        home = by_name[event_home_name]
        away = by_name[event_away_name]
        if home.get("point") is None or away.get("point") is None:
            return None
        return (
            "spread",
            [
                ("home", float(home["point"]), _american_price(home.get("price"))),
                ("away", float(away["point"]), _american_price(away.get("price"))),
            ],
        )

    by_name = {
        str(item.get("name", "")).strip().lower(): item
        for item in outcomes
        if isinstance(item, dict)
    }
    if "over" not in by_name or "under" not in by_name:
        return None
    over = by_name["over"]
    under = by_name["under"]
    if over.get("point") is None or under.get("point") is None:
        return None
    return (
        "total",
        [
            ("over", float(over["point"]), _american_price(over.get("price"))),
            ("under", float(under["point"]), _american_price(under.get("price"))),
        ],
    )


def historical_snapshot_to_market_history(
    payload: dict[str, Any],
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize one provider snapshot into reproducible two-way market rows."""

    require_columns(
        schedules,
        {"game_id", "gameday", "home_team", "away_team"},
        "odds_schedule_mapping",
    )
    captured_at = _parse_iso_utc(str(payload.get("timestamp")))
    snapshot_id = _iso_z(captured_at)
    events = payload.get("data")
    if not isinstance(events, list):
        raise DataContractError("The Odds API historical response data must be a list")

    rows: list[dict[str, object]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        source_event_id = str(event.get("id", "")).strip()
        event_home_name = str(event.get("home_team", "")).strip()
        event_away_name = str(event.get("away_team", "")).strip()
        commence_raw = event.get("commence_time")
        if not source_event_id or not event_home_name or not event_away_name or not commence_raw:
            continue
        home_team = TEAM_NAME_TO_NFLVERSE.get(event_home_name)
        away_team = TEAM_NAME_TO_NFLVERSE.get(event_away_name)
        if home_team is None or away_team is None:
            continue
        commence_time = _parse_iso_utc(str(commence_raw))
        game = _match_schedule_game(
            schedules,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
        )
        if game is None:
            continue

        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            book = str(bookmaker.get("key", "")).strip()
            if not book:
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue
            for market in markets:
                if not isinstance(market, dict):
                    continue
                pair = _market_pair(
                    market,
                    event_home_name=event_home_name,
                    event_away_name=event_away_name,
                )
                if pair is None:
                    continue
                market_type, quotes = pair
                source_last_update = market.get("last_update") or bookmaker.get("last_update")
                source_last_update_dt = (
                    _parse_iso_utc(str(source_last_update)) if source_last_update else None
                )
                for side, line, price in quotes:
                    row: dict[str, object] = {
                        "game_id": str(game["game_id"]),
                        "market_type": market_type,
                        "side": side,
                        "line": line,
                        "american_odds": price,
                        "provider": THE_ODDS_API_PROVIDER,
                        "book": book,
                        "captured_at": captured_at,
                        "snapshot_id": snapshot_id,
                        "source_event_id": source_event_id,
                        "source_commence_time": commence_time,
                        "source_last_update": source_last_update_dt,
                    }
                    if "season" in game:
                        row["season"] = game["season"]
                    if "week" in game:
                        row["week"] = game["week"]
                    rows.append(row)

    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    validate_market_history(frame.select(sorted(MARKET_HISTORY_REQUIRED)))
    return frame.sort(
        ["game_id", "provider", "book", "market_type", "snapshot_id", "side"]
    )


def fetch_market_history_for_decisions(
    client: TheOddsAPIClient,
    schedules: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    regions: tuple[str, ...] = ("us",),
    markets: tuple[str, ...] = FEATURED_MARKETS,
    bookmakers: tuple[str, ...] | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Fetch one historical snapshot per unique decision timestamp and retain target games."""

    require_columns(decisions, {"game_id", "decision_time"}, "odds_decisions")
    if decisions.is_empty():
        raise DataContractError("odds_decisions is empty")

    grouped: dict[datetime, set[str]] = {}
    for row in decisions.iter_rows(named=True):
        at = row["decision_time"]
        if not isinstance(at, datetime):
            raise DataContractError("decision_time values must be Python datetimes")
        normalized = _require_utc(at)
        grouped.setdefault(normalized, set()).add(str(row["game_id"]))

    frames: list[pl.DataFrame] = []
    for at in sorted(grouped):
        payload = client.historical_snapshot(
            at,
            regions=regions,
            markets=markets,
            bookmakers=bookmakers,
            refresh=refresh,
        )
        parsed = historical_snapshot_to_market_history(payload, schedules)
        if parsed.is_empty():
            continue
        selected = parsed.filter(pl.col("game_id").is_in(sorted(grouped[at])))
        if not selected.is_empty():
            frames.append(selected)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["game_id", "provider", "book", "market_type", "snapshot_id", "side"],
        keep="last",
    ).sort(["game_id", "captured_at", "book", "market_type", "side"])


def closing_decisions_from_history(
    decision_history: pl.DataFrame,
    *,
    minutes_before_kickoff: int = 5,
) -> pl.DataFrame:
    """Create one pre-kickoff closing query timestamp per game from provider commence time."""

    if minutes_before_kickoff < 0:
        raise ValueError("minutes_before_kickoff must be >= 0")
    require_columns(
        decision_history,
        {"game_id", "source_commence_time"},
        "decision_market_history",
    )
    if decision_history.is_empty():
        raise DataContractError("decision_market_history is empty")
    games = decision_history.select("game_id", "source_commence_time").unique()
    if games.group_by("game_id").len().filter(pl.col("len") != 1).height:
        raise DataContractError("game_id maps to multiple provider commence times")
    return games.with_columns(
        (pl.col("source_commence_time") - pl.duration(minutes=minutes_before_kickoff)).alias(
            "decision_time"
        )
    ).select("game_id", "decision_time")
