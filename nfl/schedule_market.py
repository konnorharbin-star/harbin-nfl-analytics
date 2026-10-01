"""Research-only current market snapshots from nflverse schedule fields.

The nflverse schedule feed exposes spread, total, moneyline and price fields for
upcoming games, but it does not expose a sportsbook identity or a quote-update
timestamp in the schedule schema. These rows can support current model-vs-market
research when the primary live source is unavailable, but they are not executable
quotes and must not satisfy verified-book, line-history, CLV, or production gates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import polars as pl

from .contracts import require_columns
from .espn_market import ESPNTwoWayMarket

NFLVERSE_SCHEDULE_PROVIDER = "nflverse_schedule_snapshot"
NFLVERSE_SCHEDULE_BOOK = "unattributed_nflverse_schedule"
SCHEDULE_CURRENT_MARKET_REQUIRED = {
    "game_id",
    "home_team",
    "away_team",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
}


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _american(value: object) -> int | None:
    number = _number(value)
    if number is None:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9 or rounded == 0 or -100 < rounded < 100:
        return None
    return rounded


def schedule_snapshot_markets(
    targets: pl.DataFrame,
    *,
    captured_at: datetime | None = None,
) -> list[ESPNTwoWayMarket]:
    """Normalize nflverse upcoming schedule prices as non-executable research rows."""

    require_columns(targets, SCHEDULE_CURRENT_MARKET_REQUIRED, "schedule_market_targets")
    stamp = captured_at or datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    stamp = stamp.astimezone(UTC)

    markets: list[ESPNTwoWayMarket] = []
    for row in targets.iter_rows(named=True):
        game_id = str(row["game_id"])

        home_ml = _american(row.get("home_moneyline"))
        away_ml = _american(row.get("away_moneyline"))
        if home_ml is not None and away_ml is not None:
            markets.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="moneyline",
                    provider=NFLVERSE_SCHEDULE_PROVIDER,
                    book=NFLVERSE_SCHEDULE_BOOK,
                    source_event_id=game_id,
                    captured_at=stamp,
                    first_side="home",
                    first_line=None,
                    first_american_odds=home_ml,
                    second_side="away",
                    second_line=None,
                    second_american_odds=away_ml,
                )
            )

        # nflverse spread_line is positive when home is favored. A bettable home
        # handicap therefore has the opposite sign.
        spread = _number(row.get("spread_line"))
        home_spread_price = _american(row.get("home_spread_odds"))
        away_spread_price = _american(row.get("away_spread_odds"))
        if (
            spread is not None
            and home_spread_price is not None
            and away_spread_price is not None
        ):
            home_line = -spread
            markets.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="spread",
                    provider=NFLVERSE_SCHEDULE_PROVIDER,
                    book=NFLVERSE_SCHEDULE_BOOK,
                    source_event_id=game_id,
                    captured_at=stamp,
                    first_side="home",
                    first_line=home_line,
                    first_american_odds=home_spread_price,
                    second_side="away",
                    second_line=-home_line,
                    second_american_odds=away_spread_price,
                )
            )

        total = _number(row.get("total_line"))
        over_price = _american(row.get("over_odds"))
        under_price = _american(row.get("under_odds"))
        if total is not None and over_price is not None and under_price is not None:
            markets.append(
                ESPNTwoWayMarket(
                    game_id=game_id,
                    market_type="total",
                    provider=NFLVERSE_SCHEDULE_PROVIDER,
                    book=NFLVERSE_SCHEDULE_BOOK,
                    source_event_id=game_id,
                    captured_at=stamp,
                    first_side="over",
                    first_line=total,
                    first_american_odds=over_price,
                    second_side="under",
                    second_line=total,
                    second_american_odds=under_price,
                )
            )

    return sorted(markets, key=lambda item: (item.game_id, item.market_type))


def is_research_only_market(market: ESPNTwoWayMarket) -> bool:
    return market.provider == NFLVERSE_SCHEDULE_PROVIDER
