from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.action_network import (
    ActionNetworkNFLClient,
    parse_action_network_game,
)
from nfl.espn_market import ESPNTwoWayMarket
from nfl.pro_market import CurrentOddsAPIClient, collect_current_markets


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026],
            "week": [4],
            "game_id": ["2026_04_BUF_KC"],
            "gameday": ["2026-10-04"],
            "gametime": ["20:20"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
        }
    )


def _action_game(*, timestamp: str | None = "2026-10-01T12:00:00Z") -> dict[str, object]:
    def row(book_id: int, side: str, value: float | None, odds: int):
        payload: dict[str, object] = {
            "book_id": book_id,
            "side": side,
            "odds": odds,
        }
        if value is not None:
            payload["value"] = value
        if timestamp is not None:
            payload["last_update"] = timestamp
        return payload

    return {
        "id": 991234,
        "home_team_id": 1,
        "away_team_id": 2,
        "teams": [
            {
                "id": 1,
                "display_name": "Kansas City Chiefs",
                "abbr": "KC",
            },
            {
                "id": 2,
                "display_name": "Buffalo Bills",
                "abbr": "BUF",
            },
        ],
        "markets": {
            "event": {
                "event": {
                    "moneyline": [
                        row(15, "home", None, -155),
                        row(15, "away", None, 135),
                        row(30, "home", None, -150),
                        row(30, "away", None, 130),
                    ],
                    "spread": [
                        row(15, "home", -3.0, -110),
                        row(15, "away", 3.0, -110),
                        row(30, "home", -2.5, -112),
                        row(30, "away", 2.5, -108),
                    ],
                    "total": [
                        row(15, "over", 47.5, -105),
                        row(15, "under", 47.5, -115),
                        row(30, "over", 47.0, -110),
                        row(30, "under", 47.0, -110),
                    ],
                }
            }
        },
    }


def _espn_draftkings() -> list[ESPNTwoWayMarket]:
    stamp = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    return [
        ESPNTwoWayMarket(
            game_id="2026_04_BUF_KC",
            market_type="moneyline",
            provider="espn",
            book="Draft Kings",
            source_event_id="espn-1",
            captured_at=stamp,
            first_side="home",
            first_line=None,
            first_american_odds=-160,
            second_side="away",
            second_line=None,
            second_american_odds=140,
        ),
        ESPNTwoWayMarket(
            game_id="2026_04_BUF_KC",
            market_type="spread",
            provider="espn",
            book="Draft Kings",
            source_event_id="espn-1",
            captured_at=stamp,
            first_side="home",
            first_line=-3.0,
            first_american_odds=-110,
            second_side="away",
            second_line=3.0,
            second_american_odds=-110,
        ),
        ESPNTwoWayMarket(
            game_id="2026_04_BUF_KC",
            market_type="total",
            provider="espn",
            book="Draft Kings",
            source_event_id="espn-1",
            captured_at=stamp,
            first_side="over",
            first_line=47.5,
            first_american_odds=-110,
            second_side="under",
            second_line=47.5,
            second_american_odds=-110,
        ),
    ]


def test_action_network_normalizes_independent_per_book_markets() -> None:
    rows = parse_action_network_game(
        _action_game(),
        {("KC", "BUF"): "2026_04_BUF_KC"},
    )

    assert len(rows) == 6
    assert all(row.source_quote_time_verified for row in rows)
    assert all(row.source_quote_at is not None for row in rows)
    assert {row.book for row in rows} == {"DraftKings", "FanDuel"}
    assert {
        (row.book, row.market_type)
        for row in rows
    } == {
        ("DraftKings", "moneyline"),
        ("DraftKings", "spread"),
        ("DraftKings", "total"),
        ("FanDuel", "moneyline"),
        ("FanDuel", "spread"),
        ("FanDuel", "total"),
    }
    fanduel_spread = next(
        row
        for row in rows
        if row.book == "FanDuel" and row.market_type == "spread"
    )
    assert fanduel_spread.first_line == -2.5
    assert fanduel_spread.second_line == 2.5
    assert fanduel_spread.first_american_odds == -112
    assert fanduel_spread.second_american_odds == -108


def test_action_network_requires_provider_timestamp() -> None:
    rows = parse_action_network_game(
        _action_game(timestamp=None),
        {("KC", "BUF"): "2026_04_BUF_KC"},
    )

    assert rows == []


def test_action_network_can_use_explicit_collector_observation_time() -> None:
    observed = datetime(2026, 10, 1, 12, 30, tzinfo=UTC)
    rows = parse_action_network_game(
        _action_game(timestamp=None),
        {("KC", "BUF"): "2026_04_BUF_KC"},
        observed_at=observed,
    )

    assert len(rows) == 6
    assert {row.captured_at for row in rows} == {observed}
    assert all(not row.source_quote_time_verified for row in rows)
    assert all(row.source_quote_at is None for row in rows)


def test_action_network_client_uses_free_scoreboard_and_target_week() -> None:
    seen: list[str] = []

    def fetch(url: str) -> object:
        seen.append(url)
        return {"games": [_action_game()]}

    client = ActionNetworkNFLClient(fetch_json=fetch)
    rows = client.current_markets(_targets(), week=4)

    assert len(rows) == 6
    assert len(seen) == 1
    assert "/web/v2/scoreboard/nfl?" in seen[0]
    assert "season=2026" in seen[0]
    assert "week=4" in seen[0]


def test_current_aggregation_deduplicates_draftkings_identity_for_breadth() -> None:
    class ESPNFixture:
        def current_markets(self, targets: pl.DataFrame, *, week: int):
            assert targets.height == 1
            assert week == 4
            return _espn_draftkings()

    action = ActionNetworkNFLClient(
        fetch_json=lambda _: {"games": [_action_game()]}
    )
    no_paid = CurrentOddsAPIClient(api_key=None)

    rows, meta = collect_current_markets(
        _targets(),
        week=4,
        espn_client=ESPNFixture(),
        action_client=action,
        pro_client=no_paid,
    )

    assert len(rows) == 9
    assert meta["action_network_rows"] == 6
    assert meta["action_network_books"] == 2
    assert meta["books"] == 2
    assert meta["multi_book_games"] == 1
    assert meta["multi_book_coverage"] == 1.0
    assert meta["verified_sources"] == ["action_network", "espn"]


def test_book_only_has_one_side_source_time_not_both():
    data = _action_game()
    # The paired moneyline may still be shown for non-execution research, but
    # no source-origin timestamp is certified for both quoted sides.
    away = data["markets"]["event"]["event"]["moneyline"][1]
    away.pop("last_update")
    rows = parse_action_network_game(
        data, {("KC", "BUF"): "2026_04_BUF_KC"},
        observed_at=datetime(2026, 10, 1, 12, 5, tzinfo=UTC),
    )
    moneyline = next(x for x in rows
                     if x.book=="DraftKings" and x.market_type=="moneyline")
    assert moneyline.source_quote_at is None
    assert moneyline.source_quote_time_verified is False
    assert all(x.source_quote_time_verified
               for x in rows if x.market_type in ("spread","total"))


def test_event_timestamp_does_not_certify_book_quote_origin():
    data = _action_game(timestamp=None)
    data["markets"]["event"]["event"]["last_update"] = "2026-10-01T12:00:00Z"
    rows = parse_action_network_game(
        data, {("KC", "BUF"): "2026_04_BUF_KC"},
        observed_at=datetime(2026, 10, 1, 12, 5, tzinfo=UTC),
    )
    assert rows
    assert all(not x.source_quote_time_verified for x in rows)
