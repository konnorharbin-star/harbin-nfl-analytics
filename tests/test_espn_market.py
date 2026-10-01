from datetime import UTC, datetime

import polars as pl

from nfl.espn_market import ESPNMarketClient, parse_espn_odds


def _odds() -> dict[str, object]:
    return {
        "provider": {"name": "ESPN BET"},
        "details": "KC -3.5",
        "spread": 3.5,
        "overUnder": 47.5,
        "overOdds": -108,
        "underOdds": -112,
        "homeTeamOdds": {
            "favorite": True,
            "moneyLine": -165,
            "spreadOdds": -105,
        },
        "awayTeamOdds": {
            "favorite": False,
            "moneyLine": 145,
            "spreadOdds": -115,
        },
    }


def test_parse_espn_odds_builds_all_three_markets() -> None:
    captured = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    markets = parse_espn_odds(
        _odds(),
        game_id="2026_04_BUF_KC",
        event_id="401999999",
        home_team="KC",
        away_team="BUF",
        captured_at=captured,
    )

    assert {market.market_type for market in markets} == {
        "moneyline",
        "spread",
        "total",
    }
    spread = next(market for market in markets if market.market_type == "spread")
    assert spread.first_side == "home"
    assert spread.first_line == -3.5
    assert spread.second_line == 3.5
    assert spread.first_american_odds == -105
    total = next(market for market in markets if market.market_type == "total")
    assert total.first_line == 47.5
    assert total.first_american_odds == -108
    assert total.second_american_odds == -112


def test_espn_current_market_maps_team_aliases_without_network() -> None:
    targets = pl.DataFrame(
        {
            "game_id": ["2026_04_JAX_LA"],
            "home_team": ["LA"],
            "away_team": ["JAX"],
        }
    )

    class FixtureClient(ESPNMarketClient):
        def scoreboard(self, *, week: int) -> dict[str, object]:
            assert week == 4
            return {
                "events": [
                    {
                        "id": "401888888",
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "LAR"},
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "JAC"},
                                    },
                                ],
                                "odds": [_odds()],
                            }
                        ],
                    }
                ]
            }

        def _core_odds(self, event_id: str) -> list[dict[str, object]]:
            return []

    markets = FixtureClient().current_markets(targets, week=4)

    assert len(markets) == 3
    assert {market.game_id for market in markets} == {"2026_04_JAX_LA"}
    assert {market.source_event_id for market in markets} == {"401888888"}
