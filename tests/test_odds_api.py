from datetime import UTC, datetime

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.odds_api import (
    THE_ODDS_API_PROVIDER,
    TheOddsAPIClient,
    closing_decisions_from_history,
    fetch_market_history_for_decisions,
    historical_snapshot_to_market_history,
)


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025],
            "week": [5],
            "game_id": ["2025_05_NE_BUF"],
            "gameday": ["2025-10-05"],
            "home_team": ["BUF"],
            "away_team": ["NE"],
        }
    )


def _payload() -> dict[str, object]:
    return {
        "timestamp": "2025-10-05T15:55:00Z",
        "previous_timestamp": "2025-10-05T15:50:00Z",
        "next_timestamp": "2025-10-05T16:00:00Z",
        "data": [
            {
                "id": "provider-event-1",
                "sport_key": "americanfootball_nfl",
                "commence_time": "2025-10-05T17:00:00Z",
                "home_team": "Buffalo Bills",
                "away_team": "New England Patriots",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "last_update": "2025-10-05T15:54:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "last_update": "2025-10-05T15:54:00Z",
                                "outcomes": [
                                    {"name": "Buffalo Bills", "price": -150},
                                    {"name": "New England Patriots", "price": 130},
                                ],
                            },
                            {
                                "key": "spreads",
                                "last_update": "2025-10-05T15:54:00Z",
                                "outcomes": [
                                    {"name": "Buffalo Bills", "price": -110, "point": -3.5},
                                    {
                                        "name": "New England Patriots",
                                        "price": -110,
                                        "point": 3.5,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "last_update": "2025-10-05T15:54:00Z",
                                "outcomes": [
                                    {"name": "Over", "price": -105, "point": 44.5},
                                    {"name": "Under", "price": -115, "point": 44.5},
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_historical_snapshot_normalizes_featured_markets() -> None:
    frame = historical_snapshot_to_market_history(_payload(), _schedule())

    assert frame.height == 6
    assert set(frame.get_column("market_type").to_list()) == {
        "moneyline",
        "spread",
        "total",
    }
    assert set(frame.get_column("provider").to_list()) == {THE_ODDS_API_PROVIDER}
    assert set(frame.get_column("book").to_list()) == {"draftkings"}
    assert set(frame.get_column("source_event_id").to_list()) == {"provider-event-1"}
    assert set(frame.get_column("season").to_list()) == {2025}
    assert set(frame.get_column("week").to_list()) == {5}


class _FixtureClient:
    def historical_snapshot(self, at: datetime, **_: object) -> dict[str, object]:
        assert at.tzinfo is not None
        return _payload()


def test_fetch_market_history_requires_aware_decision_times() -> None:
    naive = pl.DataFrame(
        {
            "game_id": ["2025_05_NE_BUF"],
            "decision_time": [datetime(2025, 10, 5, 16, 0)],
        }
    )

    with pytest.raises(DataContractError, match="timezone-aware"):
        fetch_market_history_for_decisions(_FixtureClient(), _schedule(), naive)


def test_fetch_market_history_and_build_close_query() -> None:
    decisions = pl.DataFrame(
        {
            "game_id": ["2025_05_NE_BUF"],
            "decision_time": [datetime(2025, 10, 5, 16, 0, tzinfo=UTC)],
        }
    )
    history = fetch_market_history_for_decisions(_FixtureClient(), _schedule(), decisions)
    closing = closing_decisions_from_history(history, minutes_before_kickoff=5)

    assert history.height == 6
    assert closing.height == 1
    assert closing.row(0, named=True)["decision_time"] == datetime(
        2025, 10, 5, 16, 55, tzinfo=UTC
    )


def test_client_fails_closed_without_api_key(tmp_path: object) -> None:
    client = TheOddsAPIClient(api_key=None, cache_dir=tmp_path)

    with pytest.raises(DataContractError, match="THE_ODDS_API_KEY"):
        client.historical_snapshot(datetime(2025, 10, 5, 16, 0, tzinfo=UTC))
