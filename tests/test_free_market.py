import polars as pl

from nfl.free_market import FreeNFLMarketStore


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2025_01_BUF_KC", "2025_02_BAL_CIN"],
            "season": [2025, 2025],
            "week": [1, 2],
            "home_team": ["KC", "CIN"],
            "away_team": ["BUF", "BAL"],
            "away_moneyline": [120, -135],
            "home_moneyline": [-140, 115],
            "spread_line": [2.5, -2.5],
            "away_spread_odds": [-105, -110],
            "home_spread_odds": [-115, -110],
            "total_line": [48.5, 46.0],
            "under_odds": [-108, None],
            "over_odds": [-112, None],
        }
    )


def _initial_lines() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "sportsbook": ["WSGT", "WSGT", "WSGT", "WSGT"],
            "type": ["SPREAD", "SPREAD", "TOTAL", "TOTAL"],
            "about": ["2025_01_BUF_KC"] * 4,
            "side": ["KC", "BUF", "Over", "Under"],
            "line": [-1.5, 1.5, 47.5, 47.5],
        }
    )


def test_free_archive_uses_opening_lines_and_final_prices() -> None:
    store = FreeNFLMarketStore(_schedules(), initial_lines=_initial_lines())

    quote = store.quote("2025_01_BUF_KC")

    assert quote is not None
    assert quote.opening_book == "WSGT"
    assert quote.open_home_spread == -1.5
    assert quote.final_home_spread == -2.5
    assert quote.open_total == 47.5
    assert quote.final_total == 48.5
    assert quote.open_home_ml == -140
    assert quote.open_away_ml == 120
    assert quote.has_distinct_open_spread is True
    assert quote.has_distinct_open_total is True
    assert quote.has_distinct_open_moneyline is False
    assert quote.open_home_spread_odds == -115
    assert quote.open_over_odds == -112


def test_free_archive_falls_back_to_final_when_opening_is_missing() -> None:
    store = FreeNFLMarketStore(_schedules())

    quote = store.quote("2025_02_BAL_CIN")

    assert quote is not None
    assert quote.open_home_spread == 2.5
    assert quote.final_home_spread == 2.5
    assert quote.open_total == 46.0
    assert quote.has_any_distinct_open is False
    assert quote.open_home_spread_odds == -110
    assert quote.open_away_spread_odds == -110
    assert quote.open_over_odds == -110
    assert quote.open_under_odds == -110
