import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_market_relative import (
    build_market_relative_games,
    evaluate_market_relative_games,
)


def _fixtures():
    games = pl.DataFrame({
        "season": [2024, 2025], "week": [5, 5],
        "game_id": ["g1", "g2"],
        "projected_home_margin": [4.0, -4.0],
        "projected_total": [44.0, 40.0],
        "actual_home_margin": [7.0, -7.0],
        "actual_total": [45.0, 39.0],
    })
    market = pl.DataFrame({
        "season": [2024, 2025], "week": [5, 5],
        "game_id": ["g1", "g2"],
        "spread_line": [6.0, -6.0], "total_line": [40.0, 45.0],
    })
    return games, market


def test_home_favorite_sign_and_paired_scoring():
    preds, schedules = _fixtures()
    games = build_market_relative_games(preds, schedules)
    assert games["market_home_margin"].to_list() == [6.0, -6.0]
    assert games["model_margin_error"].to_list() == [3.0, 3.0]
    assert games["market_margin_error"].to_list() == [1.0, 1.0]
    report = evaluate_market_relative_games(games)
    assert report["targets"]["margin"]["missing_market_games"] == 0
    assert report["targets"]["margin"]["model_beat_archive_proxy_both_years"] is False
    assert report["closing_timestamp_verified"] is False
    assert report["staking_authorized"] is False


def test_missing_market_excluded_not_filled():
    preds, schedules = _fixtures()
    schedules = schedules.filter(pl.col("game_id") == "g1")
    report = evaluate_market_relative_games(
        build_market_relative_games(preds, schedules)
    )
    assert report["targets"]["margin"]["missing_market_games"] == 1


def test_duplicate_market_keys_fail_closed():
    preds, schedules = _fixtures()
    with pytest.raises(DataContractError):
        build_market_relative_games(
            preds, pl.concat([schedules, schedules.head(1)])
        )
