from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence import build_game_intelligence, summarize_game_intelligence


def _sample() -> pl.DataFrame:
    return pl.DataFrame({
        "season": [2026, 2026], "week": [1, 2],
        "game_id": ["game-1", "game-2"],
        "home_team": ["AAA", "BBB"], "away_team": ["BBB", "AAA"],
        "projected_home_margin": [7.0, -2.0],
        "projected_total": [44.0, 40.0],
        "actual_home_margin": [3.0, -5.0],
        "actual_total": [51.0, 38.0],
    })


def test_intelligence_grades_only_completed_games():
    games = build_game_intelligence(_sample())
    assert games.height == 2
    assert games["margin_residual"].to_list() == [-4.0, -3.0]
    assert games["total_residual"].to_list() == [7.0, -2.0]
    report = summarize_game_intelligence(games)
    assert report["total"]["margin_mae"] == 3.5
    assert report["by_season"]["2026"]["games"] == 2
    assert report["staking_authorized"] is False


def test_duplicate_game_keys_fail_closed():
    rows = pl.concat([_sample(), _sample().head(1)])
    with pytest.raises(DataContractError):
        build_game_intelligence(rows)
