from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.core_market_challenger import compare_core_challenger


def test_duplicate_schedule_games_fail_closed():
    row = pl.DataFrame({
        "season": [2024], "week": [5], "game_id": ["g1"],
        "spread_line": [3.0], "total_line": [45.0],
    })
    with pytest.raises(DataContractError):
        compare_core_challenger(pl.concat([row, row]))


def test_invalid_season_order_rejected():
    empty = pl.DataFrame(schema={
        "season": pl.Int64, "week": pl.Int64, "game_id": pl.String,
        "spread_line": pl.Float64, "total_line": pl.Float64,
    })
    with pytest.raises(ValueError):
        compare_core_challenger(empty, seasons=(2025, 2024))
