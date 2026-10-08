"""Fixed challenger market comparison safety tests."""
from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.factorized_market_benchmark import evaluate_factorized_market


def test_duplicate_schedule_keys_fail_closed():
    game = pl.DataFrame({
        "season": [2024], "week": [5], "game_id": ["g1"],
        "spread_line": [3.0], "total_line": [42.0],
    })
    with pytest.raises(DataContractError):
        evaluate_factorized_market(pl.concat([game, game]), pl.DataFrame())


def test_out_of_order_test_seasons_fail():
    empty = pl.DataFrame(schema={
        "season": pl.Int64, "week": pl.Int64, "game_id": pl.String,
        "spread_line": pl.Float64, "total_line": pl.Float64,
    })
    with pytest.raises(ValueError):
        evaluate_factorized_market(empty, pl.DataFrame(), seasons=(2025, 2024))
