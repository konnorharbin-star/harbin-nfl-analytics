"""Disagreement analysis cohort safety tests."""
from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.market_disagreement_diagnostics import audit_disagreements


def _games():
    return pl.DataFrame({
        "season": [2024, 2024, 2025],
        "week": [5, 6, 5], "game_id": ["a", "b", "c"],
        "projected_home_margin": [7.0, -7.0, 4.0],
        "market_home_margin": [3.0, -3.0, 3.0],
        "actual_home_margin": [9.0, -9.0, 3.0],
        "projected_total": [51.0, 38.0, 47.0],
        "market_total": [46.0, 44.0, 46.0],
        "actual_total": [52.0, 37.0, 46.0],
    })


def test_missing_sample_is_not_a_validated_disagreement_edge():
    result = audit_disagreements(_games())
    assert result["targets"]["margin"]["by_season"]["2024"]["covered_games"] == 2
    assert all(
        bucket["status"] == "INSUFFICIENT_SAMPLE"
        for bucket in result["targets"]["margin"]["by_season"]["2024"]["bins"]
    )
    assert result["not_staking_evidence"] is True


def test_duplicate_game_fails_closed():
    data = _games()
    with pytest.raises(DataContractError):
        audit_disagreements(pl.concat([data, data.head(1)]))


def test_missing_market_quote_excluded_not_imputed():
    data = _games().with_columns(
        pl.when(pl.col("game_id") == "a").then(None)
        .otherwise(pl.col("market_total")).alias("market_total")
    )
    result = audit_disagreements(data)
    assert result["targets"]["total"]["missing_games"] == 1
