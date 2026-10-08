from __future__ import annotations

import polars as pl
import pytest

from nfl.bet_evidence_audit import audit_bet_evidence
from nfl.contracts import DataContractError


def _data():
    return pl.DataFrame({
        "season": [2024, 2024], "week": [5, 5],
        "game_id": ["g1", "g2"], "market_type": ["spread", "total"],
        "side": ["home", "over"], "result": ["win", "loss"],
        "net_units": [0.91, -1.0], "probability_edge": [0.1, 0.1],
        "expected_value_per_unit": [0.1, 0.1],
        "entry_line_observed": [False, True],
        "entry_price_verified": [False, True],
        "entry_price_stage": ["archive_final_fallback", "observed_pregame"],
    })


def test_fallback_never_counts_as_executable():
    result = audit_bet_evidence(_data())
    spread = result["targets"]["spread"]["0.0"]
    total = result["targets"]["total"]["0.0"]
    assert spread["research_bets"] == 1
    assert spread["verified_entry_bets"] == 0
    assert total["verified_entry_bets"] == 1
    assert result["bet_recommendation_approved"] is False


def test_duplicate_market_game_rejected():
    data = _data()
    with pytest.raises(DataContractError):
        audit_bet_evidence(pl.concat([data, data.head(1)]))
