from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_error_tails import analyze_error_tails
from nfl.game_intelligence_factors import PREGAME_FEATURES


def _snapshot() -> pl.DataFrame:
    rows = []
    for index in range(100):
        row = {
            "season": 2025, "week": 5 + index // 10,
            "game_id": f"2025_game_{index}",
            "margin_residual": 20.0 if index < 40 else 0.0,
            "total_residual": -20.0 if index >= 60 else 0.0,
        }
        row.update({feature: -0.2 if index < 50 else 0.2
                    for feature in PREGAME_FEATURES})
        rows.append(row)
    return pl.DataFrame(rows)


def test_extreme_error_rates_and_no_staking():
    report = analyze_error_tails(_snapshot())
    assert report["targets"]["margin"]["extreme_errors"] == 40
    assert report["targets"]["total"]["extreme_errors"] == 40
    assert report["targets"]["margin"]["extreme_error_rate"] == 0.4
    assert report["staking_authorized"] is False


def test_duplicate_games_fail_closed():
    snapshot = _snapshot()
    with pytest.raises(DataContractError):
        analyze_error_tails(pl.concat([snapshot, snapshot.head(1)]))
