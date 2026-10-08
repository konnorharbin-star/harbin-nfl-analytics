"""Test market-relative efficiency comparisons fail closed."""
from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.efficiency_market_challenger import evaluate_efficiency_market


def _fixture():
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for i in range(110):
            edge = (i % 11 - 5) / 5
            row = {
                "season": season, "week": 5 + i // 10,
                "game_id": f"{season}-{i}",
                "baseline_home_margin": 3.0, "baseline_total": 44.0,
                "margin_residual": edge, "total_residual": -edge,
                "actual_home_margin": 3.0 + edge,
                "actual_total": 44.0 - edge,
                "spread_line": 3.0, "total_line": 44.0,
            }
            row.update({
                f"recent_{metric}_{target}_signal": edge
                for metric in ("epa_per_play", "success_rate", "explosive_rate")
                for target in ("margin", "total")
            })
            rows.append(row)
    return pl.DataFrame(rows)


def test_fixed_historical_folds():
    report = evaluate_efficiency_market(_fixture())
    for target in ("margin", "total"):
        folds = report["targets"][target]["folds"]
        assert [x["train_games"] for x in folds] == [220, 330]
        assert [x["test_games"] for x in folds] == [110, 110]
    assert report["train_uses_market"] is False
    assert report["staking_authorized"] is False


def test_duplicate_game_keys_fail_closed():
    frame = _fixture()
    with pytest.raises(DataContractError):
        evaluate_efficiency_market(pl.concat([frame, frame.head(1)]))


def test_future_results_do_not_affect_earlier_fold():
    a = _fixture()
    b = a.with_columns(
        pl.when(pl.col("season") == 2025)
        .then(pl.col("margin_residual") + 20)
        .otherwise(pl.col("margin_residual")).alias("margin_residual"),
    )
    first = evaluate_efficiency_market(a)
    second = evaluate_efficiency_market(b)
    assert first["targets"]["margin"]["folds"][0] == (
        second["targets"]["margin"]["folds"][0]
    )
