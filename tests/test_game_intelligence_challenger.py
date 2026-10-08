from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_challenger import evaluate_intelligence_challenger
from nfl.game_intelligence_factors import PREGAME_FEATURES


def _games() -> pl.DataFrame:
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(90):
            value = (index % 11 - 5) / 5.0
            residual = value * 6 + (index % 3 - 1)
            row = {
                "season": season, "week": 5 + index // 10,
                "game_id": f"{season}_{index}",
                "margin_residual": residual, "total_residual": -residual,
                "projected_home_margin": 3.0, "projected_total": 44.0,
                "actual_home_margin": 3.0 + residual,
                "actual_total": 44.0 - residual,
            }
            row.update({name: value for name in PREGAME_FEATURES})
            rows.append(row)
    return pl.DataFrame(rows)


def test_walkforward_has_strictly_prior_season_training():
    result = evaluate_intelligence_challenger(_games())
    for target in ("margin", "total"):
        folds = result["targets"][target]["folds"]
        assert [fold["train_games"] for fold in folds] == [180, 270]
        assert all(fold["test_games"] == 90 for fold in folds)
        assert result["canonical_score_changed"] is False
        assert result["betting_policy_changed"] is False


def test_holdout_results_cannot_change_prior_fold():
    baseline = _games()
    changed = baseline.with_columns(
        pl.when(pl.col("season") == 2025).then(pl.col("actual_total") + 20)
        .otherwise(pl.col("actual_total")).alias("actual_total"),
        pl.when(pl.col("season") == 2025).then(pl.col("total_residual") + 20)
        .otherwise(pl.col("total_residual")).alias("total_residual"),
    )
    a = evaluate_intelligence_challenger(baseline)
    b = evaluate_intelligence_challenger(changed)
    assert a["targets"]["total"]["folds"][0] == b["targets"]["total"]["folds"][0]


def test_duplicate_games_fail_closed():
    games = _games()
    with pytest.raises(DataContractError):
        evaluate_intelligence_challenger(pl.concat([games, games.head(1)]))
