from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.ensemble_dataset import (
    _join_namespace,
    assert_no_feature_leakage,
    build_ensemble_walkforward_dataset,
    ensemble_feature_columns,
)
from nfl.ensemble_residuals import evaluate_nonlinear_ensemble


def _base_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "game_id": ["a", "b"],
            "gameday": ["2024-10-01", "2024-10-02"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "baseline_home_margin": [2.0, -1.0],
            "baseline_total": [44.0, 47.0],
            "actual_home_margin": [3.0, -2.0],
            "actual_total": [45.0, 46.0],
            "margin_residual": [1.0, -1.0],
            "total_residual": [1.0, -1.0],
            "epa_per_play_matchup_advantage": [0.1, -0.2],
        }
    )


def test_feature_contract_blocks_postgame_and_market_fields() -> None:
    frame = _base_frame().with_columns(
        pl.Series("market_edge", [0.1, 0.2]),
        pl.Series("closing_total", [44.5, 47.5]),
        pl.Series("safe_signal", [1.0, 2.0]),
    )
    columns = ensemble_feature_columns(frame)

    assert "safe_signal" in columns
    assert "baseline_home_margin" in columns
    assert "market_edge" not in columns
    assert "closing_total" not in columns
    assert "actual_home_margin" not in columns
    assert "margin_residual" not in columns
    assert_no_feature_leakage(columns)

    with pytest.raises(DataContractError):
        assert_no_feature_leakage(["safe_signal", "market_price"])


def test_supplemental_join_fails_on_coverage_mismatch() -> None:
    base = _base_frame()
    extra = pl.DataFrame(
        {
            "season": [2024],
            "week": [5],
            "game_id": ["a"],
            "signal": [1.0],
        }
    )
    with pytest.raises(DataContractError, match="changed coverage"):
        _join_namespace(base, extra, prefix="extra")


def test_combined_dataset_namespaces_only_safe_supplemental_features(
    monkeypatch,
) -> None:
    base = _base_frame()
    supplemental = base.select("season", "week", "game_id").with_columns(
        pl.Series("useful", [1.0, 2.0]),
        pl.Series("actual_total", [45.0, 46.0]),
        pl.Series("baseline_total", [44.0, 47.0]),
    )

    monkeypatch.setattr(
        "nfl.ensemble_dataset.build_walkforward_dataset",
        lambda *args, **kwargs: base,
    )
    monkeypatch.setattr(
        "nfl.ensemble_dataset.build_situational_walkforward_dataset",
        lambda *args, **kwargs: supplemental,
    )
    monkeypatch.setattr(
        "nfl.ensemble_dataset.build_drive_walkforward_dataset",
        lambda *args, **kwargs: supplemental,
    )
    monkeypatch.setattr(
        "nfl.ensemble_dataset.build_qb_walkforward_dataset",
        lambda *args, **kwargs: supplemental,
    )

    result = build_ensemble_walkforward_dataset(
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame({"season": [2024]}),
        2024,
    )

    assert "situ__useful" in result.columns
    assert "drive__useful" in result.columns
    assert "qb__useful" in result.columns
    assert "situ__actual_total" not in result.columns
    assert "drive__baseline_total" not in result.columns
    assert result.height == base.height


def _zero_residual_dataset() -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(26)
    for season in (2022, 2023, 2024, 2025):
        for index in range(110):
            margin = float(rng.normal(0.0, 7.0))
            total = float(rng.normal(45.0, 8.0))
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 14),
                    "game_id": f"{season}-{index}",
                    "baseline_home_margin": margin,
                    "baseline_total": total,
                    "actual_home_margin": margin,
                    "actual_total": total,
                    "margin_residual": 0.0,
                    "total_residual": 0.0,
                    "football_signal": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


def test_nonlinear_ensemble_selects_zero_when_tune_cannot_improve() -> None:
    result = evaluate_nonlinear_ensemble(
        _zero_residual_dataset(),
        ridge_alphas=(10.0,),
        ridge_fractions=(0.5,),
        boost_depths=(2,),
        weight_grid=(0.0, 0.5, 1.0),
    )

    assert result.margin.architecture_candidate is False
    assert result.total.architecture_candidate is False
    assert all(fold.residual_weight == 0.0 for fold in result.margin.folds)
    assert all(fold.residual_weight == 0.0 for fold in result.total.folds)
    assert result.margin.aggregate_adjusted_mae == result.margin.aggregate_baseline_mae
    assert result.total.aggregate_adjusted_rmse == result.total.aggregate_baseline_rmse
