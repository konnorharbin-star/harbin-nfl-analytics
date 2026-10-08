from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_factors import (
    PREGAME_FEATURES,
    attach_pregame_factors,
    evaluate_factor_diagnostics,
)


def test_join_keeps_future_game_features_out():
    graded = pl.DataFrame({
        "season": [2026], "week": [5], "game_id": ["g1"],
        "margin_residual": [4.0], "total_residual": [-3.0],
        "actual_total": [41.0],
    })
    snapshot = pl.DataFrame({
        "season": [2026], "week": [5], "game_id": ["g1"],
        PREGAME_FEATURES[0]: [0.32],
        "actual_total": [999.0],
    })
    joined = attach_pregame_factors(
        graded, snapshot, feature_columns=(PREGAME_FEATURES[0],)
    )
    assert joined[PREGAME_FEATURES[0]][0] == 0.32
    assert joined["actual_total"][0] == 41.0
    assert joined.height == 1


def test_duplicate_snapshots_and_outcome_features_fail_closed():
    graded = pl.DataFrame({
        "season": [2026], "week": [5], "game_id": ["g1"],
        "margin_residual": [4.0], "total_residual": [-3.0],
    })
    row = pl.DataFrame({
        "season": [2026], "week": [5], "game_id": ["g1"],
        "recent_epa_per_play_margin_signal": [0.1],
        "actual_home_margin": [3.0],
    })
    with pytest.raises(DataContractError):
        attach_pregame_factors(
            graded, pl.concat([row, row]),
            feature_columns=("recent_epa_per_play_margin_signal",)
        )
    with pytest.raises(DataContractError):
        attach_pregame_factors(
            graded, row, feature_columns=("actual_home_margin",)
        )


def test_small_sample_fails_closed_without_edge_claim():
    frame = pl.DataFrame({
        PREGAME_FEATURES[0]: [0.3, -0.2],
        "margin_residual": [5.0, -2.0],
        "total_residual": [3.0, 2.0],
    })
    result = evaluate_factor_diagnostics(frame, (PREGAME_FEATURES[0],))
    assert result["features"][PREGAME_FEATURES[0]]["status"] == "INSUFFICIENT_SAMPLE"
    assert result["staking_authorized"] is False
    assert result["pregame_model_modified"] is False
