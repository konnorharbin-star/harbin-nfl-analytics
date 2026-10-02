from __future__ import annotations

import polars as pl

from nfl.personnel_context import (
    _latest_weekly_depth_snapshot,
    _non_qb_starter_risk,
)
from nfl.personnel_context_eval import evaluate_fixed_personnel_rolling


def _fixed_signal_dataset() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(160):
            signal = -1.0 if index % 2 else 1.0
            baseline_margin = float((index % 7) - 3)
            actual_margin = baseline_margin + (1.5 * signal)
            baseline_total = 43.0 + float(index % 5)
            rows.append(
                {
                    "season": season,
                    "margin_residual": actual_margin - baseline_margin,
                    "total_residual": 0.0,
                    "actual_home_margin": actual_margin,
                    "baseline_home_margin": baseline_margin,
                    "actual_total": baseline_total,
                    "baseline_total": baseline_total,
                    "personnel_starter_diff": signal,
                    "personnel_starter_sum": abs(signal),
                }
            )
    return pl.DataFrame(rows)


def test_fixed_personnel_margin_candidate_must_clear_every_fold() -> None:
    evaluation = evaluate_fixed_personnel_rolling(
        _fixed_signal_dataset(),
        feature_sets=("starters",),
        ridge_grid=(1.0,),
    )

    assert evaluation.margin.feature_set == "starters"
    assert evaluation.margin.ridge_alpha == 1.0
    assert evaluation.margin.positive_folds == 3
    assert evaluation.margin.shadow_candidate is True
    assert all(fold.passed for fold in evaluation.margin.folds)
    assert evaluation.margin.adjusted_mae < evaluation.margin.baseline_mae
    assert evaluation.margin.adjusted_rmse < evaluation.margin.baseline_rmse
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False


def test_fixed_personnel_total_collapses_to_baseline_when_no_signal() -> None:
    evaluation = evaluate_fixed_personnel_rolling(
        _fixed_signal_dataset(),
        feature_sets=("starters",),
        ridge_grid=(1.0,),
    )

    assert evaluation.total.feature_set == "disabled"
    assert evaluation.total.ridge_alpha is None
    assert evaluation.total.shadow_candidate is False
    assert evaluation.total.adjusted_mae == evaluation.total.baseline_mae
    assert evaluation.total.adjusted_rmse == evaluation.total.baseline_rmse


def test_non_qb_starter_risk_excludes_quarterback_injury() -> None:
    depth = pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "qb",
                "position": "QB",
                "depth_rank": 1,
            },
            {
                "team": "AAA",
                "player_key": "wr",
                "position": "WR",
                "depth_rank": 1,
            },
        ]
    )
    qb_only = pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "qb",
                "severity": 1.0,
            }
        ]
    )
    mixed = pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "qb",
                "severity": 1.0,
            },
            {
                "team": "AAA",
                "player_key": "wr",
                "severity": 0.5,
            },
        ]
    )

    assert _non_qb_starter_risk("AAA", depth, qb_only) == 0.0
    assert _non_qb_starter_risk("AAA", depth, mixed) == 0.5


def test_latest_weekly_depth_snapshot_drops_stale_prior_starters() -> None:
    depth = pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "old-wr",
                "player_name": "Old WR",
                "position": "WR",
                "depth_rank": 1,
                "depth_week": 4,
                "depth_captured_at": None,
            },
            {
                "team": "AAA",
                "player_key": "new-wr",
                "player_name": "New WR",
                "position": "WR",
                "depth_rank": 1,
                "depth_week": 5,
                "depth_captured_at": None,
            },
        ]
    )

    latest = _latest_weekly_depth_snapshot(depth)

    assert latest.height == 1
    assert latest.row(0, named=True)["player_key"] == "new-wr"
    assert latest.row(0, named=True)["depth_week"] == 5
