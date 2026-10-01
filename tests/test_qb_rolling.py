import polars as pl

from nfl.qb_rolling import evaluate_fixed_qb_rolling


def _signal_dataset() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(160):
            signal = -1.0 if index % 2 else 1.0
            baseline_margin = float((index % 5) - 2)
            actual_margin = baseline_margin + (2.0 * signal)
            baseline_total = 44.0 + float(index % 3)
            rows.append(
                {
                    "season": season,
                    "margin_residual": actual_margin - baseline_margin,
                    "total_residual": 0.0,
                    "actual_home_margin": actual_margin,
                    "baseline_home_margin": baseline_margin,
                    "actual_total": baseline_total,
                    "baseline_total": baseline_total,
                    "qb_epa_margin_signal": signal,
                    "qb_epa_total_signal": signal,
                }
            )
    return pl.DataFrame(rows)


def test_fixed_qb_margin_candidate_must_work_in_every_fold() -> None:
    evaluation = evaluate_fixed_qb_rolling(
        _signal_dataset(),
        feature_sets=("epa",),
        ridge_grid=(1.0,),
    )

    assert evaluation.margin.feature_set == "epa"
    assert evaluation.margin.ridge_alpha == 1.0
    assert evaluation.margin.positive_folds == 3
    assert evaluation.margin.shadow_candidate is True
    assert all(fold.passed for fold in evaluation.margin.folds)
    assert evaluation.margin.adjusted_mae < evaluation.margin.baseline_mae
    assert evaluation.margin.adjusted_rmse < evaluation.margin.baseline_rmse


def test_fixed_qb_total_collapses_to_zero_when_baseline_cannot_be_improved() -> None:
    evaluation = evaluate_fixed_qb_rolling(
        _signal_dataset(),
        feature_sets=("epa",),
        ridge_grid=(1.0,),
    )

    assert evaluation.total.feature_set == "disabled"
    assert evaluation.total.ridge_alpha is None
    assert evaluation.total.shadow_candidate is False
    assert evaluation.total.adjusted_mae == evaluation.total.baseline_mae
    assert evaluation.total.adjusted_rmse == evaluation.total.baseline_rmse
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False
