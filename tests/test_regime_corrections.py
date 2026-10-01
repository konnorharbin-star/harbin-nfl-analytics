import polars as pl

from nfl.regime_corrections import evaluate_regime_corrections


def _stable_dataset() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(40):
            away_rest_edge = index < 20
            low_total = index % 2 == 0
            baseline_margin = float((index % 5) - 2)
            margin_residual = -3.0 if away_rest_edge else (1.0 if index % 2 else -1.0)
            baseline_total = 40.0 if low_total else 46.0
            total_residual = 2.0 if low_total else (1.0 if index % 3 else -1.0)
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 10),
                    "game_id": f"{season}-{index}",
                    "baseline_home_margin": baseline_margin,
                    "actual_home_margin": baseline_margin + margin_residual,
                    "margin_residual": margin_residual,
                    "baseline_total": baseline_total,
                    "actual_total": baseline_total + total_residual,
                    "total_residual": total_residual,
                    "ctx_rest_diff_days": -3.0 if away_rest_edge else 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_fixed_regime_corrections_can_survive_stable_signal() -> None:
    evaluation = evaluate_regime_corrections(_stable_dataset())

    assert evaluation.margin_away_rest.shadow_candidate is True
    assert evaluation.total_low_projection.shadow_candidate is True
    assert evaluation.margin_away_rest.positive_folds == 3
    assert evaluation.total_low_projection.positive_folds == 3
    assert all(
        fold.correction_points == -3.0
        for fold in evaluation.margin_away_rest.folds
    )
    assert all(
        fold.correction_points == 2.0
        for fold in evaluation.total_low_projection.folds
    )
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False


def test_future_outcomes_cannot_change_earlier_fold_correction() -> None:
    dataset = _stable_dataset()
    original = evaluate_regime_corrections(dataset)
    mutated = dataset.with_columns(
        pl.when(pl.col("season") >= 2024)
        .then(pl.col("actual_home_margin") + 50.0)
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin"),
        pl.when(pl.col("season") >= 2024)
        .then(pl.col("margin_residual") + 50.0)
        .otherwise(pl.col("margin_residual"))
        .alias("margin_residual"),
        pl.when(pl.col("season") >= 2024)
        .then(pl.col("actual_total") - 50.0)
        .otherwise(pl.col("actual_total"))
        .alias("actual_total"),
        pl.when(pl.col("season") >= 2024)
        .then(pl.col("total_residual") - 50.0)
        .otherwise(pl.col("total_residual"))
        .alias("total_residual"),
    )
    changed = evaluate_regime_corrections(mutated)

    assert (
        original.margin_away_rest.folds[0].correction_points
        == changed.margin_away_rest.folds[0].correction_points
    )
    assert (
        original.total_low_projection.folds[0].correction_points
        == changed.total_low_projection.folds[0].correction_points
    )


def test_wrong_direction_history_does_not_create_shadow_candidate() -> None:
    dataset = _stable_dataset().with_columns(
        pl.when((pl.col("season") == 2022) & (pl.col("ctx_rest_diff_days") <= -2.0))
        .then(pl.lit(3.0))
        .otherwise(pl.col("margin_residual"))
        .alias("margin_residual"),
    ).with_columns(
        pl.when((pl.col("season") == 2022) & (pl.col("ctx_rest_diff_days") <= -2.0))
        .then(pl.col("baseline_home_margin") + pl.col("margin_residual"))
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin")
    )

    evaluation = evaluate_regime_corrections(dataset)

    assert evaluation.margin_away_rest.folds[0].passed is False
    assert evaluation.margin_away_rest.shadow_candidate is False
