import math

import polars as pl

from nfl.residuals import RESIDUAL_FEATURES, ResidualRidgeModel, evaluate_nested_holdout


def _dataset(holdout_multiplier: float = 1.0) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for season in (2022, 2023, 2024):
        for index in range(30):
            x = (index - 14.5) / 10.0
            multiplier = holdout_multiplier if season == 2024 else 1.0
            margin_residual = 2.0 * x * multiplier
            total_residual = -1.5 * x * multiplier
            row: dict[str, float | int] = {
                "season": season,
                "baseline_home_margin": 0.0,
                "actual_home_margin": margin_residual,
                "margin_residual": margin_residual,
                "baseline_total": 44.0,
                "actual_total": 44.0 + total_residual,
                "total_residual": total_residual,
            }
            for feature_index, feature in enumerate(RESIDUAL_FEATURES):
                row[feature] = x if feature_index == 0 else 0.0
            rows.append(row)
    return pl.DataFrame(rows)


def test_residual_ridge_learns_training_signal() -> None:
    dataset = _dataset().filter(pl.col("season") == 2022)
    model = ResidualRidgeModel(alpha=0.1).fit(dataset, "margin_residual")
    predictions = model.predict(dataset)

    assert math.isclose(float(predictions.mean()), 0.0, abs_tol=1e-9)
    assert float(abs(predictions - dataset.get_column("margin_residual").to_numpy()).mean()) < 0.05


def test_nested_holdout_improves_known_synthetic_signal() -> None:
    evaluation = evaluate_nested_holdout(
        _dataset(),
        validation_season=2023,
        holdout_season=2024,
        min_training_games=24,
    )

    assert evaluation.training_games == 30
    assert evaluation.validation_games == 30
    assert evaluation.holdout_games == 30
    assert evaluation.margin.candidate_pass
    assert evaluation.total.candidate_pass
    assert evaluation.margin.adjusted_rmse < evaluation.margin.baseline_rmse
    assert evaluation.total.adjusted_rmse < evaluation.total.baseline_rmse


def test_alpha_selection_does_not_use_holdout_outcomes() -> None:
    normal = evaluate_nested_holdout(
        _dataset(holdout_multiplier=1.0),
        validation_season=2023,
        holdout_season=2024,
        min_training_games=24,
    )
    changed_holdout = evaluate_nested_holdout(
        _dataset(holdout_multiplier=10.0),
        validation_season=2023,
        holdout_season=2024,
        min_training_games=24,
    )

    assert normal.margin.alpha == changed_holdout.margin.alpha
    assert normal.total.alpha == changed_holdout.total.alpha
