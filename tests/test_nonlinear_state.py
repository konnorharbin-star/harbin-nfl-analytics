import numpy as np
import polars as pl

from nfl.nonlinear_state import NonlinearStateResidualModel, nonlinear_spec
from nfl.nonlinear_state_fixed import evaluate_nonlinear_state_rolling
from nfl.online_state import ONLINE_STATE_CONFIGS, ONLINE_STATE_FEATURES


def test_boosting_learns_nonlinear_interaction() -> None:
    rng = np.random.default_rng(26)
    x1 = rng.uniform(-2.0, 2.0, 600)
    x2 = rng.uniform(-2.0, 2.0, 600)
    target = 2.5 * x1 * x2
    frame = pl.DataFrame({"x1": x1, "x2": x2, "target": target})

    train = frame.head(450)
    test = frame.tail(150)
    model = NonlinearStateResidualModel(
        nonlinear_spec("boost_d2"),
        features=("x1", "x2"),
    ).fit(train, "target")
    prediction = model.predict(test)
    actual = np.asarray(test.get_column("target"), dtype=float)

    baseline_rmse = float(np.sqrt(np.mean(np.square(actual))))
    model_rmse = float(np.sqrt(np.mean(np.square(actual - prediction))))
    assert model_rmse < baseline_rmse * 0.75


def _zero_dataset(config_name: str) -> pl.DataFrame:
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for i in range(180):
            baseline_margin = ((i % 13) - 6) * 0.4
            baseline_total = 41.0 + (i % 11)
            row = {
                "season": season,
                "margin_residual": 0.0,
                "total_residual": 0.0,
                "actual_home_margin": baseline_margin,
                "baseline_home_margin": baseline_margin,
                "actual_total": baseline_total,
                "baseline_total": baseline_total,
                "state_config": config_name,
            }
            for j, name in enumerate(ONLINE_STATE_FEATURES):
                row[name] = float(((i + 1) * (j + 3) + season) % 23)
            rows.append(row)
    return pl.DataFrame(rows)


def test_nonlinear_gate_keeps_weight_zero_when_no_candidate_improves() -> None:
    datasets = {
        config.name: _zero_dataset(config.name)
        for config in ONLINE_STATE_CONFIGS
    }
    evaluation = evaluate_nonlinear_state_rolling(
        datasets,
        model_specs=("boost_d2",),
        blend_grid=(0.50,),
    )

    assert evaluation.margin.selected is None
    assert evaluation.margin.shadow_candidate is False
    assert evaluation.margin.best_tested.positive_folds == 0
    assert evaluation.total.selected is None
    assert evaluation.total.shadow_candidate is False
    assert evaluation.total.best_tested.positive_folds == 0
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False
