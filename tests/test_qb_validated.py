import polars as pl

from nfl.qb_validated import ValidatedQBAdjustment


def _training_frame() -> pl.DataFrame:
    rows = []
    for index in range(120):
        epa_margin = ((index % 9) - 4) / 10.0
        epa_total = ((index % 7) - 3) / 10.0
        cpoe_total = float((index % 5) - 2)
        rows.append(
            {
                "margin_residual": 2.0 * epa_margin,
                "total_residual": (1.5 * epa_total) + (0.1 * cpoe_total),
                "baseline_home_margin": float((index % 6) - 3),
                "baseline_total": 44.0 + float(index % 4),
                "qb_epa_margin_signal": epa_margin,
                "qb_epa_total_signal": epa_total,
                "qb_cpoe_total_signal": cpoe_total,
            }
        )
    return pl.DataFrame(rows)


def test_validated_qb_adjustment_uses_frozen_target_specific_features() -> None:
    training = _training_frame()
    model = ValidatedQBAdjustment().fit(training)
    scored = model.apply(training.head(3))

    assert model.margin_model.features == ("qb_epa_margin_signal",)
    assert model.total_model.features == (
        "qb_epa_total_signal",
        "qb_cpoe_total_signal",
    )
    assert "qb_adjusted_home_margin" in scored.columns
    assert "qb_adjusted_total" in scored.columns
    assert "qb_adjusted_home_points" in scored.columns
    assert "qb_adjusted_away_points" in scored.columns
