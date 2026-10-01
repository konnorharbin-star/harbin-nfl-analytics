import polars as pl

from nfl.residual_regimes import analyze_residual_regimes, annotate_residual_regimes


def _synthetic_frame(games_per_season: int = 30) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2023, 2024, 2025):
        for index in range(games_per_season):
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 12),
                    "game_id": f"{season}-{index}",
                    "baseline_home_margin": 1.0,
                    "actual_home_margin": 3.0,
                    "baseline_total": 45.0,
                    "actual_total": 43.0,
                    "ctx_rest_diff_days": 0.0,
                    "ctx_neutral_site": 0.0,
                }
            )
    return pl.DataFrame(rows)


def test_regime_labels_use_only_pregame_fields() -> None:
    base = pl.DataFrame(
        {
            "week": [6, 10, 16],
            "baseline_home_margin": [2.0, -5.0, 9.0],
            "baseline_total": [40.0, 45.0, 51.0],
            "ctx_rest_diff_days": [3.0, 0.0, -4.0],
            "ctx_neutral_site": [0.0, 1.0, 0.0],
        }
    )
    mutated = base.with_columns(
        pl.Series("actual_home_margin", [99.0, -99.0, 0.0]),
        pl.Series("actual_total", [100.0, 1.0, 77.0]),
    )

    labels = annotate_residual_regimes(base)
    mutated_labels = annotate_residual_regimes(mutated)
    regime_columns = [column for column in labels.columns if column.startswith("regime_")]

    assert labels.select(regime_columns).equals(mutated_labels.select(regime_columns))
    assert labels.get_column("regime_margin_band").to_list() == [
        "close_0_3",
        "medium_3_7",
        "large_7_plus",
    ]
    assert labels.get_column("regime_total_band").to_list() == [
        "low_under_42",
        "mid_42_48",
        "high_over_48",
    ]


def test_persistent_bias_requires_same_direction_sample_and_interval() -> None:
    report = analyze_residual_regimes(
        _synthetic_frame(),
        minimum_games_per_season=24,
        bootstrap_iterations=400,
    )

    assert "regime_margin_band=close_0_3" in report.persistent_margin_regimes
    assert "regime_total_band=mid_42_48" in report.persistent_total_regimes
    margin_band = next(
        item
        for item in report.margin
        if item.dimension == "regime_margin_band" and item.regime == "close_0_3"
    )
    total_band = next(
        item
        for item in report.total
        if item.dimension == "regime_total_band" and item.regime == "mid_42_48"
    )
    assert margin_band.bias_direction == "baseline_underpredicts"
    assert margin_band.bootstrap_low > 0.0
    assert total_band.bias_direction == "baseline_overpredicts"
    assert total_band.bootstrap_high < 0.0
    assert report.canonical_score_adjustment_enabled is False
    assert report.promotion_eligible is False


def test_small_regime_is_not_persistent() -> None:
    report = analyze_residual_regimes(
        _synthetic_frame(games_per_season=8),
        minimum_games_per_season=24,
        bootstrap_iterations=400,
    )
    assert report.persistent_margin_regimes == ()
    assert report.persistent_total_regimes == ()
