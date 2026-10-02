import polars as pl

from nfl.nested_probability import evaluate_nested_probability


def _dataset() -> pl.DataFrame:
    rows = []
    game_id = 0
    for season in (2021, 2022, 2023, 2024, 2025):
        for week in range(5, 19):
            for game in range(12):
                game_id += 1
                projected_margin = float(((game * 3 + week) % 17) - 8)
                projected_total = 43.0 + float((game + week) % 8)
                margin_error = float(((game_id * 7) % 25) - 12)
                total_error = float(((game_id * 11) % 29) - 14)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"g{game_id}",
                        "projected_home_margin": projected_margin,
                        "projected_total": projected_total,
                        "actual_home_margin": projected_margin + margin_error,
                        "actual_total": projected_total + total_error,
                    }
                )
    return pl.DataFrame(rows)


def test_evaluation_outcomes_cannot_change_selected_hyperparameters() -> None:
    dataset = _dataset()
    first = evaluate_nested_probability(
        dataset,
        min_core_rows=300,
        min_section_rows=60,
    )
    start = first.partition.evaluation.start
    assert start is not None
    start_season, start_week = start
    in_evaluation = (pl.col("season") > start_season) | (
        (pl.col("season") == start_season) & (pl.col("week") >= start_week)
    )
    mutated = dataset.with_columns(
        pl.when(in_evaluation)
        .then(pl.col("actual_home_margin") + 30.0)
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin"),
        pl.when(in_evaluation)
        .then(pl.col("actual_total") + 40.0)
        .otherwise(pl.col("actual_total"))
        .alias("actual_total"),
    )
    second = evaluate_nested_probability(
        mutated,
        min_core_rows=300,
        min_section_rows=60,
    )

    assert second.margin_scale == first.margin_scale
    assert second.total_scale == first.total_scale
    assert second.logistic_alpha == first.logistic_alpha
    assert second.partition == first.partition
    assert second.evaluation_selected_gaussian.margin_nll != (
        first.evaluation_selected_gaussian.margin_nll
    )


def test_calibration_and_evaluation_are_separate_whole_week_blocks() -> None:
    result = evaluate_nested_probability(
        _dataset(),
        min_core_rows=300,
        min_section_rows=60,
    )

    assert result.partition.calibration.end < result.partition.evaluation.start
    assert result.calibration_rows == result.partition.calibration.rows
    assert result.evaluation_rows == result.partition.evaluation.rows
    assert result.selection_uses_evaluation is False
    assert 0.0 <= result.evaluation_logistic_win.brier <= 1.0
    assert result.evaluation_logistic_win.log_loss > 0
    assert 0.0 <= result.evaluation_logistic_win.ece <= 1.0
    assert result.canonical_probability_change_enabled is False
    assert result.promotion_eligible is False
