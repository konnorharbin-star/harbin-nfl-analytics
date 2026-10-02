from __future__ import annotations

import numpy as np
import polars as pl

from nfl.isotonic_probability import evaluate_isotonic_probability


def _dataset() -> pl.DataFrame:
    rng = np.random.default_rng(25)
    rows: list[dict[str, object]] = []
    game = 0
    for season in (2021, 2022, 2023, 2024, 2025):
        for week in range(1, 17):
            for index in range(10):
                margin = float(rng.normal(0.0, 7.0))
                true_probability = 1.0 / (1.0 + np.exp(-margin / 7.5))
                home_win = bool(rng.random() < true_probability)
                actual_margin = float(
                    (1.0 if home_win else -1.0) * (1.0 + abs(rng.normal(6.0, 4.0)))
                )
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"{season}-{week}-{index}",
                        "projected_home_margin": margin,
                        "projected_total": 45.0 + float(rng.normal()),
                        "actual_home_margin": actual_margin,
                        "actual_total": 45.0 + float(rng.normal(0.0, 12.0)),
                    }
                )
                game += 1
    assert game == 800
    return pl.DataFrame(rows)


def test_isotonic_evaluation_uses_whole_week_partitions() -> None:
    result = evaluate_isotonic_probability(
        _dataset(),
        min_core_rows=300,
        min_section_rows=80,
    )
    assert result.partition.whole_week_boundaries is True
    assert result.partition.selection_uses_evaluation is False
    assert result.selection_uses_evaluation is False
    assert result.canonical_probability_change_enabled is False
    assert result.promotion_eligible is False


def test_evaluation_outcomes_cannot_change_selected_alpha() -> None:
    frame = _dataset()
    before = evaluate_isotonic_probability(
        frame,
        min_core_rows=300,
        min_section_rows=80,
    )
    evaluation_start = before.partition.evaluation.start
    assert evaluation_start is not None
    season, week = evaluation_start

    mutated = frame.with_columns(
        pl.when(
            (pl.col("season") > season)
            | ((pl.col("season") == season) & (pl.col("week") >= week))
        )
        .then(-pl.col("actual_home_margin"))
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin")
    )
    after = evaluate_isotonic_probability(
        mutated,
        min_core_rows=300,
        min_section_rows=80,
    )
    assert before.logistic_alpha == after.logistic_alpha
    assert before.partition.to_dict() == after.partition.to_dict()


def test_isotonic_metrics_are_finite() -> None:
    result = evaluate_isotonic_probability(
        _dataset(),
        min_core_rows=300,
        min_section_rows=80,
    )
    values = (
        result.raw_logistic.brier,
        result.raw_logistic.log_loss,
        result.raw_logistic.ece,
        result.isotonic.brier,
        result.isotonic.log_loss,
        result.isotonic.ece,
    )
    assert all(np.isfinite(value) for value in values)
