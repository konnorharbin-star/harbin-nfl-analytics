import math

import polars as pl

from nfl.probability import (
    ConditionalStudentTScoreDistribution,
    GaussianScoreDistribution,
    evaluate_conditional_probability_holdout,
    evaluate_probability_holdout,
)


def _training_frame(rows: int = 120) -> pl.DataFrame:
    projected_margin = [float((index % 9) - 4) for index in range(rows)]
    projected_total = [44.0 + float(index % 5) for index in range(rows)]
    margin_error = [float(((index * 7) % 21) - 10) for index in range(rows)]
    total_error = [float(((index * 11) % 25) - 12) for index in range(rows)]
    return pl.DataFrame(
        {
            "projected_home_margin": projected_margin,
            "projected_total": projected_total,
            "actual_home_margin": [
                projected_margin[index] + margin_error[index] for index in range(rows)
            ],
            "actual_total": [
                projected_total[index] + total_error[index] for index in range(rows)
            ],
        }
    )


def test_home_win_probability_is_monotonic_in_fair_margin() -> None:
    model = GaussianScoreDistribution().fit(_training_frame())

    underdog = model.home_win_probability(-7.0)
    pickem = model.home_win_probability(0.0)
    favorite = model.home_win_probability(7.0)

    assert 0.0 < underdog < pickem < favorite < 1.0


def test_cover_and_total_probabilities_move_with_thresholds() -> None:
    model = GaussianScoreDistribution().fit(_training_frame())

    assert model.home_cover_probability(3.0, 3.0) > model.home_cover_probability(3.0, -7.0)
    assert model.over_probability(47.0, 42.0) > model.over_probability(47.0, 52.0)


def test_distribution_columns_are_finite_probabilities() -> None:
    model = GaussianScoreDistribution().fit(_training_frame())
    projected = pl.DataFrame(
        {
            "projected_home_margin": [-3.0, 0.0, 6.0],
            "projected_total": [42.0, 46.0, 50.0],
        }
    )
    output = model.add_distribution_columns(projected)

    probabilities = output.get_column("home_win_probability").to_list()
    assert all(math.isfinite(value) and 0.0 < value < 1.0 for value in probabilities)
    assert output.get_column("margin_sigma").min() > 0
    assert output.get_column("total_sigma").min() > 0


def test_holdout_calibration_uses_only_prior_seasons() -> None:
    frames = []
    for season, residual_scale in [(2022, 1.0), (2023, 1.0), (2024, 1.2), (2025, 1.2)]:
        frame = _training_frame(120).with_columns(
            pl.lit(season).alias("season"),
            (
                pl.col("projected_home_margin")
                + (pl.col("actual_home_margin") - pl.col("projected_home_margin"))
                * residual_scale
            ).alias("actual_home_margin"),
            (
                pl.col("projected_total")
                + (pl.col("actual_total") - pl.col("projected_total")) * residual_scale
            ).alias("actual_total"),
        )
        frames.append(frame)
    dataset = pl.concat(frames, how="vertical_relaxed")

    result = evaluate_probability_holdout(
        dataset,
        validation_season=2024,
        holdout_season=2025,
        min_training_games=200,
    )

    assert result.training_games == 360
    assert result.validation_games == 120
    assert result.holdout_games == 120
    assert result.margin_scale > 0
    assert result.total_scale > 0



def _heteroscedastic_dataset() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for season in (2022, 2023, 2024, 2025):
        for index in range(180):
            counter += 1
            projected_margin = float((index % 21) - 10)
            projected_total = 39.0 + float(index % 15)
            margin_width = 5.0 if abs(projected_margin) <= 3 else 13.0
            total_width = 7.0 if projected_total <= 46.0 else 15.0
            margin_pattern = float(((counter * 7) % 21) - 10) / 10.0
            total_pattern = float(((counter * 11) % 25) - 12) / 12.0
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 14),
                    "game_id": f"{season}-{index}",
                    "projected_home_margin": projected_margin,
                    "projected_total": projected_total,
                    "actual_home_margin": (
                        projected_margin + margin_pattern * margin_width
                    ),
                    "actual_total": (
                        projected_total + total_pattern * total_width
                    ),
                }
            )
    return pl.DataFrame(rows)


def test_conditional_distribution_changes_uncertainty_by_matchup() -> None:
    dataset = _heteroscedastic_dataset().filter(pl.col("season") <= 2023)
    model = ConditionalStudentTScoreDistribution(
        margin_strength=1.0,
        total_strength=1.0,
        margin_df=8.0,
        total_df=8.0,
    ).fit(dataset)

    close_margin_sigma = model.margin_sigma_for(1.0, 44.0)
    large_margin_sigma = model.margin_sigma_for(10.0, 44.0)
    low_total_sigma = model.total_sigma_for(40.0, 1.0)
    high_total_sigma = model.total_sigma_for(52.0, 1.0)

    assert large_margin_sigma > close_margin_sigma
    assert high_total_sigma > low_total_sigma


def test_conditional_selection_never_uses_holdout_outcomes() -> None:
    dataset = _heteroscedastic_dataset()
    first = evaluate_conditional_probability_holdout(
        dataset,
        validation_season=2024,
        holdout_season=2025,
        min_training_games=300,
    )
    mutated = dataset.with_columns(
        pl.when(pl.col("season") == 2025)
        .then(pl.col("actual_home_margin") + 35.0)
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin"),
        pl.when(pl.col("season") == 2025)
        .then(pl.col("actual_total") + 40.0)
        .otherwise(pl.col("actual_total"))
        .alias("actual_total"),
    )
    second = evaluate_conditional_probability_holdout(
        mutated,
        validation_season=2024,
        holdout_season=2025,
        min_training_games=300,
    )

    assert first.margin_scale == second.margin_scale
    assert first.total_scale == second.total_scale
    assert first.margin_df == second.margin_df
    assert first.total_df == second.total_df
    assert first.margin_strength == second.margin_strength
    assert first.total_strength == second.total_strength
    assert first.holdout_candidate.margin_nll != (
        second.holdout_candidate.margin_nll
    )


def test_student_t_probabilities_remain_monotonic_and_complementary() -> None:
    model = ConditionalStudentTScoreDistribution(
        margin_df=6.0,
        total_df=6.0,
        margin_strength=0.5,
        total_strength=0.5,
    ).fit(_heteroscedastic_dataset().filter(pl.col("season") <= 2023))

    underdog = model.home_win_probability(-7.0, 45.0)
    favorite = model.home_win_probability(7.0, 45.0)
    home_cover = model.home_cover_probability(4.0, -3.5, 47.0)

    assert 0.0 < underdog < favorite < 1.0
    assert 0.0 < home_cover < 1.0
