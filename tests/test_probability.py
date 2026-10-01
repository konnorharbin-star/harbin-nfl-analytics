import math

import polars as pl

from nfl.probability import GaussianScoreDistribution, evaluate_probability_holdout


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
