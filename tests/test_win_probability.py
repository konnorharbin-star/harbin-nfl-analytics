import math

import polars as pl

from nfl.win_probability import LogisticWinModel, evaluate_win_probability_holdout


def _frame(rows: int = 160, season: int | None = None) -> pl.DataFrame:
    margins = [float((index % 17) - 8) for index in range(rows)]
    actual = [
        margin + float(((index * 5) % 19) - 9)
        for index, margin in enumerate(margins)
    ]
    data: dict[str, list[float] | list[int]] = {
        "projected_home_margin": margins,
        "projected_total": [45.0 + float(index % 4) for index in range(rows)],
        "actual_home_margin": actual,
        "actual_total": [44.0 + float((index * 3) % 13) for index in range(rows)],
    }
    if season is not None:
        data["season"] = [season] * rows
    return pl.DataFrame(data)


def test_logistic_win_probability_is_monotonic() -> None:
    model = LogisticWinModel(alpha=1.0).fit(_frame())
    values = [model.predict_probability(margin) for margin in (-10.0, 0.0, 10.0)]

    assert 0.0 < values[0] < values[1] < values[2] < 1.0


def test_logistic_output_column_is_finite() -> None:
    model = LogisticWinModel(alpha=0.1).fit(_frame())
    output = model.add_probability_column(
        pl.DataFrame({"projected_home_margin": [-7.0, 0.0, 7.0]})
    )

    assert all(
        math.isfinite(value) and 0.0 < value < 1.0
        for value in output.get_column("home_win_probability").to_list()
    )


def test_nested_win_holdout_preserves_season_boundary() -> None:
    dataset = pl.concat(
        [
            _frame(160, 2022),
            _frame(160, 2023),
            _frame(160, 2024),
            _frame(160, 2025),
        ],
        how="vertical_relaxed",
    )
    result = evaluate_win_probability_holdout(
        dataset,
        validation_season=2024,
        holdout_season=2025,
        min_training_games=300,
    )

    assert result.validation_games == 160
    assert result.holdout_games == 160
    assert result.training_games <= 480
    assert result.alpha >= 0
    assert 0.0 <= result.logistic.brier <= 1.0
    assert result.logistic.log_loss > 0
