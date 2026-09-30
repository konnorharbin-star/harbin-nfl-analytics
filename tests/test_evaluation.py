import polars as pl
import pytest

from nfl.evaluation import summarize_baseline


def test_summarize_baseline_metrics() -> None:
    dataset = pl.DataFrame(
        {
            "actual_home_margin": [3.0, -7.0],
            "baseline_home_margin": [1.0, -3.0],
            "actual_total": [44.0, 50.0],
            "baseline_total": [46.0, 46.0],
        }
    )

    metrics = summarize_baseline(dataset)

    assert metrics.games == 2
    assert metrics.margin_mae == pytest.approx(3.0)
    assert metrics.margin_rmse == pytest.approx((10.0) ** 0.5)
    assert metrics.margin_bias == pytest.approx(1.0)
    assert metrics.total_mae == pytest.approx(3.0)
    assert metrics.total_rmse == pytest.approx((10.0) ** 0.5)
    assert metrics.total_bias == pytest.approx(-1.0)
