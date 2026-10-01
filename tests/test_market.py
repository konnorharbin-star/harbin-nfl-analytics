import math

import polars as pl

from nfl.market import (
    MarketQuote,
    american_implied_probability,
    american_to_decimal,
    compare_two_way_market,
    expected_value_per_unit,
    remove_two_way_vig,
)
from nfl.probability import GaussianScoreDistribution


def _distribution() -> GaussianScoreDistribution:
    rows = 120
    margin = [float((index % 11) - 5) for index in range(rows)]
    total = [44.0 + float(index % 5) for index in range(rows)]
    frame = pl.DataFrame(
        {
            "projected_home_margin": margin,
            "projected_total": total,
            "actual_home_margin": [
                margin[index] + float(((index * 7) % 23) - 11) for index in range(rows)
            ],
            "actual_total": [
                total[index] + float(((index * 5) % 27) - 13) for index in range(rows)
            ],
        }
    )
    return GaussianScoreDistribution().fit(frame)


def test_american_odds_conversions() -> None:
    assert math.isclose(american_to_decimal(-110), 1.0 + 100.0 / 110.0)
    assert math.isclose(american_to_decimal(150), 2.5)
    assert math.isclose(american_implied_probability(100), 0.5)


def test_two_way_no_vig_probabilities_sum_to_one() -> None:
    first, second = remove_two_way_vig(-110, -110)

    assert math.isclose(first, 0.5)
    assert math.isclose(second, 0.5)
    assert math.isclose(first + second, 1.0)


def test_expected_value_uses_executable_price() -> None:
    assert expected_value_per_unit(0.55, -110) > 0
    assert expected_value_per_unit(0.50, -110) < 0


def test_moneyline_comparison_preserves_market_separation() -> None:
    distribution = _distribution()
    home, away = compare_two_way_market(
        distribution,
        projected_home_margin=3.0,
        projected_total=45.0,
        first=MarketQuote("moneyline", "home", -150, book="test"),
        second=MarketQuote("moneyline", "away", 130, book="test"),
    )

    assert math.isclose(home.no_vig_probability + away.no_vig_probability, 1.0)
    assert math.isclose(home.model_probability + away.model_probability, 1.0)
    assert home.model_probability > away.model_probability


def test_spread_probabilities_are_complements_for_opposite_lines() -> None:
    distribution = _distribution()
    home, away = compare_two_way_market(
        distribution,
        projected_home_margin=4.0,
        projected_total=47.0,
        first=MarketQuote("spread", "home", -110, line=-3.5),
        second=MarketQuote("spread", "away", -110, line=3.5),
    )

    assert math.isclose(home.model_probability + away.model_probability, 1.0, abs_tol=1e-12)


def test_total_probabilities_are_complements() -> None:
    distribution = _distribution()
    over, under = compare_two_way_market(
        distribution,
        projected_home_margin=0.0,
        projected_total=46.0,
        first=MarketQuote("total", "over", -105, line=45.5),
        second=MarketQuote("total", "under", -115, line=45.5),
    )

    assert math.isclose(over.model_probability + under.model_probability, 1.0, abs_tol=1e-12)
