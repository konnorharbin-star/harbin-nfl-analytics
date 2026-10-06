"""Sportsbook market comparison utilities kept downstream of the football model.

This module accepts market prices only after the independent football distribution has
already been produced. Prices may be converted to implied/no-vig probabilities and
compared against model probabilities, but they are never fed back into fair-score
ratings or football feature construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

from .probability_runtime import ScoreDistribution

MarketType = Literal["moneyline", "spread", "total"]
MarketSide = Literal["home", "away", "over", "under"]


@dataclass(frozen=True)
class MarketQuote:
    market_type: MarketType
    side: MarketSide
    american_odds: int
    line: float | None = None
    book: str | None = None
    captured_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.american_odds == 0 or -100 < self.american_odds < 100:
            raise ValueError("American odds must be <= -100 or >= +100")
        if self.market_type == "moneyline" and self.line is not None:
            raise ValueError("moneyline quote must not include a line")
        if self.market_type in {"spread", "total"} and self.line is None:
            raise ValueError(f"{self.market_type} quote requires a line")
        allowed = {
            "moneyline": {"home", "away"},
            "spread": {"home", "away"},
            "total": {"over", "under"},
        }
        if self.side not in allowed[self.market_type]:
            raise ValueError(f"invalid side {self.side!r} for {self.market_type}")


@dataclass(frozen=True)
class MarketComparison:
    market_type: MarketType
    side: MarketSide
    line: float | None
    american_odds: int
    decimal_odds: float
    model_probability: float
    market_implied_probability: float
    no_vig_probability: float
    probability_edge: float
    expected_value_per_unit: float
    book: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def american_to_decimal(american_odds: int) -> float:
    """Convert standard American odds to decimal odds."""

    if american_odds == 0 or -100 < american_odds < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def american_implied_probability(american_odds: int) -> float:
    return 1.0 / american_to_decimal(american_odds)


def remove_two_way_vig(
    first_american_odds: int,
    second_american_odds: int,
) -> tuple[float, float]:
    """Return proportional two-way no-vig probabilities that sum to one."""

    first = american_implied_probability(first_american_odds)
    second = american_implied_probability(second_american_odds)
    total = first + second
    if total <= 0:
        raise ValueError("two-way implied probability sum must be positive")
    return first / total, second / total


def expected_value_per_unit(model_probability: float, american_odds: int) -> float:
    """Expected net profit per unit staked at the quoted price."""

    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be in [0, 1]")
    decimal = american_to_decimal(american_odds)
    return model_probability * decimal - 1.0


def _validate_pair(first: MarketQuote, second: MarketQuote) -> None:
    if first.market_type != second.market_type:
        raise ValueError("paired quotes must have the same market_type")
    if first.side == second.side:
        raise ValueError("paired quotes must represent opposite sides")
    if first.market_type in {"spread", "total"}:
        assert first.line is not None and second.line is not None
        if first.market_type == "total" and first.line != second.line:
            raise ValueError("over/under quotes must use the same total line")
        if first.market_type == "spread" and abs(first.line + second.line) > 1e-9:
            raise ValueError("home/away spread lines must be exact opposites")


def model_probability_for_quote(
    distribution: ScoreDistribution,
    *,
    projected_home_margin: float,
    projected_total: float,
    quote: MarketQuote,
) -> float:
    """Evaluate one quote against an already-fitted football distribution."""

    if quote.market_type == "moneyline":
        home = distribution.home_win_probability(
            projected_home_margin,
            projected_total,
        )
        return home if quote.side == "home" else 1.0 - home

    if quote.market_type == "spread":
        assert quote.line is not None
        if quote.side == "home":
            return distribution.home_cover_probability(
                projected_home_margin,
                quote.line,
                projected_total,
            )
        # Away +x is the complement of home -x under a continuous residual
        # distribution. Calling the distribution method preserves conditional
        # uncertainty instead of reaching into a global sigma.
        home = distribution.home_cover_probability(
            projected_home_margin,
            -quote.line,
            projected_total,
        )
        return 1.0 - home

    assert quote.market_type == "total"
    assert quote.line is not None
    over = distribution.over_probability(
        projected_total,
        quote.line,
        projected_home_margin,
    )
    return over if quote.side == "over" else 1.0 - over


def compare_two_way_market(
    distribution: ScoreDistribution,
    *,
    projected_home_margin: float,
    projected_total: float,
    first: MarketQuote,
    second: MarketQuote,
) -> tuple[MarketComparison, MarketComparison]:
    """Compare both sides of one two-way market to the football distribution."""

    _validate_pair(first, second)
    first_no_vig, second_no_vig = remove_two_way_vig(
        first.american_odds,
        second.american_odds,
    )
    output: list[MarketComparison] = []
    for quote, no_vig in ((first, first_no_vig), (second, second_no_vig)):
        model_probability = model_probability_for_quote(
            distribution,
            projected_home_margin=projected_home_margin,
            projected_total=projected_total,
            quote=quote,
        )
        implied = american_implied_probability(quote.american_odds)
        output.append(
            MarketComparison(
                market_type=quote.market_type,
                side=quote.side,
                line=quote.line,
                american_odds=quote.american_odds,
                decimal_odds=american_to_decimal(quote.american_odds),
                model_probability=model_probability,
                market_implied_probability=implied,
                no_vig_probability=no_vig,
                probability_edge=model_probability - no_vig,
                expected_value_per_unit=expected_value_per_unit(
                    model_probability,
                    quote.american_odds,
                ),
                book=quote.book,
            )
        )
    return output[0], output[1]
