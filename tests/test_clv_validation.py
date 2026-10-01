import math
from datetime import UTC, datetime, timedelta

import polars as pl

from nfl.clv import attach_closing_line_value, summarize_clv
from nfl.market_validation import (
    evaluate_market_rule_holdout,
    select_best_market_opportunities,
)


def _decision_comparisons() -> pl.DataFrame:
    captured = datetime(2025, 10, 5, 12, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "market_type": ["spread", "spread"],
            "side": ["home", "away"],
            "line": [-3.5, 3.5],
            "american_odds": [-110, -110],
            "no_vig_probability": [0.5, 0.5],
            "probability_edge": [0.08, -0.08],
            "expected_value_per_unit": [0.10, -0.10],
            "provider": ["the_odds_api", "the_odds_api"],
            "book": ["book-a", "book-a"],
            "captured_at": [captured, captured],
            "snapshot_id": ["decision", "decision"],
        }
    )


def _closing_quotes() -> pl.DataFrame:
    captured = datetime(2025, 10, 5, 16, 55, tzinfo=UTC)
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "market_type": ["spread", "spread"],
            "side": ["home", "away"],
            "line": [-4.5, 4.5],
            "american_odds": [-120, 100],
            "provider": ["the_odds_api", "the_odds_api"],
            "book": ["book-a", "book-a"],
            "captured_at": [captured, captured],
            "snapshot_id": ["close", "close"],
            "source_event_id": ["event-1", "event-1"],
        }
    )


def test_clv_uses_same_book_side_and_later_snapshot() -> None:
    attached = attach_closing_line_value(_decision_comparisons(), _closing_quotes())
    home = attached.filter(pl.col("side") == "home").row(0, named=True)
    away = attached.filter(pl.col("side") == "away").row(0, named=True)

    assert math.isclose(home["line_clv"], 1.0)
    assert math.isclose(away["line_clv"], -1.0)
    assert home["probability_clv"] > 0
    assert away["probability_clv"] < 0
    summary = summarize_clv(attached)
    assert summary.observations == 2
    assert math.isclose(summary.average_spread_line_clv or 0.0, 0.0)


def _graded_rows() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        for index in range(4):
            game_id = f"{season}-g{index}"
            for book, ev_bonus in (("book-a", 0.0), ("book-b", 0.01)):
                rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "market_type": "spread",
                        "side": "home",
                        "book": book,
                        "american_odds": -110,
                        "probability_edge": 0.05 + ev_bonus,
                        "expected_value_per_unit": 0.04 + ev_bonus,
                        "result": "win" if index != 3 else "loss",
                        "net_units": (100.0 / 110.0) if index != 3 else -1.0,
                        "probability_clv": 0.01 + ev_bonus,
                    }
                )
    return pl.DataFrame(rows)


def test_best_opportunity_counts_one_quote_per_game_market() -> None:
    best = select_best_market_opportunities(_graded_rows())

    assert best.height == 8
    assert set(best.get_column("book").to_list()) == {"book-b"}


def test_rule_is_tuned_on_validation_then_frozen_for_holdout() -> None:
    evaluation = evaluate_market_rule_holdout(
        _graded_rows(),
        market_type="spread",
        validation_season=2024,
        holdout_season=2025,
        probability_edge_grid=(0.04, 0.05),
        expected_value_grid=(0.03, 0.04),
        min_validation_bets=3,
        min_holdout_bets=3,
    )

    assert evaluation.rule is not None
    assert evaluation.validation.bets == 4
    assert evaluation.holdout.bets == 4
    assert evaluation.validation.net_units > 0
    assert evaluation.holdout.net_units > 0
    assert evaluation.holdout.average_probability_clv > 0
    assert evaluation.candidate_pass


def test_closing_timestamp_can_be_later_than_decision() -> None:
    frame = _closing_quotes().with_columns(
        (pl.col("captured_at") + pl.duration(minutes=5)).alias("captured_at")
    )
    attached = attach_closing_line_value(_decision_comparisons(), frame)

    assert attached.get_column("closing_captured_at").min() == datetime(
        2025, 10, 5, 17, 0, tzinfo=UTC
    )
    assert attached.get_column("closing_captured_at").min() > (
        datetime(2025, 10, 5, 12, 0, tzinfo=UTC) + timedelta(0)
    )
