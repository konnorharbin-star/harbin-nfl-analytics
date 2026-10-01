import polars as pl

from nfl.free_market import FreeNFLMarketStore
from nfl.free_market_backtest import (
    build_free_archive_bets,
    evaluate_archive_holdout,
)


def _target_schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2023_02_BUF_KC"],
            "season": [2023],
            "week": [2],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "away_moneyline": [135],
            "home_moneyline": [-155],
            "spread_line": [3.0],
            "away_spread_odds": [-105],
            "home_spread_odds": [-115],
            "total_line": [47.5],
            "under_odds": [-110],
            "over_odds": [-110],
        }
    )


def _projection_dataset() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(65):
        projected_margin = float((index % 11) - 5)
        projected_total = 42.0 + float(index % 8)
        rows.append(
            {
                "season": 2023,
                "week": 1,
                "game_id": f"history-{index}",
                "home_team": "AAA",
                "away_team": "BBB",
                "projected_home_margin": projected_margin,
                "projected_total": projected_total,
                "actual_home_margin": projected_margin + float((index % 9) - 4),
                "actual_total": projected_total + float((index % 13) - 6),
            }
        )
    rows.append(
        {
            "season": 2023,
            "week": 2,
            "game_id": "2023_02_BUF_KC",
            "home_team": "KC",
            "away_team": "BUF",
            "projected_home_margin": 5.5,
            "projected_total": 46.0,
            "actual_home_margin": 7.0,
            "actual_total": 45.0,
        }
    )
    return pl.DataFrame(rows)


def test_free_archive_backtest_emits_one_side_per_market() -> None:
    store = FreeNFLMarketStore(_target_schedule())

    bets = build_free_archive_bets(_projection_dataset(), store)

    assert bets.height == 3
    assert set(bets.get_column("market_type").to_list()) == {
        "moneyline",
        "spread",
        "total",
    }
    assert bets.get_column("price_stage").unique().to_list() == [
        "archive_final_fallback"
    ]
    assert bets.get_column("clv_proxy").null_count() == 3


def test_archive_rule_is_selected_on_validation_then_scored_on_holdout() -> None:
    rows = []
    for season in (2024, 2025):
        for index in range(30):
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 10),
                    "game_id": f"{season}-{index}",
                    "market_type": "spread",
                    "side": "home",
                    "probability_edge": 0.05,
                    "expected_value_per_unit": 0.04,
                    "result": "win",
                    "net_units": 0.9,
                    "clv_proxy": 0.5,
                }
            )
    bets = pl.DataFrame(rows)

    evaluation = evaluate_archive_holdout(
        bets,
        market_type="spread",
        validation_season=2024,
        holdout_season=2025,
        min_validation_bets=25,
        min_holdout_bets=25,
    )

    assert evaluation.rule is not None
    assert evaluation.validation.bets >= 25
    assert evaluation.holdout.bets >= 25
    assert evaluation.candidate_pass is True
