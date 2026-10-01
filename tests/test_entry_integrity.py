from __future__ import annotations

import json

import polars as pl

from nfl.backtest_audit import audit_backtest_bets
from nfl.entry_integrity import annotate_historical_entry_integrity
from nfl.proof import build_evidence_report
from run_free_market_backtest import _initial_line_coverage


def _annotated_three_market_rows() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [5, 5, 5],
            "game_id": ["g-ml", "g-spread", "g-total"],
            "market_type": ["moneyline", "spread", "total"],
            "side": ["home", "home", "over"],
            "american_odds": [-110, -110, -110],
            "price_stage": ["archive_open_line_final_price"] * 3,
            "has_distinct_open": [True, True, True],
            "opening_book": ["book-a", "book-a", "book-a"],
            "probability_edge": [0.05, 0.05, 0.05],
            "expected_value_per_unit": [0.04, 0.04, 0.04],
            "result": ["win", "win", "win"],
            "net_units": [0.9, 0.9, 0.9],
            "clv_proxy": [0.02, 0.5, 1.0],
        }
    )
    return annotate_historical_entry_integrity(base)


def test_entry_integrity_separates_opening_line_from_opening_price() -> None:
    frame = _annotated_three_market_rows()

    assert frame.get_column("entry_line_observed").to_list() == [True, True, True]
    assert frame.get_column("entry_price_verified").to_list() == [True, False, False]
    assert frame.get_column("entry_price_stage").to_list() == [
        "archive_open_price",
        "archive_open_line_final_price",
        "archive_open_line_final_price",
    ]


def test_backtest_audit_counts_only_observed_entry_prices_as_verified() -> None:
    report = audit_backtest_bets(_annotated_three_market_rows())

    assert report["errors"] == 0
    integrity = report["quote_integrity"]
    assert integrity["opening_line_observed_bets"] == 3
    assert integrity["opening_price_verified_bets"] == 1
    assert integrity["verified_opening_entry_bets"] == 1
    assert integrity["research_only_entry_price_bets"] == 2


def test_initial_line_coverage_reports_non_overlapping_source_window() -> None:
    source = pl.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "type": ["SPREAD", "TOTAL", "SPREAD"],
        }
    )

    coverage = _initial_line_coverage(source, start_season=2022, end_season=2025)

    assert coverage["available"] is True
    assert coverage["seasons"] == [2021]
    assert coverage["market_types"] == ["SPREAD", "TOTAL"]
    assert coverage["backtest_window_rows"] == 0
    assert coverage["overlaps_backtest_window"] is False


def test_proof_uses_verified_price_subset_for_promotion(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(60):
        rows.append(
            {
                "season": 2024,
                "week": 5 + (index % 10),
                "game_id": f"ml-{index}",
                "market_type": "moneyline",
                "side": "home",
                "probability_edge": 0.05,
                "expected_value_per_unit": 0.04,
                "result": "win",
                "net_units": 0.9,
                "clv_proxy": 0.02,
                "has_distinct_open": True,
                "entry_line_observed": True,
                "entry_price_verified": True,
            }
        )
        rows.append(
            {
                "season": 2024,
                "week": 5 + (index % 10),
                "game_id": f"spread-{index}",
                "market_type": "spread",
                "side": "home",
                "probability_edge": 0.05,
                "expected_value_per_unit": 0.04,
                "result": "win",
                "net_units": 0.9,
                "clv_proxy": 0.5,
                "has_distinct_open": True,
                "entry_line_observed": True,
                "entry_price_verified": False,
            }
        )
    bets = pl.DataFrame(rows)
    bets_path = tmp_path / "bets.csv"
    bets.write_csv(bets_path)

    backtest = {
        "overall": {
            "bets": 120,
            "roi_per_unit_staked": 0.9,
            "roi_ci_95_low": 0.8,
            "roi_ci_95_high": 1.0,
        },
        "by_market": {
            "moneyline": {"bets": 60, "roi_per_unit_staked": 0.9},
            "spread": {"bets": 60, "roi_per_unit_staked": 0.9},
        },
        "by_season": {"2024": {"bets": 120, "roi_per_unit_staked": 0.9}},
        "source": {"historical_market": "fixture"},
    }
    backtest_path = tmp_path / "backtest.json"
    backtest_path.write_text(json.dumps(backtest), encoding="utf-8")

    report = build_evidence_report(
        backtest_path=backtest_path,
        bets_path=bets_path,
    )

    promotion = report["promotion_sample"]
    assert promotion["verified_bets"] == 60
    assert promotion["opening_line_observed_bets"] == 120
    assert promotion["excluded_unverified_bets"] == 60
    assert promotion["positive_markets"] == 1
    assert set(promotion["by_market"]) == {"moneyline"}
    assert report["status"] != "ROBUST"


def test_legacy_distinct_open_flag_cannot_verify_entry_price(tmp_path) -> None:
    bets = pl.DataFrame(
        {
            "season": [2024],
            "week": [5],
            "game_id": ["legacy"],
            "market_type": ["moneyline"],
            "side": ["home"],
            "probability_edge": [0.05],
            "expected_value_per_unit": [0.04],
            "result": ["win"],
            "net_units": [0.9],
            "clv_proxy": [0.02],
            "has_distinct_open": [True],
        }
    )
    bets_path = tmp_path / "legacy.csv"
    bets.write_csv(bets_path)
    backtest_path = tmp_path / "backtest.json"
    backtest_path.write_text(json.dumps({"overall": {"bets": 1}}), encoding="utf-8")

    report = build_evidence_report(
        backtest_path=backtest_path,
        bets_path=bets_path,
    )

    assert report["promotion_sample"]["verified_bets"] == 0
    assert report["promotion_sample"]["entry_price_verified"] is False
