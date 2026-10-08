"""NFL historical edge failure: provenance, calibration, paired CI, no promotion."""
from __future__ import annotations

import csv
import json

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.historical_edge_diagnostics import (
    _ci,
    _edge_band,
    _graded_records,
    _probability_band,
    build_historical_edge_failure_report,
    write_historical_edge_failure,
)


def sample(
    game="2024_01_A_B", season=2024, week=1, market="spread",
    p=0.70, market_p=0.50, won=False, result=None, decimal=1.91, **overrides
):
    actual = result or ("win" if won else "loss")
    profit = decimal - 1 if actual == "win" else -1 if actual == "loss" else 0
    row = {
        "game_id": game,
        "season": season, "week": week, "market_type": market,
        "side": "home", "book": "ESPN BET",
        "model_probability": p, "no_vig_probability": market_p,
        "decimal_odds": decimal, "probability_edge": p - market_p,
        "expected_value_per_unit": p * decimal - 1,
        "result": actual, "net_units": profit,
        "entry_price_verified": True, "entry_quote_verified": True,
        "entry_timestamp_verified": False, "entry_price_stage": "espn_archived_open",
    }
    row.update(overrides)
    return row


def evidence():
    archive = {
        "entry_provenance": "provider_labeled_open",
        "timestamped_entry_prices": False,
        "status": "READY",
    }
    shrinkage = {
        "status": "RESEARCH_ONLY", "holdout_season": 2025,
        "markets": {
            market: {"alpha": 0, "status": "MARKET_ONLY_PREFERRED"}
            for market in ("moneyline", "spread", "total")
        },
    }
    regime = {
        "holdout_season": 2025,
        "summary": {"market_status": {
            market: "UNRELIABLE" for market in ("moneyline", "spread", "total")
        }},
    }
    return archive, shrinkage, regime


def report_for(rows, sources=None):
    archive, shrinkage, regime = sources if sources is not None else evidence()
    return build_historical_edge_failure_report(
        pl.DataFrame(rows), archive_backtest=archive,
        market_shrinkage=shrinkage, regime_reliability=regime,
    )


def test_positive_modeled_ev_but_loss_reported_as_archive_failure():
    row = sample()
    result = report_for([row])
    assert result["status"] == "ARCHIVE_DIAGNOSTICS_NO_EXECUTABLE_ENTRY_PROOF"
    assert result["overall"]["rows"] == 1
    assert result["overall"]["mean_model_probability"] == 0.70
    assert result["overall"]["mean_no_vig_probability"] == 0.50
    assert result["overall"]["observed_win_rate"] == 0.0
    assert result["overall"]["archive_roi_per_unit"] == -1
    assert result["overall"]["mean_expected_value_per_unit"] == pytest.approx(0.337)
    assert result["overall"]["ev_realization_gap"] == pytest.approx(1.337)
    assert result["overall"]["model_minus_market_brier"] > 0
    assert result["overall"]["ci_95_model_minus_market_brier"] is None
    flags = result["by_market_failure"]["spread"]["failure_flags"]
    assert "MODEL_PROBABILITY_OVERCONFIDENCE" in flags
    assert "POSITIVE_RAW_EV_BUT_NEGATIVE_ARCHIVE_RETURN" in flags
    assert "NO_POSITIVE_HOLDOUT_SELECTED_MODEL_WEIGHT" in flags
    assert not result["betting_authorized"]
    assert result["approved_units"] == 0
    assert not result["historical_subgroup_promotion_allowed"]


def test_win_loss_push_reconciled_and_push_excluded_from_brier():
    inputs = [
        sample("game1", won=True),
        sample("game2", won=False),
        sample("game3", result="push"),
    ]
    result = report_for(inputs)
    group = result["overall"]
    assert group["rows"] == 3
    assert group["decided_rows"] == 2
    assert group["observed_win_rate"] == 0.5
    assert group["pushes"] == 1
    assert group["archive_roi_per_unit"] == pytest.approx((0.91 - 1) / 3)
    assert group["mean_market_only_ev"] == pytest.approx(0.5 * 1.91 - 1)
    assert group["model_calibration_gap"] == pytest.approx(0.2)


def test_2024_2025_split_and_positive_rule_are_fixed():
    records = [
        sample("g24", season=2024, p=.7),
        sample("g25", season=2025, p=.7),
        sample("g25_2", season=2025, p=.50, market_p=.51, won=True),
    ]
    result = report_for(records)
    assert result["data_integrity"]["included_rows"] == 3
    assert result["fixed_splits"]["validation_season"] == 2024
    assert result["fixed_splits"]["diagnostic_holdout_season"] == 2025
    groups = result["by_segment"]
    raw_2025 = next(
        r for r in groups
        if r["cohort"] == "raw_positive_season_market"
        and r["market"] == "spread" and r["season"] == 2025
    )
    assert raw_2025["rows"] == 1
    assert result["previously_accessed_2025_outcomes_not_untouched_holdout"]
    assert not result["policy_thresholds_changed"]


def test_exact_duplicate_game_market_season_aborts_instead_of_double_counting():
    row = sample()
    with pytest.raises(DataContractError, match="duplicate"):
        report_for([row, row])


def test_invalid_payoff_not_silently_included_in_profitable_archive():
    row = sample(net_units=50)
    result = report_for([row])
    assert result["status"] == "NO_VALID_HISTORICAL_EVIDENCE"
    assert result["data_integrity"]["excluded_reasons"] == {
        "net_units_not_reconciled_to_price_and_result": 1
    }


def test_discordant_model_ev_and_edge_do_not_fake_positive_signal():
    rows = [
        sample("edge_bad", probability_edge=0.6),
        sample("ev_bad", expected_value_per_unit=0.8),
    ]
    result = report_for(rows)
    reasons = result["data_integrity"]["excluded_reasons"]
    assert reasons["raw_edge_not_reconciled_to_probabilities"] == 1
    assert reasons["raw_ev_not_reconciled_to_probabilities_and_price"] == 1
    assert result["overall"]["rows"] == 0


def test_bad_probabilities_prices_and_outcomes_fail_closed():
    for changes in (
        {"model_probability": 1.2},
        {"no_vig_probability": -0.1},
        {"decimal_odds": 0.8},
        {"result": "pending"},
        {"entry_price_verified": False},
        {"entry_quote_verified": False},
    ):
        frame, excluded = _graded_records(pl.DataFrame([sample(**changes)]))
        assert len(frame) == 0
        assert sum(excluded.values()) == 1


def test_missing_or_wrong_source_reports_never_authorize_change():
    frame = [sample()]
    archive, shrinkage, regime = evidence()
    assert report_for(frame, sources=(None, shrinkage, regime))[
        "status"
    ] == "EVIDENCE_INCOMPLETE_OR_INVALID_FAIL_CLOSED"
    shrinkage["holdout_season"] = 2024
    assert report_for(frame, sources=(archive, shrinkage, regime))[
        "status"
    ] == "EVIDENCE_INCOMPLETE_OR_INVALID_FAIL_CLOSED"


def test_missing_archive_timestamp_even_when_boolean_row_says_true():
    archive, shrinkage, regime = evidence()
    result = report_for([
        sample(entry_timestamp_verified=True),
    ], sources=(archive, shrinkage, regime))
    assert result["status"] == "ARCHIVE_DIAGNOSTICS_NO_EXECUTABLE_ENTRY_PROOF"


def test_predefined_edge_buckets_are_not_an_optimized_policy():
    assert _edge_band(-.05) == "nonpositive"
    assert _edge_band(.019) == "(0,2pp)"
    assert _edge_band(.021) == "[2,5pp)"
    assert _edge_band(.05) == "[5,10pp)"
    assert _edge_band(.11) == "[10,20pp)"
    assert _edge_band(.2) == "20pp_plus"
    assert _probability_band(.49) == "below_50"
    assert _probability_band(.61) == "[60,65)"


def test_week_block_brier_ci_needs_100_rows_60_games_8_weeks_and_is_repeatable():
    rows = [
        sample(
            game=f"2024_{1 + i // 10:02d}_game{i}",
            season=2024,
            week=1 + i // 10,
            p=.7, won=i % 2 == 0,
        )
        for i in range(120)
    ]
    records, errors = _graded_records(pl.DataFrame(rows))
    assert not errors
    for row in records:
        row["brier_delta"] = row["model_brier_loss"] - row["market_brier_loss"]
    first = _ci(records, "brier_delta", seed=111)
    assert first == _ci(records, "brier_delta", seed=111)
    assert first[0] > 0
    assert _ci(records[:99], "brier_delta", seed=111) is None
    assert _ci(records[:70], "brier_delta", seed=111) is None
    # 100 different tickets in just 7 kickoff weeks is still not enough.
    close_weeks = [dict(r, week=r["week"] % 7 + 1) for r in records]
    assert _ci(close_weeks, "brier_delta", seed=111) is None
    diagnosis = report_for(rows)
    groups = diagnosis["by_segment"]
    overall = next(g for g in groups if g["cohort"] == "all")
    assert overall["ci_95_model_minus_market_brier"] is not None
    assert overall["model_minus_market_brier"] > 0


def test_empty_cohorts_and_no_retroactive_promotion():
    result = report_for([sample()])
    for item in result["by_segment"]:
        if item["rows"] == 0:
            assert item["archive_roi_per_unit"] is None
            assert item["ci_95_archive_roi"] is None
    assert all(
        item["evidence_status"] != "PROMOTED" for item in result["by_segment"]
    )


def test_outputs_are_reproducible_and_empty_csv_header_kept(tmp_path):
    result = report_for([sample()])
    write_historical_edge_failure(
        result,
        report_dir=tmp_path / "reports",
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
    )
    for filename in ("historical_edge_failure.json", "historical_edge_segments.csv"):
        expected = (tmp_path / "reports" / filename).read_bytes()
        assert (tmp_path / "outputs" / filename).read_bytes() == expected
        assert (tmp_path / "docs" / filename).read_bytes() == expected
    with (tmp_path / "docs" / "historical_edge_segments.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[0]["cohort"] == "all"
    assert json.loads(
        (tmp_path / "outputs" / "historical_edge_failure.json").read_text()
    )["betting_authorized"] is False
