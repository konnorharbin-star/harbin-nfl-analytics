"""No-hindsight tests for the frozen NFL market-only and flat-paper baselines."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl.forward_edge_validation import (
    append_forward_candidates,
    grade_forward_candidates,
    load_forward_candidates,
)
from nfl.market_benchmarks import (
    FORECAST_WEIGHTS,
    MIN_DECIDED,
    PAPER_RULES,
    _week_ci,
    build_market_benchmarks,
    write_market_benchmarks,
)

CAPTURE = datetime(2026, 10, 7, 19, tzinfo=UTC)


def current(
    game: str, week: int, *, market: str = "spread",
    model: float = 0.65, book: float = 0.50, odds: int = -110,
    source_verified: bool = True,
) -> dict[str, object]:
    price = None if market == "moneyline" else (
        44.5 if market == "total" else -3.5
    )
    return {
        "season": 2026, "week": week, "game_id": game,
        "home_team": "WAS", "away_team": "NYG",
        "kickoff": (CAPTURE + timedelta(days=4 + (week - 5) * 7)).isoformat(),
        "quant_market": market,
        "quant_side": "over" if market == "total" else "home",
        "quant_book": "DraftKings", "quant_price": price,
        "quant_odds": odds,
        "quant_quote_at": (CAPTURE - timedelta(minutes=3)).isoformat(),
        "quant_probability": model, "quant_market_probability": book,
        "quant_edge": model - book,
        "quant_ev": model * (1 + 100 / abs(odds)) - 1,
        "market_execution_verified": source_verified,
        "market_quote_timestamp_verified": source_verified,
        "market_quote_sanity_ok": True,
        "execution_ready": True,
        "portfolio_action": "PASS",
        "portfolio_stake_units": 0.0,
    }


def final(game: str, win: bool | None = True) -> dict[str, object]:
    scores = (
        {"home_score": 27.0, "away_score": 17.0} if win else
        {"home_score": 17.0, "away_score": 27.0}
    )
    if win is None:
        scores = {"home_score": None, "away_score": None}
    return {
        "game_id": game, "season": 2026,
        "home_team": "WAS", "away_team": "NYG", **scores,
    }


def fixture(
    tmp_path,
    *,
    games: int = 4, markets: tuple[str, ...] = ("spread",),
    completed_games: int | None = None, p: float = 0.65,
    verified: bool = True,
):
    candidates = []
    results = []
    for i in range(games):
        week = 5 + i // 10
        game = f"2026_{week:02d}_NYG_WAS_{i:04d}"
        for market in markets:
            candidates.append(
                current(game, week, market=market, model=p,
                        source_verified=verified)
            )
        if completed_games is None or i < completed_games:
            results.append(final(game, win=i % 2 == 0))
    ledger_path = tmp_path / "forward.csv"
    append_forward_candidates(
        pl.DataFrame(candidates), captured_at=CAPTURE, path=ledger_path
    )
    ledger = load_forward_candidates(ledger_path)
    schedules = pl.DataFrame(results) if results else pl.DataFrame({
        "game_id": [], "season": [], "home_team": [], "away_team": [],
        "home_score": [], "away_score": [],
    })
    return grade_forward_candidates(ledger, schedules)


def _row(report, predictor, *, market=None, cohort="all"):
    return next(
        row for row in report["predictor_metrics"]
        if row["predictor"] == predictor
        and row["market"] == market and row["cohort"] == cohort
    )


def _paper(report, rule, *, market=None, cohort="all"):
    return next(
        row for row in report["paper_policy_metrics"]
        if row["rule"] == rule
        and row["market"] == market and row["cohort"] == cohort
    )


def test_market_only_frozen_same_cohort_as_blends_and_raw_model(tmp_path):
    graded, source = fixture(tmp_path)
    report = build_market_benchmarks(graded, forward_report=source)
    assert report["status"] == "RESEARCH_SAMPLE_UNDERPOWERED"
    assert list(report["predeclared_forecast_weights"]) == list(FORECAST_WEIGHTS)
    assert report["predeclared_flat_unit_rules"] == list(PAPER_RULES)
    assert report["frozen"] == 4
    assert report["decided"] == 4
    for name in FORECAST_WEIGHTS:
        row = _row(report, name)
        assert row["frozen"] == 4
        assert row["settled"] == 4
        assert row["decided"] == 4
    market = _row(report, "market_only")
    full = _row(report, "raw_model")
    blend = _row(report, "market_plus_50pct_model")
    assert market["brier"] == pytest.approx(0.25)
    assert full["brier"] == pytest.approx((0.35 ** 2 + 0.65 ** 2) / 2)
    assert blend["brier"] == pytest.approx((0.425 ** 2 + 0.575 ** 2) / 2)
    assert market["market_minus_predictor_brier"] == pytest.approx(0.0)
    assert full["market_minus_predictor_brier"] < 0
    assert not report["staking_authorized"]
    assert not report["production_model_change_enabled"]


def test_all_fixed_flat_rules_are_same_opportunity_denominator(tmp_path):
    graded, source = fixture(tmp_path)
    result = build_market_benchmarks(graded, forward_report=source)
    abstain = _paper(result, "no_bet_market_only")
    flat = _paper(result, "same_selected_side_flat")
    positive = _paper(result, "model_ev_positive")
    assert abstain["hypothetical_units"] == 0
    assert abstain["hypothetical_bets"] == 0
    assert abstain["hypothetical_roi_on_bets"] is None
    assert flat["eligible_settled"] == 4
    assert flat["hypothetical_bets"] == 4
    assert flat["wins"] == flat["losses"] == 2
    assert positive["hypothetical_bets"] == 4
    assert positive["hypothetical_units"] == pytest.approx(flat["hypothetical_units"])
    assert flat["hypothetical_units"] == pytest.approx(2 * 100 / 110 - 2)
    assert flat["net_units_per_candidate_ci95"] is None


def test_fixed_thresholds_change_selection_not_predictions(tmp_path):
    graded, source = fixture(tmp_path, p=0.51)
    report = build_market_benchmarks(graded, forward_report=source)
    assert _paper(report, "same_selected_side_flat")["hypothetical_bets"] == 4
    assert _paper(report, "model_ev_positive")["hypothetical_bets"] == 0
    assert _paper(report, "model_edge_2pp_and_positive_ev")["hypothetical_bets"] == 0
    assert _row(report, "market_only")["brier"] == pytest.approx(0.25)
    assert _row(report, "raw_model")["decided"] == 4


def test_pending_quotes_never_impute_scores_or_profit(tmp_path):
    graded, source = fixture(tmp_path, games=10, completed_games=2)
    report = build_market_benchmarks(graded, forward_report=source)
    assert report["frozen"] == 10
    assert report["decided"] == 2
    assert _row(report, "market_only")["eligible_quotes"] == 10
    assert _row(report, "raw_model")["decided"] == 2
    assert _paper(report, "same_selected_side_flat")["eligible_settled"] == 2


def test_bad_or_missing_quote_remains_in_denominator_but_not_accuracy(tmp_path):
    graded, source = fixture(tmp_path, verified=False)
    report = build_market_benchmarks(graded, forward_report=source)
    assert report["status"] == "PENDING_FORWARD"
    assert report["frozen"] == 4
    assert report["decided"] == 0
    assert _row(report, "market_only")["eligible_quotes"] == 0
    assert _paper(report, "same_selected_side_flat")["hypothetical_bets"] == 0
    assert report["integrity"]["source_reconciled"]


def test_duplicate_grades_and_stale_source_summary_fail_closed(tmp_path):
    graded, source = fixture(tmp_path)
    duplicates = pl.concat([graded, graded])
    report = build_market_benchmarks(duplicates, forward_report=source)
    assert report["status"] == "BLOCKED_SOURCE_INTEGRITY"
    assert report["integrity"]["duplicate_keys"]
    assert not report["predictor_metrics"]
    wrong = dict(source)
    wrong["frozen_candidate_rows"] = 100
    other = build_market_benchmarks(graded, forward_report=wrong)
    assert other["status"] == "BLOCKED_SOURCE_INTEGRITY"
    assert not other["paper_policy_metrics"]


def test_no_frozen_source_proof_or_bad_source_permission_fails_closed(tmp_path):
    graded, source = fixture(tmp_path)
    assert build_market_benchmarks(
        graded, forward_report=None
    )["status"] == "BLOCKED_SOURCE_INTEGRITY"
    changed = dict(source, staking_authorized=True)
    assert build_market_benchmarks(
        graded, forward_report=changed
    )["status"] == "BLOCKED_SOURCE_INTEGRITY"


def test_reconciled_grade_math_is_required_even_when_source_summary_matches(tmp_path):
    graded, source = fixture(tmp_path)
    edits = (
        {"no_vig_probability": 0.51},
        {"simulated_net_units": 25.0},
        {"model_brier": 0.0001},
        {"result": "push"},
        {"quote_at": "2026-10-08T20:00:00+00:00"},
    )
    for changes in edits:
        rows = graded.to_dicts()
        rows[0].update(changes)
        result = build_market_benchmarks(
            pl.DataFrame(rows), forward_report=source
        )
        assert result["status"] == "BLOCKED_SOURCE_INTEGRITY"
        assert result["integrity"]["integrity_errors"]


def test_later_same_book_prices_never_enter_market_forecast_or_rules(tmp_path):
    graded, source = fixture(tmp_path)
    first = build_market_benchmarks(graded, forward_report=source)
    tampered = graded.with_columns(
        pl.lit("CHANGED_LATER_HANDICAP").alias("near_kickoff_status"),
        pl.lit(90.0).alias("near_kickoff_price_change_pp"),
        pl.lit(-99.5).alias("near_kickoff_line"),
    )
    second = build_market_benchmarks(tampered, forward_report=source)
    assert first["predictor_metrics"] == second["predictor_metrics"]
    assert first["paper_policy_metrics"] == second["paper_policy_metrics"]


def test_multiple_markets_count_same_game_but_week_bootstrap_clusters(tmp_path):
    graded, source = fixture(
        tmp_path, games=120, markets=("spread", "moneyline", "total")
    )
    report = build_market_benchmarks(graded, forward_report=source)
    baseline = _row(report, "market_only")
    model = _row(report, "raw_model")
    assert report["decided"] == 360
    assert baseline["brier"] == pytest.approx(0.25)
    assert model["paired_brier_ci95"] is not None
    assert model["paired_log_loss_ci95"] is not None
    assert baseline["paired_brier_ci95"] == [0.0, 0.0]
    assert model["ci_supported"]
    assert _paper(report, "no_bet_market_only")[
        "net_units_per_candidate_ci95"
    ] == [0.0, 0.0]
    assert report["intervals"]["multiplicity_adjusted"] is False
    # Market-specific cohorts remain paired rather than borrowing other market rows.
    assert _row(
        report, "raw_model", cohort="market", market="spread"
    )["decided"] == 120
    assert len(report["weekly_market_metrics"]) == 12 * 3 * len(FORECAST_WEIGHTS)


def test_ci_withholds_underpowered_rows_and_repeated_games():
    values = [
        {"game_id": f"g{i}", "season": 2026, "week": 5 + i // 10,
         "lift": .02 if i % 2 else .04}
        for i in range(120)
    ]
    assert _week_ci(values, "lift") == _week_ci(values, "lift")
    assert _week_ci(values[:99], "lift") is None
    assert _week_ci([dict(r, game_id="one") for r in values], "lift") is None
    assert _week_ci([dict(r, week=5 + r["week"] % 7) for r in values], "lift") is None
    assert MIN_DECIDED == 100


def test_report_reproducible_to_reports_outputs_and_docs(tmp_path):
    graded, source = fixture(tmp_path)
    report = build_market_benchmarks(graded, forward_report=source)
    write_market_benchmarks(
        report,
        reports_dir=tmp_path / "reports",
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
    )
    for filename in (
        "forward_market_benchmarks.json",
        "forward_market_benchmark_forecasts.csv",
        "forward_market_benchmark_policies.csv",
        "forward_market_benchmark_weeks.csv",
    ):
        left = (tmp_path / "reports" / filename).read_bytes()
        assert (tmp_path / "outputs" / filename).read_bytes() == left
        assert (tmp_path / "docs" / filename).read_bytes() == left
    with (tmp_path / "docs" / "forward_market_benchmark_forecasts.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4 * len(FORECAST_WEIGHTS)
    summary = json.loads(
        (tmp_path / "reports" / "forward_market_benchmarks.json").read_text()
    )
    assert summary["status"] == "RESEARCH_SAMPLE_UNDERPOWERED"
    assert summary["no_calibration_weights_fitted_to_forward_results"]


def test_empty_future_cohort_reports_pending_without_invented_history():
    forward = {
        "schema_version": 1,
        "spec_version": "nfl_frozen_market_edge_forward_v1",
        "status": "PENDING_FORWARD",
        "research_only": True,
        "staking_authorized": False,
        "production_model_change_enabled": False,
        "forward_results_used_to_refit": False,
        "frozen_candidate_rows": 0,
        "duplicate_frozen_keys": 0,
        "summary": {"graded": 0, "decided": 0},
    }
    report = build_market_benchmarks(pl.DataFrame(), forward_report=forward)
    assert report["status"] == "PENDING_FORWARD"
    assert report["frozen"] == 0
    assert report["decided"] == 0
    assert _row(report, "market_only")["brier"] is None
    assert _paper(report, "no_bet_market_only")["hypothetical_units"] == 0
