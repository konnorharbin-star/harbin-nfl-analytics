from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from nfl.backtest_audit import audit_backtest_bets
from nfl.execution_market import validate_execution_row
from nfl.feature_audit import leakage_columns
from nfl.grading_audit import audit_decisions, audit_graded_bets
from nfl.policy import DEFAULT_POLICY
from nfl.portfolio_audit import audit_portfolio


def test_feature_audit_flags_market_and_outcome_leakage() -> None:
    leaking = leakage_columns(
        ["epa_per_play_matchup_advantage", "closing_spread", "actual_home_margin"]
    )
    assert leaking == ["closing_spread", "actual_home_margin"]


def test_backtest_audit_separates_verified_open_and_final_fallback() -> None:
    bets = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "a",
                "market_type": "spread",
                "side": "home",
                "american_odds": -110,
                "price_stage": "archive_open_line_final_price",
                "has_distinct_open": True,
                "opening_book": "WSGT",
                "clv_proxy": 1.0,
                "result": "win",
                "net_units": 0.91,
            },
            {
                "season": 2025,
                "week": 1,
                "game_id": "b",
                "market_type": "total",
                "side": "over",
                "american_odds": -110,
                "price_stage": "archive_final_fallback",
                "has_distinct_open": False,
                "opening_book": None,
                "clv_proxy": None,
                "result": "loss",
                "net_units": -1.0,
            },
        ]
    )
    report = audit_backtest_bets(bets)
    assert report["errors"] == 0
    assert report["status"] == "WARN"
    assert report["quote_integrity"]["verified_opening_entry_bets"] == 1
    assert report["quote_integrity"]["unverified_or_final_fallback_bets"] == 1


def test_backtest_audit_rejects_clv_on_unverified_entry() -> None:
    bets = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_id": "a",
                "market_type": "spread",
                "side": "home",
                "american_odds": -110,
                "price_stage": "archive_final_fallback",
                "has_distinct_open": False,
                "opening_book": None,
                "clv_proxy": 1.0,
                "result": "win",
                "net_units": 0.91,
            }
        ]
    )
    report = audit_backtest_bets(bets)
    assert report["status"] == "FAIL"
    assert any(issue["code"] == "clv_proxy_on_unverified_entry" for issue in report["issues"])


def test_execution_market_rejects_post_kickoff_quote() -> None:
    kickoff = datetime(2026, 10, 4, 17, tzinfo=UTC)
    quote = kickoff + timedelta(minutes=1)
    row = {
        "quant_market": "spread",
        "quant_side": "home",
        "quant_book": "ESPN BET",
        "quant_price": -3.0,
        "quant_odds": -110,
        "quant_quote_at": quote.isoformat(),
        "market_book_count": 1,
        "kickoff": kickoff.isoformat(),
    }
    ready, reason = validate_execution_row(
        row,
        limits={"require_quote_timestamp_for_execution": True},
        now=quote + timedelta(minutes=1),
    )
    assert not ready
    assert "pre-kickoff" in reason


def test_grading_audit_excludes_post_kickoff_decision() -> None:
    kickoff = datetime(2026, 10, 4, 17, tzinfo=UTC)
    decisions = pl.DataFrame(
        [
            {
                "decision_at": (kickoff + timedelta(minutes=1)).isoformat(),
                "kickoff": kickoff.isoformat(),
                "game_id": "g",
                "quant_market": "spread",
                "portfolio_candidate_units": 0.5,
                "portfolio_action": "PAPER",
                "execution_ready": True,
            }
        ]
    )
    report = audit_decisions(decisions)
    assert report["status"] == "FAIL"
    assert report["timing_excluded_rows"] == 1


def test_graded_audit_rejects_invalid_close_chronology() -> None:
    kickoff = datetime(2026, 10, 4, 17, tzinfo=UTC)
    decision = kickoff - timedelta(hours=2)
    graded = pl.DataFrame(
        [
            {
                "game_id": "g",
                "quant_market": "moneyline",
                "decision_at": decision.isoformat(),
                "kickoff": kickoff.isoformat(),
                "portfolio_verified": True,
                "result": "win",
                "net_units": 1.0,
                "closing_snapshot_at": (kickoff + timedelta(minutes=1)).isoformat(),
            }
        ]
    )
    report = audit_graded_bets(graded)
    assert report["status"] == "FAIL"
    assert any(issue["code"] == "invalid_closing_snapshot_timing" for issue in report["issues"])


def test_portfolio_audit_blocks_real_stake_without_production_gate() -> None:
    kickoff = datetime(2026, 10, 4, 17, tzinfo=UTC)
    quote = kickoff - timedelta(hours=1)
    frame = pl.DataFrame(
        [
            {
                "game_id": "g",
                "home_team": "BUF",
                "away_team": "NE",
                "quant_market": "spread",
                "quant_side": "home",
                "quant_book": "ESPN BET",
                "quant_odds": -110,
                "quant_quote_at": quote.isoformat(),
                "market_book_count": 1,
                "kickoff": kickoff.isoformat(),
                "portfolio_candidate_units": 0.5,
                "portfolio_stake_units": 0.5,
                "portfolio_action": "BET",
                "execution_ready": True,
            }
        ]
    )
    report = audit_portfolio(
        frame,
        policy=DEFAULT_POLICY,
        release_gate={"release_state": "PAPER", "production_eligible": False},
    )
    assert report["status"] == "FAIL"
    assert any(issue["code"] == "real_stake_without_production_gate" for issue in report["issues"])
