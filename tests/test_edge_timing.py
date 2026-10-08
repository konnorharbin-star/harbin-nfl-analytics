"""NFL price timing research: chronological same-book, never an allocation signal."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl.edge_timing import (
    _same_line_minimum_odds,
    enrich_edge_timing,
    write_edge_timing,
)
from nfl.market import american_implied_probability, expected_value_per_unit
from nfl.policy import DEFAULT_POLICY

NOW = datetime(2026, 10, 7, 18, tzinfo=UTC)
KICKOFF = (NOW + timedelta(days=4)).isoformat()


def candidate(**changes):
    row = {
        "season": 2026,
        "week": 5,
        "game_id": "2026_05_NYG_WAS",
        "kickoff": KICKOFF,
        "away_team": "NYG",
        "home_team": "WAS",
        "quant_market": "spread",
        "quant_side": "home",
        "quant_book": "DraftKings",
        "quant_price": -3.5,
        "quant_odds": -110,
        "quant_quote_at": (NOW - timedelta(minutes=2)).isoformat(),
        "quant_probability": 0.70,
        "quant_ev": 0.30,
        "quant_edge": 0.20,
        "market_book_count": 4,
        "market_execution_verified": True,
        "market_quote_timestamp_verified": True,
        "market_quote_sanity_ok": True,
        "market_disagreement_severity": "LOW",
        "market_dispersion_high": False,
        "edge_discovery_tier": "SUPPORTED_RESEARCH",
        "edge_shrunk_probability": 0.63,
        "edge_validated_shrinkage_alpha": 0.75,
        "regime_reliability_ready": True,
        "probability_reliability_ready": True,
        "qb_certainty_veto": False,
        "context_freshness_veto": False,
        "context_injuries_personnel_fresh": True,
        "context_veto": False,
        "quant_signal": "PASS",
        "research_signal": "BET",
        "portfolio_signal": "PASS",
        "portfolio_action": "PASS",
        "portfolio_stake_units": 0.0,
    }
    row.update(changes)
    return row


def capture(*, minutes_before_now=120, line=-3.0, odds=-110,
            book="DraftKings", market="spread", game="2026_05_NYG_WAS",
            kickoff=KICKOFF, side="home"):
    if market == "moneyline":
        first_line, second_line = None, None
    elif market == "spread":
        first_line, second_line = line, -line
    else:
        first_line = second_line = line
    second_side = (
        "away" if market in {"spread", "moneyline"} else "under"
    )
    return {
        "captured_at": (NOW - timedelta(minutes=minutes_before_now)).isoformat(),
        "kickoff": kickoff,
        "game_id": game,
        "market_type": market,
        "provider": "action_network",
        "book": book,
        "first_side": side,
        "first_line": first_line,
        "first_american_odds": odds,
        "second_side": second_side,
        "second_line": second_line,
        "second_american_odds": -110,
    }


def run(row=None, captures=None):
    rows = [row or candidate()]
    history = pl.DataFrame(captures) if captures else pl.DataFrame()
    frame, report = enrich_edge_timing(
        pl.DataFrame(rows), history, policy=DEFAULT_POLICY, as_of=NOW
    )
    return frame.to_dicts()[0], report


def test_observed_spread_worsening_is_shadow_bet_now_with_same_book():
    original = candidate()
    result, report = run(original, [capture()])
    assert result["edge_timing_action"] == "BET_NOW_RESEARCH"
    assert result["edge_timing_market_move"] == "WORSENED"
    assert result["edge_timing_prior_line"] == -3.0
    assert result["edge_timing_line_change_points"] == pytest.approx(-0.5)
    assert result["edge_timing_prior_book"] == "DraftKings"
    assert result["edge_timing_staking_authorized"] is False
    assert result["edge_timing_bet_to_line"] is None
    assert result["edge_timing_same_line_min_american_odds"] is not None
    assert report["actions"]["BET_NOW_RESEARCH"] == 1
    assert report["validated_timing_strategy"] is False
    assert not report["production_stake_authorized"]
    for key, value in original.items():
        assert result[key] == value


def test_spread_improves_and_wait_only_if_historical_observation():
    result, report = run(captures=[capture(line=-4.0)])
    assert result["edge_timing_action"] == "WAIT_MONITOR"
    assert result["edge_timing_market_move"] == "IMPROVED"
    assert report["actions"]["WAIT_MONITOR"] == 1


def test_both_sides_of_total_use_correct_direction():
    over = candidate(quant_market="total", quant_side="over", quant_price=47.5)
    past = capture(market="total", side="over", line=46.5)
    result, _ = run(over, [past])
    assert result["edge_timing_market_move"] == "WORSENED"
    assert result["edge_timing_action"] == "BET_NOW_RESEARCH"
    under = candidate(quant_market="total", quant_side="under", quant_price=47.5)
    past_under = capture(market="total", side="under", line=46.5)
    result, _ = run(under, [past_under])
    assert result["edge_timing_market_move"] == "IMPROVED"
    assert result["edge_timing_action"] == "WAIT_MONITOR"


def test_odds_only_moneyline_uses_current_implied_break_even():
    line = candidate(
        quant_market="moneyline", quant_side="home", quant_price=None,
        quant_odds=140
    )
    old = capture(market="moneyline", line=None, odds=170)
    result, _ = run(line, [old])
    assert result["edge_timing_action"] == "BET_NOW_RESEARCH"
    assert result["edge_timing_reference_line"] is None
    assert result["edge_timing_price_change_pp"] < -1.0


def test_cross_book_and_other_games_never_supply_trend():
    row, _ = run(captures=[
        capture(book="FanDuel", line=-3.0),
        capture(game="2026_05_KC_LV", line=-2.0),
        capture(market="moneyline", line=None, odds=-110),
    ])
    assert row["edge_timing_market_move"] == "NO_HISTORY"
    assert row["edge_timing_action"] == "PASS"


def test_same_book_canonical_aliases_are_safe_and_normalized():
    r, _ = run(captures=[
        capture(book="DraftKings Sportsbook", line=-3.0),
    ])
    assert r["edge_timing_action"] == "BET_NOW_RESEARCH"
    assert r["edge_timing_prior_book"] == "DraftKings Sportsbook"


def test_future_post_kickoff_and_bad_kickoff_capture_not_used():
    row = candidate()
    wrong_kickoff = (NOW + timedelta(days=5)).isoformat()
    captures = [
        capture(minutes_before_now=1, line=-2.0),
        capture(minutes_before_now=-10, line=-1.5),
        capture(minutes_before_now=60, line=-2.0, kickoff=wrong_kickoff),
    ]
    r, _ = run(row, captures)
    assert r["edge_timing_action"] == "PASS"
    assert r["edge_timing_prior_captured_at"] is None


def test_no_out_of_window_capture_no_fake_timing_bet():
    too_old = capture(minutes_before_now=780)
    too_near = capture(minutes_before_now=5)
    r, _ = run(captures=[too_old, too_near])
    assert r["edge_timing_action"] == "PASS"
    assert r["edge_timing_market_move"] == "NO_HISTORY"


def test_falls_back_to_most_recent_eligible_observation_not_best_line():
    captures = [
        capture(minutes_before_now=185, line=-1),
        capture(minutes_before_now=120, line=-4),
        capture(minutes_before_now=20, line=-3),
    ]
    r, _ = run(captures=captures)
    assert r["edge_timing_prior_line"] == -3
    assert r["edge_timing_market_move"] == "WORSENED"
    assert r["edge_timing_observation_gap_minutes"] == pytest.approx(18.0)


def test_line_vs_price_mixed_and_no_arbitrary_win():
    mixed = candidate(quant_odds=-145)
    r, _ = run(mixed, [capture(line=-4.0, odds=-110)])
    assert r["edge_timing_market_move"] == "MIXED_LINE_PRICE"
    assert r["edge_timing_action"] == "PASS"
    assert r["edge_timing_price_change_pp"] < -1.0


def test_aligned_odds_worsening_without_line_move():
    r, _ = run(
        candidate(quant_odds=-125),
        [capture(line=-3.5, odds=-110)]
    )
    assert r["edge_timing_action"] == "BET_NOW_RESEARCH"
    assert r["edge_timing_price_change_pp"] < -1


def test_current_quote_expiry_future_missing_or_nonverified_pass():
    for override in (
        {"quant_quote_at": (NOW - timedelta(hours=3)).isoformat()},
        {"quant_quote_at": (NOW + timedelta(minutes=1)).isoformat()},
        {"market_execution_verified": False},
        {"market_quote_sanity_ok": False},
        {"market_quote_timestamp_verified": False},
        {"quant_book": ""},
        {"quant_odds": -90},
    ):
        r, _ = run(candidate(**override), [capture()])
        assert r["edge_timing_action"] == "PASS"
        assert not r["edge_timing_staking_authorized"]


def test_unreliable_nfl_evidence_cannot_authorize_timing_even_for_strong_signal():
    r, _ = run(
        candidate(
            edge_discovery_tier="EVIDENCE_OR_CONTEXT_BLOCKED",
            edge_validated_shrinkage_alpha=0.0,
            edge_shrunk_probability=0.5,
            research_signal="STRONG",
        ),
        [capture()]
    )
    assert r["edge_timing_market_move"] == "WORSENED"
    assert r["edge_timing_action"] == "PASS"
    assert r["edge_timing_same_line_min_american_odds"] is None
    assert "NO_VALIDATED_NFL_EDGE" in r["edge_timing_reason"]


def test_context_and_cross_book_disagreement_veto_research_action():
    for override in (
        {"qb_certainty_veto": True},
        {"context_freshness_veto": True},
        {"market_dispersion_high": True},
        {"market_disagreement_severity": "HIGH"},
        {"probability_reliability_ready": False},
        {"regime_reliability_ready": False},
    ):
        r, _ = run(candidate(**override), [capture()])
        assert r["edge_timing_action"] == "PASS"


def test_conservative_same_line_price_limit_satisfies_both_guards():
    for p in (0.40, 0.54, 0.61, 0.78):
        odds, break_even = _same_line_minimum_odds(p)
        assert odds is not None and break_even is not None
        assert american_implied_probability(odds) <= p - 0.02 + 1e-10
        assert expected_value_per_unit(p, odds) >= 0.01 - 1e-10
    assert _same_line_minimum_odds(None) == (None, None)


def test_no_repricing_across_spread_or_total_lines():
    for market in ("spread", "total"):
        r, _ = run(
            candidate(quant_market=market),
            [capture(market=market)]
        )
        assert r["edge_timing_bet_to_line"] is None


def test_empty_and_missing_snapshot_schema_does_not_make_up_history():
    enriched, report = enrich_edge_timing(
        pl.DataFrame(), pl.DataFrame(), policy=DEFAULT_POLICY, as_of=NOW
    )
    assert enriched.is_empty()
    assert report["rows"] == 0
    empty, report = run(captures=[capture()])
    assert empty["edge_timing_staking_authorized"] is False


def test_timestamp_aware_report_and_csv_copies(tmp_path):
    row, report = run(captures=[capture()])
    tagged = pl.DataFrame([row])
    write_edge_timing(
        tagged, report, output_dir=tmp_path / "outputs", docs_dir=tmp_path / "docs"
    )
    for name in ("edge_timing.csv", "edge_timing_signals.csv", "edge_timing_report.json"):
        assert (tmp_path / "outputs" / name).read_bytes() == (
            tmp_path / "docs" / name
        ).read_bytes()
    with (tmp_path / "outputs" / "edge_timing_signals.csv").open() as f:
        out = list(csv.DictReader(f))
    assert len(out) == 1
    assert out[0]["edge_timing_action"] == "BET_NOW_RESEARCH"
    loaded = json.loads((tmp_path / "docs" / "edge_timing_report.json").read_text())
    assert loaded["actions"]["BET_NOW_RESEARCH"] == 1


def test_reference_nfl_production_data_alpha_zero_stays_research_pass():
    r, _ = run(
        candidate(edge_discovery_tier="EVIDENCE_OR_CONTEXT_BLOCKED",
                  edge_validated_shrinkage_alpha=0),
        [capture(line=-3)]
    )
    assert r["edge_timing_action"] == "PASS"
    assert r["edge_timing_bet_to_line"] is None
