"""Release-gated edge ranks cannot turn stale prices or raw EV into bets."""
from datetime import UTC, datetime

import pytest

from scripts.final_edge_board import build, inspect

NOW = datetime(2026, 10, 9, 23, 30, tzinfo=UTC)


def nfl(**kwargs):
    row = {
        "game_id": "n1", "kickoff": "2026-10-11T20:00:00Z",
        "away_team": "A", "home_team": "H", "quant_market": "spread",
        "quant_side": "home", "quant_odds": "-110", "quant_price": "-3.5",
        "quant_book": "DraftKings", "quant_quote_at": "2026-10-09T23:00:00Z",
        "quant_probability": ".65", "quant_ev": ".20",
        "edge_discovery_tier": "SUPPORTED_RESEARCH", "conservative_ev": ".05",
        "regime_reliability_ready": "true",
        "probability_reliability_ready": "true",
        "market_execution_verified": "true",
        "market_quote_timestamp_verified": "true",
        "market_quote_sanity_ok": "true", "betting_action": "BET",
    }
    row.update(kwargs)
    return row


def cfb(**kwargs):
    row = {
        "game_id": "c1", "date": "2026-10-10T18:00:00Z",
        "home_team": "H", "away_team": "A", "market": "spread",
        "side": "H", "line": "-7.5", "odds": "-110",
        "book": "DraftKings", "quote_at": "2026-10-09T23:00:00Z",
        "probability": ".65", "ev": ".2", "edge_priority": "ROBUST_CORE",
        "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "price_evidence_status": "CONFIRMED",
        "subgroup_holdout_confirmed": "True", "subgroup_holdout_bets": "25",
        "historical_price_wilson_lower": ".57",
        "conservative_price_margin": ".046", "line_cushion_points": "1.0",
        "market_execution_verified": "true",
        "market_quote_timestamp_verified": "true",
    }
    row.update(kwargs)
    return row


def test_both_sports_obey_global_release():
    for sport, value in (("nfl", nfl()), ("cfb", cfb())):
        result = inspect(value, sport, {"production_eligible": False}, NOW)
        assert not result["bet_approved"]
        assert result["decision"] == "RESEARCH_CORE"
        assert "GLOBAL_RELEASE_BLOCKED" in result["reasons"]


def test_only_validated_fresh_price_can_be_approved():
    for sport, value in (("nfl", nfl()), ("cfb", cfb())):
        result = inspect(value, sport, {"production_eligible": True}, NOW)
        assert result["bet_approved"]
        assert result["decision"] == "BET_READY"


def test_robust_cfb_cannot_use_unknown_book_or_stale_quote():
    result = inspect(cfb(book="ActionNetwork book 69"), "cfb",
                     {"production_eligible": True}, NOW)
    assert result["decision"] == "RECHECK_PRICE"
    assert "UNRESOLVED_BOOK" in result["reasons"]
    result = inspect(
        cfb(quote_at="2026-10-09T18:00:00Z"), "cfb",
        {"production_eligible": True}, NOW,
    )
    assert "STALE_QUOTE" in result["reasons"]
    assert not result["bet_approved"]


def test_no_book_execution_or_provider_time_never_bet():
    r = inspect(cfb(market_execution_verified="", market_quote_timestamp_verified=""),
                "cfb", {"production_eligible": True}, NOW)
    assert not r["bet_approved"]
    assert "BOOK_EXECUTION_UNVERIFIED" in r["reasons"]
    assert "PROVIDER_TIMESTAMP_UNVERIFIED" in r["reasons"]


def test_one_game_not_three_independent_edges():
    rows = [nfl(quant_market="spread"), nfl(quant_market="moneyline"),
            nfl(quant_market="total", quant_side="over")]
    out, report = build(rows, sport="nfl", gate={"production_eligible": False}, as_of=NOW)
    assert report["raw_market_candidates"] == 3
    assert report["distinct_games"] == 1
    assert report["approved_bets"] == 0
    assert len(out) == 1


def test_game_started_and_unsupported_raw_edge_never_recommended():
    started = inspect(nfl(kickoff="2026-10-09T22:00:00Z"),
                      "nfl", {"production_eligible": True}, NOW)
    assert started["decision"] == "NO_BET"
    huge_ev = inspect(nfl(edge_discovery_tier="EVIDENCE_OR_CONTEXT_BLOCKED",
                          quant_ev="1.9"),
                      "nfl", {"production_eligible": True}, NOW)
    assert huge_ev["decision"] == "NO_BET"


def test_release_authority_is_required():
    with pytest.raises(ValueError, match="release authority"):
        build([nfl()], sport="nfl", gate={}, as_of=NOW)
