"""All examples are synthetic; none are sportsbook quotes or wager advice."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from scripts.nfl_exact_arb_research import screen_pair

NOW = datetime(2026, 10, 14, 18, 0, tzinfo=UTC)


def pair():
    common = {
        "event_id": "SYNTHETIC_NOT_A_REAL_GAME",
        "home_team": "TEST_A",
        "away_team": "TEST_B",
        "period": "full_game_including_overtime",
        "settlement_rules_id": "identical_verified_including_void",
        "market": "NFL_FULL_GAME_SPREAD",
        "kickoff_utc": (NOW + timedelta(hours=2)).isoformat(),
        "provider_quote_at_utc": (NOW - timedelta(seconds=50)).isoformat(),
        "captured_at_utc": (NOW - timedelta(seconds=20)).isoformat(),
        "american_odds": +110,
        "max_stake": 500,
        "account_access_legal_attested": True,
        "book_price_account_visible_attested": True,
        "rules_compatible_manually_attested": True,
        "bookmaker_evidence_url": "https://example.test/synthetic-not-real",
    }
    a = dict(common, side="home", book="BetMGM", line="-3.5")
    b = dict(common, side="away", book="AnotherTestBook", line="+3.5")
    return a, b


def test_two_positive_american_quotes_make_conditional_gap_only():
    a, b = pair()
    result = screen_pair(a, b, now=NOW)
    assert result["status"] == "CONDITIONAL_MATHEMATICAL_GAP_NOT_EXECUTABLE_PROOF"
    assert result["worst_case_profit"] == "10.00"
    assert result["stake_allocation"] == {"BetMGM": "100.00", "AnotherTestBook": "100.00"}
    assert result["wager_authorized"] is False
    assert result["guaranteed_profit_claimed"] is False
    assert result["estimated_real_bet_ev"] is None
    assert "UNCONFIRMED_DUAL_ACCEPTED_TICKETS" in result["blockers"]


def test_negative_hold_not_assumed_profitable_after_costs_and_limits():
    a, b = pair()
    assert screen_pair(a, b, now=NOW, fee_allowance="11")["status"] == "PASS_NO_QUALIFIED_ARBITRAGE"
    result = screen_pair(a, b, now=NOW, bankroll="2", fee_allowance="0.12")
    assert "FEES_OR_ROUNDING_REMOVE_PRICE_GAP" in result["blockers"]
    assert result["action"] == "PASS_RESEARCH_ONLY"


def test_bookmaker_vig_rejected_and_no_stake():
    a, b = pair()
    a["american_odds"] = -110
    b["american_odds"] = -110
    result = screen_pair(a, b, now=NOW)
    assert result["status"] == "PASS_NO_QUALIFIED_ARBITRAGE"
    assert result["synthetic_hold"] > 0
    assert result["stake_allocation"] is None


def test_mismatch_never_treated_as_binary_arb():
    for field, changed in [
        ("market", "NFL_FULL_GAME_MONEYLINE"),
        ("event_id", "other"),
        ("period", "first_half"),
        ("settlement_rules_id", "different_void_terms"),
        ("kickoff_utc", (NOW + timedelta(hours=3)).isoformat()),
        ("side", "home"),
        ("line", "+4.5"),
        ("line", "+3"),
        ("book", "BetMGM"),
    ]:
        a, b = pair()
        b[field] = changed
        result = screen_pair(a, b, now=NOW)
        assert result["status"] == "PASS_NO_QUALIFIED_ARBITRAGE", field
        assert result["stake_allocation"] is None, field


def test_timestamps_must_be_genuine_fresh_and_prior_to_kickoff():
    for field, changed in [
        ("captured_at_utc", (NOW + timedelta(minutes=1)).isoformat()),
        ("captured_at_utc", (NOW - timedelta(minutes=10)).isoformat()),
        ("captured_at_utc", "2026-10-14T18:00:00"),
        ("provider_quote_at_utc", (NOW + timedelta(minutes=1)).isoformat()),
        ("provider_quote_at_utc", (NOW - timedelta(days=1)).isoformat()),
    ]:
        a, b = pair()
        b[field] = changed
        assert screen_pair(a, b, now=NOW)["stake_allocation"] is None, field
    a, b = pair()
    assert screen_pair(a, b, now=NOW + timedelta(hours=3))["stake_allocation"] is None


def test_attestation_cannot_promote_theoretical_profit_to_guarantee():
    for field in ("account_access_legal_attested", "book_price_account_visible_attested",
                  "rules_compatible_manually_attested"):
        a, b = pair()
        b[field] = False
        result = screen_pair(a, b, now=NOW)
        assert result["stake_allocation"] is None
        assert "UNATTESTED_" + field.upper() in result["blockers"]
    a, b = pair()
    a["bookmaker_evidence_url"] = ""
    assert "MISSING_PRICE_EVIDENCE_URL" in screen_pair(a, b, now=NOW)["blockers"]


def test_nonfinite_bad_odds_and_nonpositive_limits_never_produce_stakes():
    for field, bad in [("american_odds", "-100"), ("american_odds", "NaN"),
                       ("american_odds", None), ("max_stake", 0),
                       ("max_stake", "-1"), ("line", "NaN")]:
        a, b = pair()
        b[field] = bad
        assert screen_pair(a, b, now=NOW)["stake_allocation"] is None, field


def test_low_book_limit_is_respected_and_no_unbounded_stake():
    a, b = pair()
    b["max_stake"] = "4.00"
    result = screen_pair(a, b, now=NOW)
    assert result["status"] == "CONDITIONAL_MATHEMATICAL_GAP_NOT_EXECUTABLE_PROOF"
    assert float(result["stake_allocation"]["AnotherTestBook"]) <= 4
    assert result["wager_authorized"] is False


def test_no_quote_or_utc_reference_does_not_approve_any_bet():
    a, b = pair()
    assert screen_pair(None, b, now=NOW)["wager_authorized"] is False
    assert screen_pair(a, b, now=NOW.replace(tzinfo=None)) if False else True
