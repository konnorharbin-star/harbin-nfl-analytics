"""Regression tests for exact-contract professional-method guard."""
import copy
import math
from scripts.professional_method_audit import audit, fair_snapshot, decimal_odds


def q(selection, *, line="44.5", quote="2026-10-11T20:00:00Z",
      captured="2026-10-11T20:05:00Z", odds="-110",
      source="EDITORIAL", snap="test-one"):
    return {
        "event_id": "2026_05_BUF_LA", "kickoff_utc": "2026-10-13T00:15:00Z",
        "market": "GAME_TOTAL", "period": "FULL_GAME", "participant": "GAME",
        "line": line, "selection": selection, "american_odds": odds,
        "bookmaker": "BetMGM", "region": "US_UNVERIFIED", "snapshot_id": snap,
        "quote_utc": quote, "captured_utc": captured,
        "source_type": source, "source_url": "https://example.com/dated-odds",
    }


def test_no_vig_50_50_and_vig():
    snap = fair_snapshot(q("OVER"), q("UNDER"))
    assert math.isclose(snap["no_vig_conditional_nonpush"]["OVER"], .5)
    assert math.isclose(snap["overround"], 0.047619047619, abs_tol=1e-9)


def test_never_promotes_even_positive_diagnostic_ev():
    m = {"source_kind": "INDEPENDENT_MODEL", "p_win": .60,
         "p_push": 0, "p_loss": .40, "forecast_utc": "2026-10-11T19:00:00Z",
         "validation_status": "CLAIMED_VALIDATED"}
    r = audit(q("OVER"), q("UNDER"), selected="OVER", model=m)
    assert r["decision"] == "PASS" and not r["bet_authorized"]
    assert r["model"]["diagnostic_ev_per_dollar"] > 0
    assert "NO_VERIFIED_EXECUTABLE_US_BETMGM_ACCOUNT_QUOTE" in r["blockers"]


def test_mismatched_spread_or_total_threshold_is_not_devig():
    r = audit(q("OVER", line="44.5"), q("UNDER", line="43.5"),
              selected="OVER")
    assert r["entry_market"] is None
    assert any("same two-sided" in e for e in r["blockers"])


def test_different_quote_timestamps_block_devig():
    r = audit(q("OVER"), q("UNDER", quote="2026-10-11T20:02:00Z"),
              selected="OVER")
    assert r["entry_market"] is None


def test_no_model_no_ev():
    r = audit(q("OVER"), q("UNDER"), selected="OVER")
    assert r["model"] is None and r["decision"] == "PASS"


def test_editorial_expert_not_used_as_model():
    model = {"source_kind": "EXPERT_PICK", "p_win": .99}
    r = audit(q("OVER"), q("UNDER"), selected="OVER", model=model)
    assert r["model"] is None
    assert "EXPERT_OPINION_IS_NOT_A_CALIBRATED_PROBABILITY" in r["blockers"]


def test_future_model_forecast_is_rejected():
    model = {"source_kind": "INDEPENDENT_MODEL", "p_win": .6,
             "p_push": 0, "p_loss": .4, "forecast_utc": "2026-10-11T21:00:00Z"}
    r = audit(q("OVER"), q("UNDER"), selected="OVER", model=model)
    assert r["model"] is None


def test_push_expected_value_math():
    model = {"source_kind": "INDEPENDENT_MODEL", "p_win": .50,
             "p_push": .05, "p_loss": .45, "forecast_utc": "2026-10-11T19:00:00Z"}
    r = audit(q("OVER"), q("UNDER"), selected="OVER", model=model)
    assert math.isclose(r["model"]["diagnostic_ev_per_dollar"], 0.50 * 100/110 - 0.45)
    assert math.isclose(r["model"]["conditional_break_even_win"], .95 / (1 + 100/110))


def test_clv_same_contract_price_only():
    entry_o = q("OVER", odds="+110")
    entry_u = q("UNDER", odds="-130")
    close_o = q("OVER", odds="-110", quote="2026-10-12T23:59:00Z",
                captured="2026-10-13T00:01:00Z", snap="close-two")
    close_u = q("UNDER", odds="-110", quote="2026-10-12T23:59:00Z",
                captured="2026-10-13T00:01:00Z", snap="close-two")
    r = audit(entry_o, entry_u, selected="OVER", close_a=close_o, close_b=close_u)
    assert math.isclose(r["closing_line_value"]["price_ratio_minus_one"], .05)
    assert r["decision"] == "PASS"


def test_clv_cross_threshold_is_blocked():
    r = audit(q("OVER"), q("UNDER"), selected="OVER",
              close_a=q("OVER", line="45.5", snap="close"),
              close_b=q("UNDER", line="45.5", snap="close"))
    assert r["closing_line_value"] is None
    assert any("closing contract differs" in e for e in r["blockers"])


def test_no_after_kickoff_snapshot():
    r = audit(q("OVER", captured="2026-10-13T00:20:00Z"),
              q("UNDER", captured="2026-10-13T00:20:00Z"),
              selected="OVER")
    assert r["entry_market"] is None and r["decision"] == "PASS"


def test_spread_requires_home_signed_line():
    a = q("OVER")
    b = q("UNDER")
    for v, side in ((a, "HOME"), (b, "AWAY")):
        v["market"] = "GAME_SPREAD"
        v["selection"] = side
        v["line"] = "-3.5"
    r = audit(a, b, selected="HOME")
    assert r["entry_market"] is None
    a["line_convention"] = "HOME_SIGNED"
    b["line_convention"] = "HOME_SIGNED"
    r2 = audit(a, b, selected="HOME")
    assert r2["entry_market"] is not None


def test_fair_snapshot_never_makes_absolute_push_claim():
    a, b = q("OVER", line="44"), q("UNDER", line="44")
    s = fair_snapshot(a, b)
    assert "p_push" not in s
    assert "no_vig_conditional_nonpush" in s
