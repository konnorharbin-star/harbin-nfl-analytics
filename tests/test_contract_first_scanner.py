"""Strict contract-level pricing checks; synthetic odds, no real wager."""
import copy
import math
import pytest

from scripts.contract_first_scanner import screen

NOW="2026-10-11T20:08:00Z"


def quote(market="GAME_TOTAL",selected="OVER",line="44",odds="-110",
          bookmaker="BetMGM",region="US",quote_time="2026-10-11T20:00:00Z",
          captured="2026-10-11T20:04:00Z"):
    item={
        "event_id":"SYNTHETIC_NFL_1","kickoff_utc":"2026-10-12T00:20:00Z",
        "market":market,"period":"FULL_GAME","participant":"GAME",
        "line":line,"selection":selected,"american_odds":odds,
        "bookmaker":bookmaker,"region":region,"source_type":"PUBLIC_BOOK_PAGE",
        "source_url":"https://example.org/artificial","quote_utc":quote_time,
        "captured_utc":captured,"snapshot_id":"synth1"}
    if market=="GAME_SPREAD":item["line_convention"]="HOME_SIGNED"
    return item


def pair(market="GAME_TOTAL",line="44",odds="-110",other_odds="-110"):
    if market=="GAME_TOTAL":sides=("OVER","UNDER")
    elif market=="GAME_SPREAD":sides=("HOME","AWAY")
    else:sides=("HOME","AWAY")
    return [quote(market,sides[0],line,odds),quote(market,sides[1],line,other_odds)]


def doc(a, model=None, now=NOW):
    return {"decision_utc":now,
            "markets":[{"quotes":a,"selected":a[0]["selection"],
                        "independent_model":model}]}


def model(q,pwin=.57,ppush=.05,ploss=.38,lower=.52,**updates):
    m={
        "source_kind":"INDEPENDENT_FORECAST",
        "event_id":q["event_id"],"market":q["market"],
        "period":q["period"],"participant":q["participant"],
        "selection":q["selection"],"line":q["line"],
        "p_win":pwin,"p_push":ppush,"p_loss":ploss,
        "p_win_lower_95":lower,"model_version":"SYNTHETIC_UNVERIFIED",
        "forecast_utc":"2026-10-11T19:45:00Z",
        "validation_evidence_id":"DUMMY_NOT_VALIDATION",
    }
    m.update(updates)
    return m


def test_nothing_offered_means_explicit_pass():
    r=screen({"decision_utc":NOW,"markets":[]})
    assert r["decision"]=="PASS" and r["screened_contracts"]==0
    assert r["best_verified_edge"] is None and not r["recommending_bet"]


def test_market_only_may_not_infer_absolute_win_or_push():
    r=screen(doc(pair()))
    x=r["markets"][0]
    assert x["status"]=="RESEARCH_ONLY"
    assert math.isclose(x["conditional_nonpush_market_probability"],.5)
    assert x["no_vig_probability_does_not_supply_push_probability"]
    assert x["independent_probability"] is None
    assert not x["verified_edge"] and r["best_verified_edge"] is None


def test_incorrect_integer_push_math_would_be_detected():
    a=pair()
    m=model(a[0],pwin=.5,ppush=.05,ploss=.45,lower=.48)
    r=screen(doc(a,m))
    d=r["markets"][0]["independent_probability"]
    expected=.5*100/110-.45
    conservative=.48*100/110-(1-.05-.48)
    assert math.isclose(d["diagnostic_ev_per_dollar"],expected,abs_tol=1e-8)
    assert math.isclose(d["diagnostic_conservative_ev_per_dollar"],conservative,abs_tol=1e-8)
    assert not d["betting_recommendation"]
    assert not r["recommending_bet"]


def test_large_positive_ev_does_not_overrule_no_independent_validation():
    a=pair()
    m=model(a[0],pwin=.99,ppush=0,ploss=.01,lower=.95)
    r=screen(doc(a,m))
    assert r["markets"][0]["independent_probability"]["diagnostic_ev_per_dollar"]>0.8
    assert "MODEL_NOT_APPROVED_BY_INDEPENDENT_FORWARD_TEST" in r["markets"][0]["blockers"]
    assert r["best_verified_edge"] is None


def test_half_point_threshold_push_not_allowed():
    a=pair(line="44.5")
    m=model(a[0],pwin=.57,ppush=.05,ploss=.38)
    r=screen(doc(a,m))
    assert r["markets"][0]["status"]=="BLOCKED"
    assert any("fractional NFL threshold cannot push" in error for error in r["markets"][0]["blockers"])


def test_provided_model_mismatching_exact_line_fails():
    a=pair(line="43.5")
    m=model(a[0],pwin=.60,ppush=0,ploss=.40,line="44")
    r=screen(doc(a,m))
    assert r["markets"][0]["status"]=="BLOCKED"
    assert any("exact selected contract" in v for v in r["markets"][0]["blockers"])


def test_pre_kickoff_enforced_from_decision_time():
    r=screen(doc(pair(),now="2026-10-12T00:20:00Z"))
    assert r["markets"][0]["status"]=="BLOCKED"
    assert any("already kicked off" in error for error in r["markets"][0]["blockers"])


def test_stale_public_price_cannot_be_called_executable():
    a=pair()
    r=screen(doc(a,now="2026-10-11T23:00:00Z"))
    assert "QUOTE_STALE_MORE_THAN_15_MINUTES" in r["markets"][0]["blockers"]
    assert not r["markets"][0]["market_provenance"]["account_executable_verified"]


def test_mismatched_two_way_quotes_are_rejected():
    a=pair()
    a[1]["line"]="45"
    r=screen(doc(a))
    assert r["markets"][0]["status"]=="BLOCKED"
    assert r["markets"][0]["independent_probability"] is None


def test_moneyline_no_vig_market_probability_research_only():
    a=pair("GAME_MONEYLINE","NA",odds="-175",other_odds="+145")
    r=screen(doc(a))
    x=r["markets"][0]
    assert x["status"]=="RESEARCH_ONLY"
    assert 0.60<x["conditional_nonpush_market_probability"]<0.65
    assert x["independent_probability"] is None and not r["recommending_bet"]


def test_player_prop_exact_participant_and_period():
    a=pair(market="PLAYER_RECEPTIONS",line="4.5")
    for x in a:
        x["participant"]="SYNTHETIC_PLAYER"
        x["selection"]="OVER" if x is a[0] else "UNDER"
    m=model(a[0],pwin=.6,ppush=0,ploss=.4,lower=.54)
    r=screen(doc(a,m))
    assert r["markets"][0]["status"]=="RESEARCH_ONLY"
    assert r["markets"][0]["contract"]["participant"]=="SYNTHETIC_PLAYER"


def test_incorrect_probability_sum_is_blocked():
    a=pair()
    m=model(a[0],pwin=.7,ppush=.05,ploss=.38)
    assert screen(doc(a,m))["markets"][0]["status"]=="BLOCKED"


def test_unknown_forecast_asof_not_accepted():
    a=pair()
    m=model(a[0],forecast_utc="2026-10-12T00:23:00Z")
    r=screen(doc(a,m))
    assert r["markets"][0]["status"]=="BLOCKED"


def test_region_us_and_book_mismatch_keep_research_only():
    a=pair()
    for x in a:x["region"]="CAN"
    r=screen(doc(a))
    assert "US_ACCOUNT_REGION_NOT_VERIFIED" in r["markets"][0]["blockers"]


def test_multiple_markets_sorted_diagnostic_but_all_pass():
    a=pair()
    b=pair("GAME_SPREAD","-3",odds="-110")
    m1=model(a[0],pwin=.5,ppush=0,ploss=.5,lower=.48)
    m2=model(b[0],pwin=.7,ppush=0,ploss=.3,lower=.59)
    d={"decision_utc":NOW,"markets":[
        {"quotes":a,"selected":"OVER","independent_model":m1},
        {"quotes":b,"selected":"HOME","independent_model":m2}]}
    r=screen(d)
    ranked=r["provisional_only_diagnostic_contracts"]
    assert len(ranked)==2 and ranked[0]["market"]=="GAME_SPREAD"
    assert not ranked[0]["approved"] and r["best_verified_edge"] is None

def test_model_first_created_after_quote_is_blocked_even_if_pregame():
    a=pair()
    # Price observed at 20:00, captured at 20:04; retroactively matching a
    # model forecast first created at 20:02 would produce false backtest ROI.
    m=model(a[0],forecast_utc="2026-10-11T20:02:00Z")
    r=screen(doc(a,m))
    assert r["decision"]=="PASS"
    assert r["markets"][0]["status"]=="BLOCKED"
    assert any("original quote time" in reason for reason in r["markets"][0]["blockers"])
    assert r["markets"][0]["independent_probability"] is None
    assert r["best_verified_edge"] is None


def test_model_at_exact_original_quote_time_is_still_research_only():
    a=pair()
    m=model(a[0],forecast_utc="2026-10-11T20:00:00Z")
    r=screen(doc(a,m))
    assert r["markets"][0]["status"]=="RESEARCH_ONLY"
    assert r["markets"][0]["independent_probability"] is not None
    assert not r["recommending_bet"]
