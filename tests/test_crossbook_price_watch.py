"""Cross-book public observation diagnostics are never betting recommendations."""
import copy
import math

from scripts.crossbook_price_watch import audit,evaluate_capture

def q(book,market="total",line=54.5,first=-110,second=-110):
    return {
        "market":market,
        "book_as_attributed_by_aggregator":book,
        "first_side":"over" if market=="total" else "home",
        "second_side":"under" if market=="total" else "away",
        "first_line":line if market!="moneyline" else None,
        "second_line":-line if market=="spread" else line if market=="total" else None,
        "first_american_odds":first,
        "second_american_odds":second,
        "source_original_book_offer_update_utc":None,
    }

def snap(quotes=None,event="2026_05_BUF_LA",observed="2026-10-12T02:51:53+00:00"):
    if quotes is None:
        quotes=[
            q("BetMGM"),q("DraftKings"),q("FanDuel"),q("BetRivers"),q("PointsBet")
        ]
    return {"event_id":event,"collector_observed_utc":observed,
            "kickoff_utc":"2026-10-13T00:15:00+00:00",
            "league":"NFL","betting_decision":"PASS","quotes":quotes}

def test_zero_implied_outperformance_no_signal():
    x=audit([snap()])
    assert x["distinct_events_checked"]==1
    assert x["comparable_same_threshold_selections"]==2
    assert x["market_proxy_research_watches"]==0
    assert x["best_validated_betmgm_edge"] is None
    assert x["decision"]=="PASS"

def test_large_market_only_proxy_never_becomes_bet():
    ss=snap([q("BetMGM",first=+130,second=-175),
             q("DraftKings"),q("BetRivers"),q("FanDuel")])
    result=audit([ss])
    assert result["market_proxy_research_watches"]==1
    best=result["watchlist"][0]
    assert best["selection"]=="OVER"
    assert best["same_contract_market_proxy"]["not_true_expected_value"]
    assert best["model_ev"] is None
    assert best["betting_decision"]=="PASS"
    assert result["bet_recommendations"]==[]

def test_different_lines_do_not_count_as_same_contract():
    x=audit([snap([q("BetMGM"),q("DraftKings",line=53.5),
                   q("FanDuel",line=54.5),q("BetRivers",line=54.5)])])
    assert x["comparable_same_threshold_selections"]==0
    assert x["market_proxy_research_watches"]==0

def test_opaque_or_duplicate_book_names_not_accepted_as_comparators():
    x=audit([snap([q("BetMGM"),q("ActionNetwork book 49"),
                   q("DraftKings"),q("FanDuel")])])
    assert x["comparable_same_threshold_selections"]==0

def test_cross_book_disagreement_blocks_even_if_median_looks_positive():
    x=audit([snap([q("BetMGM",first=+140,second=-180),
                   q("DraftKings",first=-300,second=+240),
                   q("FanDuel"),q("BetRivers")])])
    assert x["comparable_same_threshold_selections"]==0
    assert x["market_proxy_research_watches"]==0

def test_whole_total_has_unknown_push():
    x=audit([snap([q("BetMGM",line=54),q("DraftKings",line=54),
                   q("BetRivers",line=54),q("FanDuel",line=54)])])
    assert all(r["p_push"] is None for r in x["watchlist"]) or x["watchlist"]==[]
    rows,_=evaluate_capture(snap([q("BetMGM",line=54),q("DraftKings",line=54),
                   q("BetRivers",line=54),q("FanDuel",line=54)]))
    assert all(r["p_push"] is None for r in rows)

def test_halfpoint_total_cannot_push():
    x,_=evaluate_capture(snap())
    assert len(x)==2
    assert all(x["p_push"]==0 for x in x)

def test_latest_capture_per_event_only():
    older=snap([q("BetMGM",first=+130,second=-175),
                q("DraftKings"),q("BetRivers"),q("FanDuel")],
                observed="2026-10-12T02:20:00+00:00")
    current=snap()
    r=audit([older,current])
    assert r["distinct_events_checked"]==1
    assert r["market_proxy_research_watches"]==0

def test_source_incompatible_is_excluded_not_a_bet():
    invalid=snap()
    invalid["quotes"][0]["source_original_book_offer_update_utc"]="2026-10-12T02:49:00Z"
    x=audit([invalid])
    assert x["market_proxy_research_watches"]==0
    assert x["invalid_source_row_exclusions"]
    assert x["bet_recommendations"]==[]

def test_original_capture_after_kickoff_is_rejected():
    bad=snap(observed="2026-10-13T00:16:00+00:00")
    out=audit([bad])
    assert out["market_proxy_research_watches"]==0
    assert out["invalid_source_row_exclusions"]

def test_market_reference_is_conditional_nonpush_only():
    r=audit([snap([q("BetMGM",market="spread",line=-3,first=-102,second=-115),
                   q("FanDuel",market="spread",line=-3),
                   q("DraftKings",market="spread",line=-3),
                   q("BetRivers",market="spread",line=-3)])])
    assert r["comparable_same_threshold_selections"]==2
    assert r["true_ev_calibrated"] is False
