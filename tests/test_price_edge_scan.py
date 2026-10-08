"""Market-price scouting is manual research, not sportsbook execution."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from nfl.espn_market import ESPNTwoWayMarket
from nfl.price_edge_scan import scan_price_edges, write_price_edge_report

NOW = datetime(2026, 10, 8, 23, 10, tzinfo=UTC)
KICKOFF = NOW + timedelta(hours=1)
GAME = "2026_05_TB_DAL"


def market(book, side_a="home", odds_a=110, odds_b=-105,
           market_type="moneyline", line_a=None, line_b=None,
           provider="action_network", captured=None, game=GAME):
    return ESPNTwoWayMarket(
        game_id=game, market_type=market_type, provider=provider,
        book=book, source_event_id="upstream-game-123",
        captured_at=captured or NOW - timedelta(minutes=2),
        first_side=side_a, first_line=line_a,
        first_american_odds=odds_a,
        second_side={
            "home": "away", "away": "home",
            "over": "under", "under": "over",
        }[side_a],
        second_line=line_b, second_american_odds=odds_b,
    )


def scan(data, **kw):
    return scan_price_edges(
        data, kickoffs={GAME: KICKOFF}, as_of=kw.get("now", NOW)
    )


def test_real_math_two_book_moneyline_arbitrage_is_only_theoretical():
    # Book A: home +115 / away -150. Book B: home -150 / away +115.
    result = scan([
        market("Book A", odds_a=115, odds_b=-150),
        market("Book B", odds_a=-150, odds_b=115),
    ])
    assert result["summary"]["theoretical_arbitrage_observations"] == 1
    entry = result["opportunities"][0]
    assert entry["first_book"] == "Book A"
    assert entry["second_book"] == "Book B"
    assert entry["theoretical_fixed_stake_yield"] > 0.07
    assert entry["first_capture_at"]
    assert entry["actual_arbitrage_achieved"] is False
    assert not result["automatic_betting_enabled"]
    assert not result["paid_api_enabled"]
    assert not result["wager_execution_implemented"]


def test_normal_vig_is_not_an_arbitrage():
    result = scan([
        market("Book A", odds_a=-110, odds_b=-110),
        market("Book B", odds_a=-110, odds_b=-110),
    ])
    assert result["opportunities"] == []


def test_identical_book_aliases_not_two_independent_sportsbooks():
    result = scan([
        market("Draft Kings", odds_a=120, odds_b=-150),
        market("DraftKings", odds_a=-150, odds_b=120),
    ])
    assert result["opportunities"] == []


def test_different_spread_lines_cannot_create_a_fake_middle_arb():
    a = market("Book A", market_type="spread",
               odds_a=120, odds_b=-150, line_a=-7.5, line_b=7.5)
    b = market("Book B", market_type="spread",
               odds_a=-150, odds_b=120, line_a=-8.5, line_b=8.5)
    assert scan([a, b])["opportunities"] == []


def test_same_point_spread_and_total_can_generate_theoretical_opportunity():
    spreads = [
        market("Book A", market_type="spread",
               odds_a=120, odds_b=-150, line_a=-7.5, line_b=7.5),
        market("Book B", market_type="spread",
               odds_a=-150, odds_b=120, line_a=-7.5, line_b=7.5),
    ]
    totals = [
        market("Book A", side_a="over", market_type="total",
               odds_a=120, odds_b=-150, line_a=48.5, line_b=48.5),
        market("Book B", side_a="over", market_type="total",
               odds_a=-150, odds_b=120, line_a=48.5, line_b=48.5),
    ]
    result = scan(spreads+totals)
    assert {x["market"] for x in result["opportunities"]} == {"spread","total"}


def test_stale_and_post_kickoff_snapshots_never_qualify():
    good = market("Book A", odds_a=120, odds_b=-140)
    stale = market("Book B", odds_a=-150, odds_b=120,
                   captured=NOW-timedelta(minutes=11))
    later = market("Book B", odds_a=-150, odds_b=120,
                   captured=KICKOFF+timedelta(minutes=1))
    assert not scan([good, stale])["opportunities"]
    assert not scan([good, later])["opportunities"]
    assert not scan([good, replace(later, captured_at=NOW+timedelta(seconds=1))])["opportunities"]


def test_synchronization_and_game_identity_verified():
    first = market("Book A", odds_a=120, odds_b=-150,
                   captured=NOW-timedelta(minutes=9))
    second = market("Book B", odds_a=-150, odds_b=120,
                    captured=NOW-timedelta(minutes=1))
    assert not scan([first, second])["opportunities"]
    assert not scan([first, replace(second, game_id="OTHER")])["opportunities"]


def test_unverified_fallback_and_paid_provider_are_never_evidence():
    q = [
        market("Book A", odds_a=120, odds_b=-150),
        market("Book B", odds_a=-150, odds_b=120,
               provider="nflverse_schedule_snapshot"),
    ]
    assert not scan(q)["opportunities"]
    q[1] = replace(q[1], provider="odds_api")
    assert not scan(q)["opportunities"]


def test_invalid_lines_and_prices_rejected():
    q = market("Book A", market_type="spread",
               line_a=-6.5, line_b=6.0)
    assert not scan([q])["opportunities"]
    q2 = market("Book B", odds_a=90, odds_b=-110)
    assert not scan([q2])["opportunities"]
    q3 = market("Book C", market_type="total",
                line_a=float("nan"), line_b=float("nan"),
                side_a="over")
    assert not scan([q3])["opportunities"]


def test_non_utc_aware_asof_refused():
    with pytest.raises(ValueError):
        scan_price_edges([],kickoffs={},as_of=datetime(2026, 10, 8, 23))


def test_publish_report_without_bankroll_execution(tmp_path):
    target=tmp_path/"price.json"
    result=write_price_edge_report([], kickoffs={}, as_of=NOW,path=target)
    assert target.exists()
    assert result["summary"]["theoretical_arbitrage_observations"] == 0
    assert result["model_win_probability_used"] is False



def test_leave_one_book_out_consensus_research_not_false_proven_edge():
    # Three other books agree, while the fourth has a better home price.
    markets = [
        market("Book A",odds_a=125,odds_b=-155),
        market("Book B",odds_a=-125,odds_b=105),
        market("Book C",odds_a=-125,odds_b=105),
        market("Book D",odds_a=-125,odds_b=105),
    ]
    report = scan(markets)
    watch = report["disagreement_watchlist"]
    assert watch
    best = next(x for x in watch if x["book"]=="Book A" and x["side"]=="home")
    assert best["market_based_theoretical_ev"]>0.10
    assert best["other_books"]==3
    assert best["reference_excludes_candidate_book"] is True
    assert best["model_win_probability_used"] is False
    assert best["true_edge_proven"] is False
    assert "NOT_AN_EXECUTABLE_BET" in best["human_action"]


def test_one_or_two_books_cannot_be_market_fair_price():
    markets = [
        market("Book A",odds_a=125,odds_b=-155),
        market("Book B",odds_a=-125,odds_b=105),
    ]
    assert scan(markets)["disagreement_watchlist"]==[]


def test_high_disagreement_reference_fails_closed():
    markets = [
        market("Book A",odds_a=125,odds_b=-155),
        market("Book B",odds_a=-125,odds_b=105),
        market("Book C",odds_a=-150,odds_b=125),
        market("Book D",odds_a=-110,odds_b=-110),
    ]
    assert not [x for x in scan(markets)["disagreement_watchlist"]
                if x["book"]=="Book A"]


def test_reference_excludes_candidate_and_never_uses_paid_or_stale():
    rows = [
        market("Book A",odds_a=125,odds_b=-155),
        market("Book B",odds_a=-125,odds_b=105),
        market("Book C",odds_a=-125,odds_b=105,
               captured=NOW-timedelta(minutes=11)),
        market("Book D",odds_a=-125,odds_b=105,provider="odds_api"),
    ]
    assert scan(rows)["disagreement_watchlist"]==[]
