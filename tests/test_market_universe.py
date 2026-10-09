"""Independent research scanner covers all supported markets without fake odds."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from nfl.espn_market import ESPNTwoWayMarket
from nfl.market_universe import (
    MARKET_CATALOG, evaluate_market_universe, from_game_markets,
)

NOW = datetime(2026, 10, 8, 23, 0, tzinfo=UTC)
START = NOW + timedelta(hours=1)
GAME = "2026_05_TB_DAL"


def quote(book, side, odds, *, period="full_game", market="total",
          line=48.5, subject="", stamp=None, verified=True, game=GAME):
    ts = stamp or NOW - timedelta(minutes=2)
    return {
        "game_id": game, "period": period, "market": market,
        "subject_id": subject, "line": line, "side": side, "book": book,
        "american_odds": odds, "provider": "action_network",
        "source_event_id": "event-1", "captured_at": ts,
        "source_quote_at": ts, "source_offer_timestamp_verified": verified,
    }


def add_pair(collection, book, o, u, **args):
    collection.extend([quote(book, "over", o, **args),
                       quote(book, "under", u, **args)])


def run(rows):
    return evaluate_market_universe(
        rows, kickoffs={GAME: START}, as_of=NOW
    )


def test_catalog_includes_main_alt_period_team_and_player_props():
    assert "moneyline" in MARKET_CATALOG["full_game"]
    assert "alt_spread" in MARKET_CATALOG["full_game"]
    assert "spread" in MARKET_CATALOG["first_half"]
    assert "total" in MARKET_CATALOG["fourth_quarter"]
    assert "team_total" in MARKET_CATALOG["first_quarter"]
    assert "receptions" in MARKET_CATALOG["player"]
    assert "anytime_touchdown" in MARKET_CATALOG["player"]
    assert "rushing_yards" in MARKET_CATALOG["player"]


def test_player_receptions_price_discrepancy_requires_real_player_id():
    rows = []
    add_pair(rows, "Book A", 130, -150, period="player", market="receptions",
             line=4.5, subject="player-34")
    for book in ("Book B", "Book C", "Book D"):
        add_pair(rows, book, -120, 100, period="player", market="receptions",
                 line=4.5, subject="player-34")
    r = run(rows)
    assert r["summary"]["research_price_discrepancies"] >= 1
    value = next(x for x in r["research_candidates"]
                 if x["side"] == "over" and x["book"] == "Book A")
    assert value["period"] == "player"
    assert value["subject_id"] == "player-34"
    assert value["other_books"] == 3
    assert value["conservative_consensus_proxy_ev"] > 0.10
    assert value["true_probability_verified"] is False
    assert value["price_executable_verified"] is False
    assert value["bet_placed"] is False
    assert not r["automatic_betting_enabled"]


def test_missing_player_id_is_not_filled_in_by_guess():
    rows = []
    for book in ("A", "B", "C", "D"):
        add_pair(rows, book, +140, -175, period="player",
                 market="receptions", line=4.5, subject="")
    r = run(rows)
    assert r["summary"]["origin_valid_offers"] == 0
    assert r["exclusions"]["MISSING_SUBJECT_ID"] == 8
    assert not r["research_candidates"]


def test_alt_spreads_require_same_home_handicap_across_books():
    rows = []
    add_pair(rows, "A", +150, -180, market="alt_total", line=52.5)
    for book in ("B", "C", "D"):
        add_pair(rows, book, -125, +105, market="alt_total", line=51.5)
    assert not run(rows)["research_candidates"]


def test_game_pair_conversion_preserves_home_side_canonical_line():
    market = ESPNTwoWayMarket(
        game_id=GAME, market_type="spread", provider="action_network",
        book="Book A", source_event_id="event-1",
        captured_at=NOW - timedelta(minutes=2),
        first_side="away", first_line=+9.5,
        first_american_odds=-110, second_side="home",
        second_line=-9.5, second_american_odds=-110,
        source_quote_at=NOW - timedelta(minutes=2),
        source_quote_time_verified=True,
    )
    offers = from_game_markets([market])
    assert len(offers) == 2
    assert {x["line"] for x in offers} == {-9.5}
    assert {x["side"] for x in offers} == {"home", "away"}


def test_no_fictitious_props_report_missing_free_coverage():
    r=run([])
    assert r["summary"]["research_price_discrepancies"] == 0
    assert r["summary"]["origin_valid_offers"] == 0
    assert r["summary"]["markets_without_supplied_source"] > 20
    assert next(x for x in r["market_coverage"] if x["market"]=="receptions"
                )["coverage"] == "NO_FREE_SOURCE_OBSERVED"


def test_unverified_capture_origin_prevents_fake_arbitrage():
    rows = []
    add_pair(rows, "A", +140, -180, verified=False)
    for book in ("B", "C", "D"):
        add_pair(rows, book, -125, +105, verified=False)
    r=run(rows)
    assert not r["research_candidates"]
    assert r["summary"]["origin_valid_offers"] == 0
    assert r["exclusions"]["UNVERIFIED_BOOK_QUOTE_ORIGIN"]==8


def test_missing_three_other_books_cannot_call_fair_price():
    rows=[]
    for book in ("A","B","C"):
        add_pair(rows,book,+160,-180)
    assert not run(rows)["research_candidates"]


def test_book_aliases_do_not_count_as_independent_books():
    rows=[]
    for book in ("Draft Kings", "DraftKings", "BetMGM", "FanDuel"):
        add_pair(rows,book,+150,-175)
    assert run(rows)["summary"]["multi_book_two_way_markets"]==0


def test_different_players_and_markets_never_mix():
    rows=[]
    add_pair(rows,"A",+130,-150,period="player",market="receptions",
             line=4.5,subject="player-1")
    for book in ("B","C","D"):
        add_pair(rows,book,-115,-105,period="player",market="receptions",
                 line=4.5,subject="player-2")
    assert not run(rows)["research_candidates"]


def test_stale_undated_and_postkickoff_quotes_rejected():
    rows=[
        quote("A","over",+200,stamp=NOW-timedelta(minutes=15)),
        quote("B","over",+200,stamp=NOW+timedelta(minutes=2)),
        quote("C","over",+200,stamp=START+timedelta(minutes=2)),
        quote("D","over",+200,stamp=NOW.replace(tzinfo=None)),
    ]
    result=run(rows)
    assert not result["research_candidates"]
    assert result["summary"]["origin_valid_offers"]==0


def test_conflicting_identical_time_at_one_book_excluded():
    rows=[]
    for book in ("A","B","C","D"):
        add_pair(rows,book,-125,+105)
    rows.append(quote("A","over",+160))
    r=run(rows)
    assert not [x for x in r["research_candidates"] if x["book"]=="A"]


def test_baseline_consensus_must_be_stable():
    rows=[]
    add_pair(rows,"A",+150,-180)
    add_pair(rows,"B",-105,-115)
    add_pair(rows,"C",-175,+150)
    add_pair(rows,"D",-150,+125)
    assert not [x for x in run(rows)["research_candidates"]
                if x["book"]=="A"]


def test_improper_market_side_and_subject_are_rejected():
    rows=[
        quote("A","home",-110,market="total"),
        quote("B","over",-110,market="team_total",subject=""),
        quote("C","over",-110,market="total",subject="player-1"),
    ]
    d=run(rows)["exclusions"]
    assert d["INVALID_MARKET_SIDE"]==1
    assert d["MISSING_SUBJECT_ID"]==1
    assert d["UNEXPECTED_SUBJECT_ID"]==1


def test_invalid_negative_and_even_odds_fail_closed():
    rows=[
        quote("A","over",+99),
        quote("A","under",-50),
        quote("B","over",0),
    ]
    assert run(rows)["exclusions"]["INVALID_AMERICAN_ODDS"]==3


def test_does_not_report_an_edge_when_other_books_agree_price_is_fair():
    rows=[]
    for book in ("A","B","C","D"):
        add_pair(rows,book,-110,-110)
    assert run(rows)["research_candidates"] == []


def test_non_aware_asof_is_rejected():
    with pytest.raises(ValueError):
        evaluate_market_universe([],kickoffs={},as_of=NOW.replace(tzinfo=None))
