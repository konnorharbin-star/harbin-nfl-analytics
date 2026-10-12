"""Tests for timestamp-preserving, original-offer-time quote ingestion."""
from datetime import datetime
from types import SimpleNamespace
from scripts.action_network_contract_feed import adapt,probe

NOW="2026-10-12T02:35:00Z"
TIMESTAMP=datetime.fromisoformat("2026-10-12T02:30:00+00:00")
GAMES={"2026_05_BUF_LA":"2026-10-13T00:15:00Z"}

def offer(kind="spread",origin=TIMESTAMP,provider="action_network",book="BetMGM"):
    home,away={"spread":(-3.5,3.5),"total":(54.5,54.5),"moneyline":(None,None)}[kind]
    first,second=("over","under") if kind=="total" else ("home","away")
    return SimpleNamespace(game_id="2026_05_BUF_LA",market_type=kind,
      provider=provider,book=book,source_event_id="fixture",
      captured_at=datetime.fromisoformat("2026-10-12T02:35:00+00:00"),
      source_quote_time_verified=origin is not None,source_quote_at=origin,
      first_side=first,first_line=home,first_american_odds=-110,
      second_side=second,second_line=away,second_american_odds=-105)

def test_spread_normalizes_home_signed_for_both_sides():
    inp,errors=adapt([offer()],GAMES,observed_utc=NOW)
    assert not errors and len(inp["markets"])==2
    q=inp["markets"][0]["quotes"]
    assert [r["line"] for r in q]==[-3.5,-3.5]
    assert [r["selection"] for r in q]==["HOME","AWAY"]
    assert q[0]["quote_utc"]=="2026-10-12T02:30:00+00:00"
    assert q[0]["captured_utc"]!=q[0]["quote_utc"]
    assert q[0]["region"]=="US_UNVERIFIED"

def test_price_is_research_only_even_when_book_is_betmgm():
    _,r=probe([offer()],GAMES,NOW)
    assert r["eligible_quote_pairs"]==1
    assert r["account_verified_quotes"]==0 and r["decision"]=="PASS"
    assert all(not m["approved_wager"] for m in r["markets"])

def test_missing_origin_never_becomes_now():
    inp,rejected=adapt([offer(origin=None)],GAMES,observed_utc=NOW)
    assert not inp["markets"]
    assert rejected[0]["reason"]=="MISSING_TWO_SIDED_OFFER_ORIGIN_TIMES"

def test_espn_without_book_origin_is_rejected():
    inp,rejected=adapt([offer(provider="espn")],GAMES,observed_utc=NOW)
    assert not inp["markets"]

def test_stale_origin_rejected():
    old=datetime.fromisoformat("2026-10-12T01:00:00+00:00")
    inp,rejected=adapt([offer(origin=old)],GAMES,observed_utc=NOW)
    assert not inp["markets"] and rejected[0]["reason"]=="STALE_SOURCE_ORIGIN_OLDER_THAN_15_MIN"

def test_future_quote_not_retroactive():
    future=datetime.fromisoformat("2026-10-12T02:40:00+00:00")
    inp,_=adapt([offer(origin=future)],GAMES,observed_utc=NOW)
    assert not inp["markets"]

def test_started_game_not_an_unplayed_market():
    inp,_=adapt([offer()],{"2026_05_BUF_LA":"2026-10-12T02:34:00Z"},observed_utc=NOW)
    assert not inp["markets"]

def test_noncomplementary_spread_blocked():
    m=offer();m.second_line=2.5
    inp,rejected=adapt([m],GAMES,observed_utc=NOW)
    assert not inp["markets"] and rejected[0]["reason"]=="NON_COMPLEMENTARY_SPREAD_LINES"

def test_moneyline_and_total_conversions():
    inp,_=adapt([offer("moneyline"),offer("total")],GAMES,observed_utc=NOW)
    assert len(inp["markets"])==4
    assert inp["markets"][0]["quotes"][0]["line"]=="NA"
    assert inp["markets"][2]["quotes"][0]["line"]==54.5

def test_empty_snapshot_means_no_edge():
    _,r=probe([],GAMES,NOW)
    assert r["screened_contracts"]==0 and r["best_verified_edge"] is None
