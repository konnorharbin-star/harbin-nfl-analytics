"""Public sportsbook observed snapshot tests — all synthetic prices."""
import json
from datetime import datetime,timezone
from types import SimpleNamespace
import pytest
from scripts.public_quote_capture_journal import capture,append_file,VERSION

def market(book="BetMGM",mkt="spread",origin=None):
    first,second=("over","under") if mkt=="total" else ("home","away")
    a,b={"spread":(-3.5,3.5),"total":(54.5,54.5),"moneyline":(None,None)}[mkt]
    return SimpleNamespace(
      game_id="2026_05_BUF_LA",market_type=mkt,provider="action_network",
      book=book,source_event_id="fixture_source_event",
      source_quote_time_verified=origin is not None,source_quote_at=origin,
      first_side=first,first_line=a,first_american_odds=-110,
      second_side=second,second_line=b,second_american_odds=-110)

def one(markets=None,now="2026-10-12T02:50:00Z"):
    if markets is None:markets=[market()]
    return capture(markets,event_id="2026_05_BUF_LA",
      kickoff_utc="2026-10-13T00:15:00Z",observed_utc=now,
      source_url="https://api.actionnetwork.com/web/v2/scoreboard/nfl")

def test_no_source_quote_time_not_imputed():
    x=one()
    assert x["paired_public_market_count"]==1
    assert x["bookmaker_original_quote_times_available"]==0
    assert x["quotes"][0]["source_original_book_offer_update_utc"] is None
    assert x["collector_observed_utc"]=="2026-10-12T02:50:00+00:00"
    assert not x["is_account_accepted_betmgm_quote"]
    assert x["betting_decision"]=="PASS"

def test_capture_all_books_and_markets_without_wager():
    x=one([market(book="BetMGM"),market(book="FanDuel",mkt="total"),
           market(book="Pinnacle",mkt="moneyline")])
    assert x["paired_public_market_count"]==3
    assert {a["book_as_attributed_by_aggregator"] for a in x["quotes"]}=={"BetMGM","FanDuel","Pinnacle"}

def test_original_source_time_is_distinct_when_provided():
    ts=datetime(2026,10,12,2,48,tzinfo=timezone.utc)
    x=one([market(origin=ts)])
    assert x["bookmaker_original_quote_times_available"]==1
    assert x["quotes"][0]["source_original_book_offer_update_utc"]=="2026-10-12T02:48:00+00:00"

def test_game_after_kickoff_cannot_retroactively_capture():
    with pytest.raises(ValueError,match="started"):
        one(now="2026-10-13T00:16:00Z")

def test_invalid_unmatched_spread_excluded():
    bad=market();bad.second_line=2.5
    x=one([bad])
    assert x["paired_public_market_count"]==0
    assert x["rejected_source_rows"]==["OPPOSING_SPREAD_LINES_DIFFER"]

def test_append_only_refuses_same_stamp_or_content(tmp_path):
    p=tmp_path/"captures.jsonl"
    x=one()
    assert append_file(p,x)==1
    with pytest.raises(ValueError,match="overwrite"):
        append_file(p,x)
    newer=one(now="2026-10-12T03:00:00Z")
    assert append_file(p,newer)==2
    assert [r["collector_observed_utc"] for r in map(json.loads,p.read_text().splitlines())]==[
        "2026-10-12T02:50:00+00:00","2026-10-12T03:00:00+00:00"]

def test_wrong_event_is_never_copied():
    bad=market();bad.game_id="different"
    x=one([bad])
    assert not x["quotes"] and "WRONG_EVENT" in x["rejected_source_rows"]
