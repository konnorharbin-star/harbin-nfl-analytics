"""Market-reference safety and provenance tests: synthetic quotes ONLY."""
from scripts.market_first_reference import market_reference


def pair(market, line, *, quote="2026-10-11T17:00:00Z",
         capture="2026-10-11T17:01:00Z", book="BetMGM",
         source="EDITORIAL"):
    sides = ("HOME", "AWAY") if market=="GAME_SPREAD" else ("OVER", "UNDER")
    result=[]
    for i,side in enumerate(sides):
        item=dict(event_id="SYNTHETIC_GAME",kickoff_utc="2026-10-13T00:00:00Z",
                  market=market,period="FULL_GAME",participant="GAME",
                  line=str(line),selection=side,american_odds="-110",
                  bookmaker=book,region="US_UNVERIFIED",snapshot_id="s_"+market,
                  quote_utc=quote,captured_utc=capture,source_type=source,
                  source_url="https://example.org/dummy-odds")
        if market=="GAME_SPREAD":item["line_convention"]="HOME_SIGNED"
        result.append(item)
    return result


def test_market_center_and_no_false_edge():
    s=pair("GAME_SPREAD",-3.5)
    t=pair("GAME_TOTAL",45.5)
    r=market_reference(s,t,independent=dict(
        projected_home_margin=1.0,projected_total=42.0,
        forecast_utc="2026-10-11T16:00:00Z",model_version="SYNTHETIC"))
    assert r["status"]=="RESEARCH_MARKET_REFERENCE_ONLY"
    x=r["market_reference"]
    assert x["home_expected_margin_reference"]==3.5
    assert x["game_total_reference"]==45.5
    assert x["home_points_reference"]==24.5
    assert x["away_points_reference"]==21.0
    assert r["model_disagreement"]["margin_vs_observed_market_reference"]==-2.5
    assert r["model_disagreement"]["total_vs_observed_market_reference"]==-3.5
    assert not r["bet_authorized"] and not r["validated_profitable_edge"]
    assert not x["reference_is_independent_model"]


def test_cross_timestamp_blocks_market_reference():
    t=pair("GAME_TOTAL",45.5,quote="2026-10-11T17:03:00Z",
           capture="2026-10-11T17:04:00Z")
    r=market_reference(pair("GAME_SPREAD",-3.5),t)
    assert r["status"]=="BLOCKED" and r["market_reference"] is None


def test_different_event_blocks_reference():
    t=pair("GAME_TOTAL",45.5)
    t[0]["event_id"]="OTHER"
    t[1]["event_id"]="OTHER"
    r=market_reference(pair("GAME_SPREAD",-3.5),t)
    assert r["status"]=="BLOCKED"


def test_only_one_side_or_wrong_line_blocks():
    s=pair("GAME_SPREAD",-3.5)
    assert market_reference(s[:1],pair("GAME_TOTAL",45.5))["status"]=="BLOCKED"
    s[1]["line"]="-2.5"
    assert market_reference(s,pair("GAME_TOTAL",45.5))["status"]=="BLOCKED"


def test_future_model_timestamp_fails():
    r=market_reference(pair("GAME_SPREAD",-3.5),pair("GAME_TOTAL",45.5),
                       independent=dict(projected_home_margin=1.5,projected_total=44.,
                                        forecast_utc="2026-10-12T00:00:00Z"))
    assert r["market_reference"] is None and r["status"]=="BLOCKED"


def test_cross_book_or_period_rejected():
    s,t=pair("GAME_SPREAD",-3.5),pair("GAME_TOTAL",45.5)
    for row in t:row["bookmaker"]="OTHER"
    assert market_reference(s,t)["status"]=="BLOCKED"
    for row in t:
        row["bookmaker"]="BetMGM"
        row["period"]="FIRST_HALF"
    assert market_reference(s,t)["status"]=="BLOCKED"


def test_after_kickoff_rejected():
    s=pair("GAME_SPREAD",-3.5,capture="2026-10-13T00:01:00Z")
    t=pair("GAME_TOTAL",45.5,capture="2026-10-13T00:01:00Z")
    assert market_reference(s,t)["status"]=="BLOCKED"


def test_public_book_page_provisional_only():
    s=pair("GAME_SPREAD",-3.5,source="PUBLIC_BOOK_PAGE")
    t=pair("GAME_TOTAL",45.5,source="PUBLIC_BOOK_PAGE")
    r=market_reference(s,t)
    assert r["market_reference"]["market_source_status"].startswith("PUBLIC")
    assert not r["market_reference"]["reference_is_validated_betmgm_executable_quote"]


def test_account_observed_still_not_executable():
    s=pair("GAME_SPREAD",-3.5,source="ACCOUNT_OBSERVED_UNVERIFIED")
    t=pair("GAME_TOTAL",45.5,source="ACCOUNT_OBSERVED_UNVERIFIED")
    r=market_reference(s,t)
    assert r["market_reference"]["market_source_status"]=="ACCOUNT_OBSERVED_NOT_ACCEPTANCE_VERIFIED"
    assert r["decision"]=="PASS"


def test_integer_spread_reference_not_fair_push_probability():
    s=pair("GAME_SPREAD",-3)
    t=pair("GAME_TOTAL",44)
    r=market_reference(s,t)
    assert r["market_reference"]["home_expected_margin_reference"]==3.
    assert "p_push" not in r["market_reference"]
