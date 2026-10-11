"""Deterministic public bettor audit fixtures: NO REAL BETS OR JASON PICKS."""
import copy
import math
import pytest
from scripts.public_bettor_market_audit import audit


def pick(market="GAME_SPREAD",selection="HOME",line="-3",
         odds=-110,stamp="2026-10-11T20:01:00Z"):
    r={"record_id":"synthetic_1","bettor":"RESEARCH_FIXTURE",
       "source_url":"https://example.com/test", "source_type":"PUBLIC_X_POST",
       "posted_utc":stamp,"captured_utc":"2026-10-11T20:05:00Z",
       "kickoff_utc":"2026-10-12T00:15:00Z","event_id":"synthetic_event",
       "league":"NFL","market":market,"period":"FULL_GAME","participant":"GAME",
       "selection":selection,"line":line if market!="GAME_MONEYLINE" else "NA",
       "american_odds":odds,"bookmaker":"PUBLIC_POST_UNVERIFIED","region":"US"}
    if market=="GAME_SPREAD":r["line_convention"]="HOME_SIGNED"
    return r


def final(home=24,away=21):
    return {"event_id":"synthetic_event","home_score":home,"away_score":away,
            "final_utc":"2026-10-12T04:00:00Z"}


def quote(p,price=-130,line=None):
    return {"event_id":p["event_id"],"market":p["market"],
            "period":p["period"],"participant":p["participant"],
            "selection":p["selection"],"line":p["line"] if line is None else line,
            "american_odds":price,"source_url":"https://example.com/quote",
            "bookmaker":"OTHER","region":"US","quote_utc":"2026-10-11T20:08:00Z",
            "captured_utc":"2026-10-11T20:09:00Z"}


def test_empty_jason_profile_graph_proves_no_wager():
    x=audit([])
    assert x["status"]=="NO_INDIVIDUAL_PICKS_AVAILABLE"
    assert not x["pikkit_profits_confirmed"] and x["decision"]=="PASS"
    assert not x["actual_bets_verified"]


def test_whole_spread_push_is_refund():
    x=audit([pick()], [final()])
    y=x["individual_picks"][0]
    assert y["grade"]=="P" and y["poster_price_flat_one_unit_pnl"]==0
    assert x["by_market"]["GAME_SPREAD"]["pushes"]==1


def test_signed_away_spread_is_opposite_contract():
    p=pick(selection="AWAY",line="-3.5")
    x=audit([p],[final()])
    assert x["individual_picks"][0]["grade"]=="W"


def test_home_spread_cover_and_american_price():
    x=audit([pick(line="-2.5")],[final()])
    assert x["individual_picks"][0]["grade"]=="W"
    assert math.isclose(x["by_bettor"]["RESEARCH_FIXTURE"]["poster_price_hypothetical_one_unit_profit"],100/110,abs_tol=1e-5)


def test_total_under_and_odd_integer_push():
    x=audit([pick(market="GAME_TOTAL",selection="UNDER",line="45")],[final(24,21)])
    assert x["individual_picks"][0]["grade"]=="P"
    p=pick(market="GAME_TOTAL",selection="UNDER",line="45.5")
    assert audit([p],[final(24,21)])["individual_picks"][0]["grade"]=="W"


def test_follower_worse_price_changes_hypothetical_profit():
    p=pick(line="-2.5")
    q=quote(p,price=-130)
    x=audit([p],[final()],{"synthetic_1":q})
    r=x["individual_picks"][0]
    assert r["poster_price_flat_one_unit_pnl"]>r["follower_public_quote_flat_one_unit_pnl"]
    assert not r["actual_wager_verified"]
    assert x["by_market"]["GAME_SPREAD"]["follower_quotes_with_settlement"]==1


def test_stale_follower_price_before_post_rejected():
    p=pick()
    q=quote(p)
    q["quote_utc"]="2026-10-11T19:00:00Z"
    with pytest.raises(ValueError,match="follower quote"):
        audit([p],[final()],{"synthetic_1":q})


def test_follower_different_threshold_rejected():
    p=pick()
    with pytest.raises(ValueError,match="different betting contract"):
        audit([p],[final()],{"synthetic_1":quote(p,line="-2.5")})


def test_profile_screenshot_cannot_impersonate_pick():
    p=pick()
    p["source_type"]="SOCIAL_PROFILE_SCREENSHOT"
    with pytest.raises(ValueError,match="unsupported"):
        audit([p],[final()])


def test_post_kickoff_pick_rejected():
    p=pick(stamp="2026-10-12T00:16:00Z")
    p["captured_utc"]="2026-10-12T00:17:00Z"
    with pytest.raises(ValueError,match="before kickoff"):
        audit([p],[final()])


def test_missing_price_and_stake_never_imputed():
    p=pick()
    p["american_odds"]=None
    with pytest.raises(ValueError,match="provenance"):
        audit([p],[final()])


def test_no_postgame_graded_as_win():
    x=audit([pick()])
    assert x["individual_picks"][0]["grade"]=="PENDING"
    assert x["by_market"]["GAME_SPREAD"]["graded"]==0
    assert x["by_market"]["GAME_SPREAD"]["poster_price_hypothetical_roi"] is None


def test_cannot_score_player_props_or_parlays_from_game_score():
    for m in ("PLAYER_RECEIVING_YARDS","PARLAY"):
        p=pick()
        p["market"]=m
        with pytest.raises(ValueError,match="unsupported market"):
            audit([p],[final()])


def test_duplicate_public_post_rejected():
    p=pick()
    with pytest.raises(ValueError,match="duplicate"):
        audit([p,p],[final()])


def test_moneyline_can_grade_with_no_line():
    p=pick(market="GAME_MONEYLINE",selection="AWAY",line="NA",odds=125)
    x=audit([p],[final(21,24)])
    assert x["individual_picks"][0]["grade"]=="W"
    assert x["by_market"]["GAME_MONEYLINE"]["poster_price_hypothetical_one_unit_profit"]==1.25
