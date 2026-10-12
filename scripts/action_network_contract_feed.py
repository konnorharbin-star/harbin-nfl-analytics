"""Free-provider original-time NFL quote ingestion for contract research.

Accept Action Network public scoreboard markets ONLY when both sides have
offer-origin times. Never promote capture time, event update time, or ESPN's
fabricated -110 fallback to actual sportsbook quote time. Source is third-party
US-attributed and NEVER account executable. No bookmaker login or paid API.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math

from scripts.professional_method_audit import _utc
from scripts.contract_first_scanner import screen

SOURCE_TYPE="AGGREGATOR"
SOURCE_URL="https://api.actionnetwork.com/web/v2/scoreboard/nfl"
SUPPORTED={"moneyline":"GAME_MONEYLINE",
           "spread":"GAME_SPREAD","total":"GAME_TOTAL"}
SIDES={"moneyline":("HOME","AWAY"),
       "spread":("HOME","AWAY"),"total":("OVER","UNDER")}


def _iso(dt):
    if not isinstance(dt,datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("source timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat()


def adapt(markets, schedule_by_game, *, observed_utc, origin_url=SOURCE_URL):
    """Produce original-time two-sided snapshots and explicit exclusion counts.

    Accepted object type is existing nfl.espn_market.ESPNTwoWayMarket.
    schedule_by_game maps game_id to verified UTC kickoff datetime strings.
    """
    at=_utc(observed_utc)
    if not isinstance(schedule_by_game,dict):
        raise ValueError("schedule mapping required")
    rows=[]
    rejected=[]
    for m in markets:
        game=str(m.game_id)
        market=m.market_type
        reason=None
        if market not in SUPPORTED:
            reason="UNSUPPORTED_MARKET"
        elif m.provider!="action_network":
            reason="SOURCE_NOT_ORIGIN_TIMED_ACTION_NETWORK"
        elif not m.source_quote_time_verified or m.source_quote_at is None:
            reason="MISSING_TWO_SIDED_OFFER_ORIGIN_TIMES"
        elif not isinstance(m.book,str) or not m.book.strip():
            reason="UNNAMED_SPORTSBOOK"
        elif game not in schedule_by_game:
            reason="MISSING_OFFICIAL_KICKOFF"
        if reason:
            rejected.append({"event_id":game,"market":market,"book":m.book,"reason":reason})
            continue
        source_time=_utc(_iso(m.source_quote_at))
        kickoff=_utc(schedule_by_game[game])
        if not source_time<=at<kickoff:
            rejected.append({"event_id":game,"market":market,"book":m.book,
                             "reason":"AFTER_KICKOFF_OR_FUTURE_ORIGIN_TIME"})
            continue
        if at-source_time>timedelta(minutes=15):
            rejected.append({"event_id":game,"market":market,"book":m.book,
                             "reason":"STALE_SOURCE_ORIGIN_OLDER_THAN_15_MIN"})
            continue
        # Existing class records 'spread' HOME handicap and corresponding AWAY
        # handicap. Normalize to one HOME_SIGNED common threshold for BOTH sides.
        if market=="moneyline":
            if m.first_line is not None or m.second_line is not None:
                reason="MONEYLINE_MUST_NOT_HAVE_POINT_THRESHOLD"
            line="NA"
        elif market=="spread":
            if m.first_line is None or m.second_line is None or not math.isclose(
                float(m.first_line)+float(m.second_line),0,abs_tol=1e-9):
                reason="NON_COMPLEMENTARY_SPREAD_LINES"
                line=None
            else:line=float(m.first_line)
        else:
            if m.first_line is None or m.second_line is None or not math.isclose(
                float(m.first_line),float(m.second_line),abs_tol=1e-9):
                reason="NON_MATCHING_TOTAL_THRESHOLD"
                line=None
            else:line=float(m.first_line)
        if reason:
            rejected.append({"event_id":game,"market":market,"book":m.book,"reason":reason})
            continue
        if {str(m.first_side).lower(),str(m.second_side).lower()}!=(
            {"home","away"} if market!="total" else {"over","under"}
        ):
            rejected.append({"event_id":game,"market":market,"book":m.book,
                             "reason":"NON_COMPLEMENTARY_SELECIONS"})
            continue
        # Exact book and country attribution are NOT independently certified;
        # name "BetMGM" in third party feed does not prove a real US ticket.
        ident=hashlib.sha256((
            game+market+str(m.book)+str(line)+_iso(m.source_quote_at)
        ).encode()).hexdigest()[:18]
        quote_list=[]
        for side,price in ((m.first_side,m.first_american_odds),
                           (m.second_side,m.second_american_odds)):
            q=dict(event_id=game,kickoff_utc=kickoff.isoformat(),
                   market=SUPPORTED[market],period="FULL_GAME",
                   participant="GAME",selection=str(side).upper(),line=line,
                   american_odds=int(price),bookmaker=m.book,
                   region="US_UNVERIFIED",source_type=SOURCE_TYPE,
                   source_url=origin_url,snapshot_id=ident,
                   quote_utc=source_time.isoformat(),
                   captured_utc=at.isoformat())
            if market=="spread":
                q["line_convention"]="HOME_SIGNED"
            quote_list.append(q)
        rows.append({"quotes":quote_list,"selected":quote_list[0]["selection"]})
        rows.append({"quotes":quote_list,"selected":quote_list[1]["selection"]})
    return {"decision_utc":at.isoformat(),"markets":rows},rejected


def probe(markets, schedule_by_game, observed_utc, **kwargs):
    request,exclusions=adapt(markets,schedule_by_game,
                             observed_utc=observed_utc,**kwargs)
    report=screen(request)
    report["source_kind"]="ACTION_NETWORK_THIRD_PARTY_NOT_ACCOUNT_QUOTES"
    report["eligible_quote_pairs"]=len(request["markets"])//2
    report["rejected_source_rows"]=exclusions
    report["live_original_time_verified_quote_pairs"]=len(request["markets"])//2
    report["account_verified_quotes"]=0
    report["bookmakers_seen"]=sorted({
        m["quotes"][0]["bookmaker"] for m in request["markets"]})
    report["no_fake_sportsbook_prices"]=True
    return request,report
