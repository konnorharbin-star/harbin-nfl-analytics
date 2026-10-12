"""Prospective observation journal for *public aggregator* NFL quotes.

This is NOT an original bookmaker update timestamp and is NOT evidence that an
account could accept a wager. Saves exactly what was returned by an unauthenticated
public source at collection time, before kickoff, without guessing prices/odds.
Do not supply these rows to the stricter offer-origin contract-first gate.
"""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from scripts.professional_method_audit import _utc

VERSION="PUBLIC_AGGREGATOR_CAPTURE_ONLY_V1"


def _fmt(dt):
    if dt is None:return None
    if not isinstance(dt,datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("invalid original quote timestamp")
    return dt.astimezone(timezone.utc).isoformat()


def _validate_entry(m):
    if m.market_type not in {"spread","moneyline","total"}:
        return "UNSUPPORTED_MARKET"
    if m.provider!="action_network":
        return "NOT_MATCHED_PUBLIC_ACTION_NETWORK_FEED"
    if not m.book or not m.source_event_id:
        return "MISSING_BOOK_OR_SOURCE_EVENT"
    if m.market_type=="moneyline":
        if m.first_line is not None or m.second_line is not None:
            return "MONEYLINE_WITH_LINE"
    else:
        if m.first_line is None or m.second_line is None:
            return "MISSING_POINT_THRESHOLD"
        if not all(math.isfinite(float(z)) for z in (m.first_line,m.second_line)):
            return "INVALID_POINT_THRESHOLD"
        if m.market_type=="spread" and abs(float(m.first_line)+float(m.second_line))>1e-9:
            return "OPPOSING_SPREAD_LINES_DIFFER"
        if m.market_type=="total" and abs(float(m.first_line)-float(m.second_line))>1e-9:
            return "OVER_UNDER_TOTALS_DIFFER"
    if not (isinstance(m.first_american_odds,int) and
            isinstance(m.second_american_odds,int) and
            (m.first_american_odds>=100 or m.first_american_odds<=-100) and
            (m.second_american_odds>=100 or m.second_american_odds<=-100)):
        return "MISSING_OR_INVALID_TWO_WAY_AMERICAN_ODDS"
    if {m.first_side,m.second_side}!=(
        {"over","under"} if m.market_type=="total" else {"home","away"}
    ):
        return "WRONG_OPPOSING_SIDES"
    return None


def capture(markets, *, event_id, kickoff_utc, observed_utc, source_url):
    observed=_utc(observed_utc)
    kickoff=_utc(kickoff_utc)
    if not observed<kickoff:
        raise ValueError("cannot backfill started game as a pregame quote")
    if not isinstance(source_url,str) or not source_url.startswith("https://"):
        raise ValueError("actual public source url required")
    parts=[]
    rejects=[]
    for m in markets:
        if str(m.game_id)!=event_id:
            rejects.append("WRONG_EVENT")
            continue
        invalid=_validate_entry(m)
        if invalid:
            rejects.append(invalid)
            continue
        origin=(_fmt(m.source_quote_at) if m.source_quote_time_verified else None)
        parts.append({
            "market":m.market_type,"book_as_attributed_by_aggregator":m.book,
            "source_event_id":str(m.source_event_id),
            "source_original_book_offer_update_utc":origin,
            "offer_origin_proven":origin is not None,
            "first_side":m.first_side,"first_line":m.first_line,
            "first_american_odds":m.first_american_odds,
            "second_side":m.second_side,"second_line":m.second_line,
            "second_american_odds":m.second_american_odds,
            "source_region":"US_BOOK_ATTRIBUTION_UNVERIFIED",
            "account_acceptance_verifiable":False,
            "source_bookmarket_status_verifiable":False,
        })
    parts.sort(key=lambda x:(x["book_as_attributed_by_aggregator"],x["market"]))
    r={
        "version":VERSION,"league":"NFL","event_id":event_id,
        "kickoff_utc":kickoff.isoformat(),
        "collector_observed_utc":observed.isoformat(),
        "source_url":source_url,
        "source_provenance":"THIRD_PARTY_PUBLIC_SCOREBOARD_CAPTURE_ONLY",
        "bookmaker_original_quote_times_available":sum(
            bool(x["offer_origin_proven"]) for x in parts),
        "paired_public_market_count":len(parts),
        "rejected_source_rows":rejects,
        "quotes":parts,
        "is_authorized_betting_quote":False,
        "is_account_accepted_betmgm_quote":False,
        "any_validated_positive_ev":False,
        "betting_decision":"PASS",
        "notes":"Capture-time historical evidence, not validated fresh book offer or BetMGM wager.",
    }
    r["snapshot_sha256"]=sha256(json.dumps(
        r,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return r


def append_file(path,entry):
    """Append-only: same capture timestamp may never be changed or duplicated."""
    path=Path(path)
    if not entry.get("snapshot_sha256") or entry.get("version")!=VERSION:
        raise ValueError("must provide frozen valid capture")
    history=[]
    if path.exists():
        for line in path.read_text(encoding="utf8").splitlines():
            if line.strip():history.append(json.loads(line))
    if any(x.get("snapshot_sha256")==entry["snapshot_sha256"] or
           (x.get("event_id"),x.get("collector_observed_utc"))==
           (entry["event_id"],entry["collector_observed_utc"])
           for x in history):
        raise ValueError("no silent overwrite of original capture")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as f:
        f.write(json.dumps(entry,sort_keys=True,separators=(",",":"))+"\n")
    return len(history)+1
