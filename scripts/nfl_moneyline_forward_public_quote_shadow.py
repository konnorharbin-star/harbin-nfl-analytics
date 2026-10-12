"""Prospective research-only NFL moneyline Elo versus public attributed market.

Compare only source journal quotes COLLECTED after an original immutable
pregame model probability; quote capture is NOT the bookmaker's true
offer-origin timestamp, nor a confirmed executable U.S. BetMGM price.
No conditional 'true EV' or betting recommendations.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
from hashlib import sha256
import json
from pathlib import Path
from scripts.nfl_moneyline_elo_research import (
    VERSION as ELO_VERSION, market_home_no_vig,
)

VERSION="NFL_FROZEN_MONEYLINE_PUBLIC_PRICE_SHADOW_V1"
ORIGIN="PUBLIC_ACTION_NETWORK_CAPTURE_NO_ORIGINAL_OFFER_TIME"


def utc(s):
    if not isinstance(s,str) or not s:
        raise ValueError("timestamp required")
    dt=datetime.fromisoformat(s.replace("Z","+00:00"))
    if dt.tzinfo is None or dt.utcoffset()!=timedelta(0):
        raise ValueError("UTC timestamp required")
    return dt.astimezone(timezone.utc)


def validate_quote_capture(source):
    raw={key:val for key,val in source.items() if key!="snapshot_sha256"}
    check=sha256(json.dumps(raw,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if source.get("snapshot_sha256")!=check:
        raise ValueError("quote source was not a verified original journal snapshot")
    if (source.get("is_account_accepted_betmgm_quote") is not False or
        source.get("is_authorized_betting_quote") is not False or
        source.get("source_provenance") != "THIRD_PARTY_PUBLIC_SCOREBOARD_CAPTURE_ONLY"):
        raise ValueError("aggregator capture cannot be an authorized book ticket")
    return source


def pair_originals(originals,captures):
    by_event={}
    for row in originals:
        if (row.get("model")!=ELO_VERSION or row.get("decision")!="PASS"
            or row.get("wager") is not False
            or row.get("bookmaker_account_acceptance") is not False):
            raise ValueError("original model source not a moneyline research forecast")
        gid=str(row["game_id"])
        at=utc(row["first_observed_at_utc"])
        ko=utc(row["kickoff_utc"])
        if at>=ko:
            raise ValueError("original probability at/after kickoff")
        if gid in by_event:
            raise ValueError("duplicate immutable probability")
        home=float(row["p_home_win"])
        away=float(row["p_away_win"])
        push=float(row["p_tie_push"])
        cond=float(row["conditional_p_home_win_given_no_tie"])
        if (min(home,away,push)<0 or max(home,away,push)>1 or
            abs(home+away+push-1)>1e-6 or
            abs(cond-home/(home+away))>1e-6):
            raise ValueError("corrupted original moneyline probabilities")
        by_event[gid]=row
    results={}
    rejects={"quote_before_or_at_original_model":0,
             "source_after_kickoff_or_mismatched":0,
             "no_frozen_forecast":0,
             "no_two_way_betmgm_attributed_moneyline":0,
             "not_original_live_BetMGM_offer":0}
    for capture in captures:
        validate_quote_capture(capture)
        gid=str(capture["event_id"])
        model=by_event.get(gid)
        if model is None:
            rejects["no_frozen_forecast"]+=1
            continue
        observed=utc(capture["collector_observed_utc"])
        model_at=utc(model["first_observed_at_utc"])
        kickoff=utc(model["kickoff_utc"])
        if utc(capture["kickoff_utc"])!=kickoff or observed>=kickoff:
            rejects["source_after_kickoff_or_mismatched"]+=1
            continue
        if observed<=model_at:
            rejects["quote_before_or_at_original_model"]+=1
            continue
        prices=[]
        for q in capture.get("quotes",[]):
            if (q.get("book_as_attributed_by_aggregator") != "BetMGM"
                or q.get("market") != "moneyline"
                or {q.get("first_side"),q.get("second_side")}!={"home","away"}
                or q.get("first_line") is not None or q.get("second_line") is not None):
                continue
            # Original-book origin and account acceptance remain unverified,
            # EVEN WHEN an aggregator attributes the entry to BetMGM.
            home_odds=(q["first_american_odds"] if q["first_side"]=="home"
                       else q["second_american_odds"])
            away_odds=(q["first_american_odds"] if q["first_side"]=="away"
                       else q["second_american_odds"])
            p=market_home_no_vig(home_odds,away_odds)
            if p is None:
                continue
            prices.append((p,home_odds,away_odds,q))
        if len(prices)!=1:
            rejects["no_two_way_betmgm_attributed_moneyline"]+=1
            continue
        p_market,home_odds,away_odds,q=prices[0]
        rejects["not_original_live_BetMGM_offer"]+=1
        result={
            "event_id":gid,"kickoff_utc":kickoff.isoformat(),
            "frozen_model_first_observed_utc":model_at.isoformat(),
            "aggregator_capture_utc":observed.isoformat(),
            "aggregator_snapshot_sha256":capture["snapshot_sha256"],
            "source_url":capture["source_url"],
            "book_as_attributed_by_aggregator":"BetMGM",
            "US_BetMGM_account_offer_verified":False,
            "book_original_offer_update_time":None,
            "model_p_home_win":model["p_home_win"],
            "model_p_away_win":model["p_away_win"],
            "model_p_tie_push":model["p_tie_push"],
            "model_conditional_home_given_no_tie":model["conditional_p_home_win_given_no_tie"],
            "aggregator_home_american_odds":home_odds,
            "aggregator_away_american_odds":away_odds,
            "aggregator_de_vig_home_conditional":round(p_market,8),
            "model_minus_market_conditional_reference_difference":round(
                float(model["conditional_p_home_win_given_no_tie"])-p_market,8),
            "estimated_true_EV":None,"validated_profit_edge":False,
            "decision":"PASS",
            "limitations":"Collector capture is NOT sportsbook original offer time; do not infer tradable EV.",
        }
        # First observed postmodel capture per event is audit anchor;
        # later captures can be researched separately, not independent games.
        if gid not in results or observed<utc(results[gid]["aggregator_capture_utc"]):
            results[gid]=result
    paired=sorted(results.values(),key=lambda r:(r["kickoff_utc"],r["event_id"]))
    return {
        "version":VERSION,"decision":"PASS",
        "moneyline_frozen_games":len(by_event),
        "aggregator_observations_scanned":len(captures),
        "postmodel_first_capture_game_pairs":len(paired),
        "origin_bookmaker_offer_times_confirmed":0,
        "accepted_BetMGM_US_account_offers":0,
        "validated_profitable_bets":0,
        "estimated_true_EV":None,
        "wager_recommendations":[],
        "reject_reason_counts":rejects,
        "same_contract_research_only":paired,
    }


def read_jsonl(path):
    source=Path(path)
    if not source.exists():
        return []
    return [json.loads(x) for x in source.read_text(encoding="utf8").splitlines() if x.strip()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecasts",type=Path,default=Path("history/moneyline_elo_v1_2026.jsonl"))
    ap.add_argument("--quotes",type=Path,default=Path("history/public_aggregator_quotes_2026.jsonl"))
    ap.add_argument("--report",type=Path,default=Path("reports/moneyline_forward_public_price_shadow.json"))
    args=ap.parse_args()
    result=pair_originals(read_jsonl(args.forecasts),read_jsonl(args.quotes))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in (
        "decision","moneyline_frozen_games","aggregator_observations_scanned",
        "postmodel_first_capture_game_pairs","accepted_BetMGM_US_account_offers",
        "validated_profitable_bets","reject_reason_counts")},indent=2))


if __name__=="__main__":
    main()
