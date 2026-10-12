"""NFL market capture-only same-contract cross-book price disagreement screen.

This uses free, *observed-at* public aggregator quotes, NOT source-origin-time-
verified or account-accepted sportsbook wagers. No independent win-probability
model. A reference-book no-vig difference is a *proxy*, never true bet EV.
Conservative market proxy uses the weakest comparator probability among
distinct identifiable books quoting the identical threshold. No market version
or betting approval is changed.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path

VERSION="NFL_CROSSBOOK_CAPTURE_PROXY_V1"
MIN_OTHER_BOOKS=3
MAX_REFERENCE_SPREAD=.025
WATCH_PROXY_THRESHOLD=.05
TARGET_BOOK="BetMGM"


def _decimal(odds):
    if isinstance(odds,bool) or not isinstance(odds,int) or (
        -100 < odds < 100 or odds==0
    ):
        raise ValueError("invalid observed American odds")
    return 1+odds/100 if odds>0 else 1+100/abs(odds)


def _normalized(x):
    first,second=x["first_american_odds"],x["second_american_odds"]
    a,b=1/_decimal(first),1/_decimal(second)
    return a/(a+b),b/(a+b)


def _exact_line(market,entry):
    if market=="moneyline":
        if entry["first_line"] is not None or entry["second_line"] is not None:
            raise ValueError("moneyline includes line")
        return "NA"
    a,b=float(entry["first_line"]),float(entry["second_line"])
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("bad finite threshold")
    if market=="spread":
        if abs(a+b)>1e-9:
            raise ValueError("mismatched signed spread threshold")
        return a
    if market=="total":
        if abs(a-b)>1e-9:
            raise ValueError("total line mismatch")
        return a
    raise ValueError("unsupported full-game market")


def evaluate_capture(snapshot):
    if snapshot.get("league")!="NFL" or snapshot.get("betting_decision")!="PASS":
        raise ValueError("expected previously captured NFL research source")
    if not snapshot.get("collector_observed_utc") or not snapshot.get("kickoff_utc"):
        raise ValueError("original observation and kickoff required")
    if snapshot["collector_observed_utc"]>=snapshot["kickoff_utc"]:
        raise ValueError("postkickoff observation cannot enter pregame watch")
    grouped=defaultdict(list)
    exclusions=[]
    for quote in snapshot.get("quotes",[]):
        try:
            market=str(quote["market"])
            line=_exact_line(market,quote)
            if str(quote["first_side"]) != (
                "over" if market=="total" else "home"
            ) or str(quote["second_side"]) != (
                "under" if market=="total" else "away"
            ):
                raise ValueError("side identity mismatch")
            if not isinstance(quote["book_as_attributed_by_aggregator"],str):
                raise ValueError("unnamed book")
            if quote.get("source_original_book_offer_update_utc") is not None:
                # Source-origin verified quotes belong in stronger separately
                # regulated workflow, never silently commingle.
                raise ValueError("strict-source quote belongs to another protocol")
            grouped[(market,line)].append(quote)
        except (KeyError,ValueError,TypeError) as e:
            exclusions.append(str(e)[:140])
    market_results=[]
    for (market,line),entries in sorted(grouped.items(),key=lambda x:(x[0][0],str(x[0][1]))):
        target=[x for x in entries if x["book_as_attributed_by_aggregator"]==TARGET_BOOK]
        if len(target)!=1:
            continue
        reference={}
        for x in entries:
            book=x["book_as_attributed_by_aggregator"]
            if book==TARGET_BOOK or book.startswith("ActionNetwork book "):
                continue
            if book in reference:
                raise ValueError("duplicate independent sportsbook in exact line group")
            reference[book]=_normalized(x)
        tgt=target[0]
        for side_idx,side in enumerate((
            ("OVER","UNDER") if market=="total" else ("HOME","AWAY")
        )):
            odds=tgt["first_american_odds"] if side_idx==0 else tgt["second_american_odds"]
            reference_probs=[x[side_idx] for x in reference.values()]
            base={
                "event_id":snapshot["event_id"],"market":market,
                "line":line,"selection":side,
                "betmgm_attributed_american_odds":odds,
                "captured_utc":snapshot["collector_observed_utc"],
                "reference_named_book_count":len(reference),
                "reference_names":sorted(reference),
                "source_origin_timestamps_verified":False,
                "betmgm_account_offer_verified":False,
                "independently_validated_probability":None,
                "p_push":None if (
                    market=="moneyline" or (
                        isinstance(line,(int,float)) and float(line).is_integer()
                    )
                ) else 0.0,
                "model_ev":None,
                "betting_decision":"PASS",
                "same_contract_market_proxy":None,
                "research_watch":False,
            }
            if len(reference_probs)>=MIN_OTHER_BOOKS:
                lo,hi=min(reference_probs),max(reference_probs)
                if hi-lo<=MAX_REFERENCE_SPREAD:
                    conditional_proxy=_decimal(odds)*lo-1
                    base["same_contract_market_proxy"]={
                        "reference_min_conditional_nonpush":round(lo,7),
                        "reference_max_conditional_nonpush":round(hi,7),
                        "reference_probability_span":round(hi-lo,7),
                        "bookmaker_only_hypothetical_proxy":round(conditional_proxy,7),
                        "not_true_expected_value":True,
                        "multiple_testing_warning":True,
                    }
                    base["research_watch"]=conditional_proxy>=WATCH_PROXY_THRESHOLD
                else:
                    base["exclusion_reason"]="REFERENCE_BOOKS_DISAGREE_TOO_WIDELY"
            else:
                base["exclusion_reason"]="FEWER_THAN_3_INDEPENDENT_SAME_LINE_BOOKS"
            market_results.append(base)
    return market_results,exclusions


def audit(snapshots):
    if not isinstance(snapshots,list):
        raise ValueError("snapshots must be list")
    by_event={}
    for x in snapshots:
        event=x["event_id"]
        if event not in by_event or x["collector_observed_utc"]>by_event[event]["collector_observed_utc"]:
            by_event[event]=x
    records=[]
    exclusions=[]
    for event,snap in sorted(by_event.items()):
        try:
            matched,err=evaluate_capture(snap)
            records.extend(matched)
            exclusions.extend(err)
        except (KeyError,TypeError,ValueError) as e:
            exclusions.append(f"{event}: {type(e).__name__}: {e}")
    ranked=sorted(
        (x for x in records if x["research_watch"]),
        key=lambda x:x["same_contract_market_proxy"]["bookmaker_only_hypothetical_proxy"],
        reverse=True,
    )
    return {
        "version":VERSION,"status":"MARKET_CAPTURE_ONLY_PROSPECTIVE_DIAGNOSTIC",
        "decision":"PASS","bet_recommendations":[],
        "best_validated_betmgm_edge":None,
        "distinct_events_checked":len(by_event),
        "book_attributed_selections_considered":len(records),
        "comparable_same_threshold_selections":sum(
            x["same_contract_market_proxy"] is not None for x in records
        ),
        "market_proxy_research_watches":len(ranked),
        "watchlist":ranked,
        "invalid_source_row_exclusions":exclusions[:50],
        "original_book_price_update_times_verified":False,
        "account_accepted_odds_verified":False,
        "true_ev_calibrated":False,
        "note":"Reference prices are correlated and have no original offer-update timestamps. Conditional no-push probability is not true three-way win likelihood or profitable bet EV.",
    }


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal",type=Path,
                   default=Path("history/public_aggregator_quotes_2026.jsonl"))
    p.add_argument("--report",type=Path,
                   default=Path("reports/crossbook_research_watch.json"))
    args=p.parse_args()
    history=[json.loads(s) for s in args.journal.read_text(encoding="utf8").splitlines()
             if s.strip()]
    out=audit(history)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps({
        "decision":out["decision"],"events":out["distinct_events_checked"],
        "same_line_comparisons":out["comparable_same_threshold_selections"],
        "proxy_only_research_watches":out["market_proxy_research_watches"],
        "verified_profitable_bets":0,
        "report":str(args.report),
    },sort_keys=True))
if __name__=="__main__":
    main()
