"""Attach strictly matched, first-seen NFL prop shadows to contract research.

The historical file contains 2026 original pregame probability curves, not
sportsbook lines. This is an independent *research* probability only. Existing
2025 historical synthetic-threshold scores do NOT validate BetMGM profit.
"""
from __future__ import annotations
import argparse
import csv
from datetime import timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from scripts.contract_first_scanner import screen
from scripts.professional_method_audit import _utc

VERSION="FROZEN_PROP_TO_CONTRACT_SHADOW_V1"
SOURCE_MARKETS={
    "receiving_yards":"PLAYER_RECEIVING_YARDS",
    "rushing_yards":"PLAYER_RUSHING_YARDS",
    "receptions":"PLAYER_RECEPTIONS",
}
SOURCE_VERSION="prop_empirical_error_cdf_v1"


def _line(v):
    try:return Decimal(str(v))
    except (TypeError,InvalidOperation):raise ValueError("malformed numeric threshold") from None


def load_frozen_forecasts(path):
    with Path(path).open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    index={}
    for row in rows:
        if row.get("version")!=SOURCE_VERSION:
            raise ValueError("frozen source version unexpected")
        if row.get("market") not in SOURCE_MARKETS:
            continue  # passing-yards contract mapping not yet independently audited
        key=(row["game_id"],SOURCE_MARKETS[row["market"]],
             row["player_id"],_line(row["synthetic_line"]))
        if key in index:
            raise ValueError("duplicate frozen player/game/threshold")
        first=_utc(row["first_observed_at_utc"])
        kickoff=_utc(row["kickoff_utc"])
        if first>=kickoff:
            raise ValueError("first-seen record occurred post-kickoff")
        over=float(row["probability_over"])
        under=float(row["probability_under"])
        if not (0<=over<=1 and 0<=under<=1 and abs(over+under-1)<1e-7):
            raise ValueError("invalid frozen probability pair")
        index[key]=row
    return index


def join(document,index):
    if not isinstance(document,dict) or not isinstance(document.get("markets",[]),list):
        raise ValueError("markets input must be a list")
    at=_utc(document["decision_utc"])
    enriched=json.loads(json.dumps(document))
    matched=[]
    missed=[]
    for market in enriched.get("markets",[]):
        if market.get("independent_model") is not None:
            raise ValueError("refusing to overwrite provided independent model")
        try:
            quotes=market["quotes"]
            if not isinstance(quotes,list) or len(quotes)!=2:raise ValueError
            selected=market["selected"]
            q=next(x for x in quotes if x["selection"]==selected)
            key=(q["event_id"],q["market"],q["participant"],_line(q["line"]))
            frozen=index.get(key)
            if frozen is None:
                missed.append({"event_id":q["event_id"],"selection":selected,
                               "reason":"NO_FROZEN_EXACT_PLAYER_MARKET_THRESHOLD"})
                continue
            freeze_time=_utc(frozen["first_observed_at_utc"])
            kickoff=_utc(frozen["kickoff_utc"])
            if kickoff!=_utc(q["kickoff_utc"]):
                raise ValueError("frozen/game quote kickoff mismatch")
            if not (freeze_time<=_utc(q["quote_utc"])<=at<kickoff):
                raise ValueError("forecast not frozen before quote, or game already started")
            p=float(frozen["probability_over"] if selected=="OVER" else frozen["probability_under"])
            if selected not in {"OVER","UNDER"}:
                raise ValueError("player shadow uses over/under only")
            market["independent_model"]={
                "source_kind":"INDEPENDENT_FORECAST",
                "model_version":SOURCE_VERSION,
                "event_id":q["event_id"],"market":q["market"],
                "period":q["period"],"participant":q["participant"],
                "selection":selected,"line":q["line"],
                "forecast_utc":freeze_time.isoformat(),
                "p_win":p,"p_push":0.0,"p_loss":1-p,
                "p_win_lower_95":None,
                "validation_evidence_id":"2025_ALREADY_INSPECTED_SYNTHETIC_THRESHOLD_DIAGNOSTIC",
                "player_name":frozen["player_name"],
                "player_id":frozen["player_id"],
                "source_probability_only":"FROZEN_FIRST_SEEN_NO_BOOK_ODDS",
            }
            matched.append({"event_id":q["event_id"],"player_id":q["participant"],
                            "market":q["market"],"line":q["line"],"selection":selected})
        except (KeyError,StopIteration,ValueError,TypeError) as e:
            missed.append({"event_id":market.get("quotes",[{}])[0].get("event_id"),
                           "reason":"INCOMPATIBLE_QUOTE_OR_TIME: "+str(e)})
    report=screen(enriched)
    report["adapter_version"]=VERSION
    report["frozen_prop_matches"]=matched
    report["frozen_prop_unmatched"]=missed
    report["source_model_not_approved_for_betting"]=True
    report["source_csv_contains_no_verified_book_prices"]=True
    report["validation_2025_not_pristine"]=True
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quotes",type=Path,required=True)
    p.add_argument("--frozen",type=Path,
                   default=Path("history/player_prop_probability_forward.csv"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=join(json.loads(a.quotes.read_text(encoding="utf8")),
                load_frozen_forecasts(a.frozen))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(f"research only | matches {len(result['frozen_prop_matches'])} | decision PASS")
if __name__=="__main__":
    main()
