"""Point-in-time bookmaker market reference for NFL spread/total forecast audit.

This adopts a demonstrably stronger *reference forecast* (market prices), not
a supposed profitable betting strategy. One book is not a cross-book consensus.
Quotes must be contemporaneous, valid two-way markets before kickoff. All
figures are research-only. No market inputs enter independent football ratings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from scripts.professional_method_audit import fair_snapshot, _utc

VERSION = "MARKET_REFERENCE_V1"
REQUIRED = {"GAME_SPREAD", "GAME_TOTAL"}


def market_reference(spread_pair, total_pair, *, independent=None):
    """Return a paired-price reference or a fail-closed block.

    Spread-line convention: HOME_SIGNED is the handicap given to the
    HOME selection (e.g. -3 home favorite); negative of that is the
    reference HOME expected win margin.
    """
    report = {
        "version": VERSION,
        "status": "BLOCKED",
        "decision": "PASS",
        "market_reference": None,
        "model_disagreement": None,
        "bet_authorized": False,
        "validated_profitable_edge": False,
        "blockers": [],
        "note": "Observed market reference, not independent prediction or betting EV",
    }
    try:
        if len(spread_pair) != 2 or len(total_pair) != 2:
            raise ValueError("spread and total both require two complementary quotes")
        s = fair_snapshot(*spread_pair)
        t = fair_snapshot(*total_pair)
        if {s["contract"]["market"], t["contract"]["market"]} != REQUIRED:
            raise ValueError("both GAME_SPREAD and GAME_TOTAL required")
        # The source cannot be a mix of unrelated books, countries, kickoff
        # dates, periods, capture times or independently quoted snapshots.
        for fld in ("event_id", "kickoff_utc", "period"):
            if s["contract"][fld] != t["contract"][fld]:
                raise ValueError("spread and total refer to different events or periods")
        if s["contract"]["period"] != "FULL_GAME":
            raise ValueError("only full-game market reference supported")
        if (s["bookmaker"] != t["bookmaker"] or
                spread_pair[0]["region"] != total_pair[0]["region"]):
            raise ValueError("reference requires the same bookmaker and region")
        if _utc(s["quote_utc"]) != _utc(t["quote_utc"]):
            raise ValueError("spread and total observation timestamps differ")
        if _utc(s["captured_utc"]) != _utc(t["captured_utc"]):
            raise ValueError("spread and total capture timestamps differ")
        home_spread = float(s["contract"]["line"])
        total_line = float(t["contract"]["line"])
        if not math.isfinite(home_spread) or not math.isfinite(total_line):
            raise ValueError("market lines must be finite")
        if s["contract"]["participant"] != "GAME" or t["contract"]["participant"] != "GAME":
            raise ValueError("game markets must have GAME participant")
        source_types={s["source_type"], t["source_type"]}
        market_status=(
            "ACCOUNT_OBSERVED_NOT_ACCEPTANCE_VERIFIED"
            if source_types=={"ACCOUNT_OBSERVED_UNVERIFIED"}
            else "PUBLIC_OR_EDITORIAL_PROVISIONAL_NOT_EXECUTABLE"
        )
        margin = -home_spread
        fair = {
            "event_id": s["contract"]["event_id"],
            "kickoff_utc": s["contract"]["kickoff_utc"],
            "quote_utc": s["quote_utc"],
            "captured_utc": s["captured_utc"],
            "bookmaker": s["bookmaker"],
            "region": spread_pair[0]["region"],
            "market_source_status": market_status,
            "home_expected_margin_reference": margin,
            "game_total_reference": total_line,
            "home_points_reference": (total_line+margin)/2,
            "away_points_reference": (total_line-margin)/2,
            "spread_conditional_nonpush_no_vig":s["no_vig_conditional_nonpush"],
            "total_conditional_nonpush_no_vig":t["no_vig_conditional_nonpush"],
            "spread_snapshot_id":s["snapshot_id"],
            "total_snapshot_id":t["snapshot_id"],
            "reference_is_independent_model":False,
            "reference_is_cross_book_consensus":False,
            "reference_is_validated_betmgm_executable_quote":False,
        }
        report["market_reference"]=fair
        report["status"]="RESEARCH_MARKET_REFERENCE_ONLY"
        if independent is not None:
            if not isinstance(independent,dict):
                raise ValueError("independent score model must be a dict")
            required=("projected_home_margin","projected_total","forecast_utc")
            if any(k not in independent for k in required):
                raise ValueError("model score and frozen timestamp required")
            if _utc(independent["forecast_utc"]) > _utc(s["captured_utc"]):
                raise ValueError("model forecast may not be created after captured reference")
            m=float(independent["projected_home_margin"])
            t_model=float(independent["projected_total"])
            if not math.isfinite(m) or not math.isfinite(t_model):
                raise ValueError("non-finite independent model scores")
            report["model_disagreement"]={
                "independent_home_margin":m,
                "independent_total":t_model,
                "margin_vs_observed_market_reference":m-margin,
                "total_vs_observed_market_reference":t_model-total_line,
                "disagreement_is_not_a_positive_ev_signal":True,
                "independent_model_version":independent.get("model_version","UNREGISTERED"),
            }
        report["blockers"].extend([
            "NO_INDEPENDENT_PROSPECTIVE_INSTRUMENT_EV_VALIDATION",
            "NO_EXECUTABLE_ACCOUNT_ACCEPTANCE_VERIFICATION",
        ])
    except (ValueError,TypeError,KeyError) as e:
        report["status"]="BLOCKED"
        report["market_reference"]=None
        report["model_disagreement"]=None
        report["blockers"].append(str(e))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text(encoding="utf-8"))
    report=market_reference(
        data["spread_pair"],data["total_pair"],
        independent=data.get("independent")
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("PASS — research-only reference; no inferred betting edge")


if __name__=="__main__":
    main()
