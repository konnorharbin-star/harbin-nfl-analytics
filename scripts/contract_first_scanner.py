"""Contract-first NFL pricing research: market reference plus independent probabilities.

This is the *replacement decision interface* for legacy score-gap-driven
picks. Individual quotes must be exactly matched, timestamped pregame
two-sided markets. With no approved independent calibration, output PASS,
not a fake BetMGM edge. Supports player props (only with real model inputs).
No sports scores, account access, odds API, or paid data are required.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from scripts.professional_method_audit import _utc, fair_snapshot, decimal_odds

VERSION = "CONTRACT_FIRST_RESEARCH_V1"
# No empirically approved independent NFL betting models currently exist.
# New models can only be registered after chronological evaluation plus
# prospective calibration and independent readback of a versioned policy.
APPROVED_MODEL_VERSIONS: frozenset[str] = frozenset()
MAX_QUOTE_AGE_MINUTES = 15


def _number(x, name):
    if isinstance(x, bool):
        raise ValueError(name + " must not be boolean")
    try:
        v=float(x)
    except (TypeError, ValueError):
        raise ValueError("invalid " + name) from None
    if not math.isfinite(v):
        raise ValueError("non-finite " + name)
    return v


def _contract_match(quote, model):
    for key in ("event_id","market","period","participant","selection","line"):
        if str(quote[key]) != str(model.get(key)):
            raise ValueError("independent probability is not the exact selected contract: "+key)


def _model_diagnostic(model, quote, decision_time):
    """Diagnostic not approval; conservative result assumes push fixed."""
    _contract_match(quote,model)
    if model.get("source_kind")!="INDEPENDENT_FORECAST":
        raise ValueError("expert pick or market-only source not an independent probability")
    forecast=_utc(model["forecast_utc"])
    if forecast > decision_time or forecast >= _utc(quote["kickoff_utc"]):
        raise ValueError("forecast is future or post-kickoff")
    win=_number(model["p_win"],"p_win")
    push=_number(model["p_push"],"p_push")
    loss=_number(model["p_loss"],"p_loss")
    if min(win,push,loss)<0 or max(win,push,loss)>1 or not math.isclose(
        win+push+loss,1,rel_tol=0,abs_tol=1e-8
    ):
        raise ValueError("invalid win/push/loss simplex")
    threshold=_number(quote["line"],"line") if quote["market"]!="GAME_MONEYLINE" else None
    if threshold is not None and not math.isclose(threshold,round(threshold),abs_tol=1e-10) and push>1e-10:
        # NFL yards/TD counts and scores are integers; noninteger half-point
        # threshold (or any noninteger threshold) cannot settle as a PUSH.
        raise ValueError("fractional NFL threshold cannot push")
    lower=model.get("p_win_lower_95")
    if lower is not None:
        lower=_number(lower,"p_win_lower_95")
        if not 0<=lower<=win:
            raise ValueError("conservative bound must be within [0,p_win]")
    dec=decimal_odds(quote["american_odds"])
    net=dec-1
    ev=win*net-loss
    conservative=lower*net-(1-push-lower) if lower is not None else None
    mv=str(model.get("model_version",""))
    version_approved=mv in APPROVED_MODEL_VERSIONS
    return {
        "model_version":mv,
        "model_source":"INDEPENDENT_FORECAST_UNVALIDATED_BY_THIS_SCANNER",
        "p_win":win,"p_push":push,"p_loss":loss,
        "p_win_lower_95":lower,
        "diagnostic_ev_per_dollar":round(ev,9),
        "diagnostic_conservative_ev_per_dollar":
            round(conservative,9) if conservative is not None else None,
        "model_version_independently_approved":version_approved,
        "validation_evidence_id":model.get("validation_evidence_id"),
        "betting_recommendation":False,
        "calibration_has_not_been_independently_verified_here":True,
    }


def _one_market(market, at):
    row={
        "status":"BLOCKED","decision":"PASS",
        "event_id":None,"market":None,"selected":None,
        "conditional_nonpush_market_probability":None,
        "market_provenance":None,"independent_probability":None,
        "rank_score":None,"verified_edge":False,"approved_wager":False,
        "blockers":[],
    }
    try:
        quotes=market["quotes"]
        if not isinstance(quotes,list) or len(quotes)!=2:
            raise ValueError("two exact complementary quotes required")
        a,b=quotes
        fair=fair_snapshot(a,b)
        selection=market["selected"]
        if selection not in {a["selection"],b["selection"]}:
            raise ValueError("selected outcome does not appear in quote pair")
        quote=a if a["selection"]==selection else b
        row.update({
            "event_id":a["event_id"],"market":a["market"],
            "selected":selection,
            "contract":fair["contract"],
            "price_american":quote["american_odds"],
            "market_provenance":{
                "bookmaker":quote["bookmaker"],"region":quote["region"],
                "quote_utc":quote["quote_utc"],
                "captured_utc":quote["captured_utc"],
                "source_type":quote["source_type"],
                "source_url":quote["source_url"],
                "snapshot_id":quote["snapshot_id"],
                "account_executable_verified":False,
            },
            "conditional_nonpush_market_probability":
                fair["no_vig_conditional_nonpush"][selection],
            "no_vig_probability_does_not_supply_push_probability":True,
        })
        qt=_utc(quote["quote_utc"])
        capture=_utc(quote["captured_utc"])
        kickoff=_utc(quote["kickoff_utc"])
        if capture>at or qt>at:
            raise ValueError("quote/capture recorded later than research decision")
        if not at<kickoff:
            raise ValueError("game already kicked off")
        if at-qt>timedelta(minutes=MAX_QUOTE_AGE_MINUTES):
            row["blockers"].append("QUOTE_STALE_MORE_THAN_15_MINUTES")
        if quote["bookmaker"].casefold()!="betmgm":
            row["blockers"].append("NOT_BETMGM_BOOK")
        if quote["region"]!="US":
            row["blockers"].append("US_ACCOUNT_REGION_NOT_VERIFIED")
        # Even a public BetMGM-looking page cannot certify a live account offer.
        row["blockers"].append("BETMGM_PRICE_UNVERIFIED_FOR_ACCOUNT_ACCEPTANCE")
        mod=market.get("independent_model")
        if mod is None:
            row["blockers"].append("NO_INDEPENDENT_CONTRACT_PROBABILITY_MODEL")
        else:
            diag=_model_diagnostic(mod,quote,at)
            row["independent_probability"]=diag
            if not diag["model_version_independently_approved"]:
                row["blockers"].append("MODEL_NOT_APPROVED_BY_INDEPENDENT_FORWARD_TEST")
            if diag["p_win_lower_95"] is None:
                row["blockers"].append("NO_VALIDATED_LOWER_PROBABILITY_BOUND")
            elif diag["diagnostic_conservative_ev_per_dollar"]<=0:
                row["blockers"].append("CONSERVATIVE_EV_NOT_POSITIVE")
        row["status"]="RESEARCH_ONLY"
    except (ValueError,KeyError,TypeError,OverflowError) as e:
        row["blockers"].append(str(e))
        row["independent_probability"]=None
        row["status"]="BLOCKED"
    return row


def screen(document):
    """One record per exact contract; results never imply automatic bets."""
    if not isinstance(document,dict):
        raise ValueError("input JSON must be an object")
    at=_utc(document["decision_utc"])
    markets=document.get("markets",[])
    if not isinstance(markets,list):
        raise ValueError("markets must be an array")
    rows=[_one_market(m,at) for m in markets]
    # Never rank unverified diagnostic EV as a verified betting recommendation.
    diagnostic=[x for x in rows if x.get("independent_probability") is not None]
    diagnostic.sort(key=lambda x:x["independent_probability"]["diagnostic_ev_per_dollar"],reverse=True)
    return {
        "version":VERSION,"decision_utc":at.isoformat(),
        "engine_role":"CONTRACT_FIRST_NFL_RESEARCH_NOT_LEGACY_SCORE_GAP",
        "legacy_score_model":"RETIRED_AS_PRIMARY_BETTING_SIGNAL_BENCHMARK_ONLY",
        "screened_contracts":len(rows),"decision":"PASS",
        "best_verified_edge":None,
        "recommending_bet":False,
        "independently_approved_models":sorted(APPROVED_MODEL_VERSIONS),
        "provisional_only_diagnostic_contracts":[
            {"event_id":r["event_id"],"market":r["market"],
             "selected":r["selected"],
             "diagnostic_ev_per_dollar":r["independent_probability"]["diagnostic_ev_per_dollar"],
             "approved":False} for r in diagnostic
        ],
        "markets":rows,
        "blockers_global":[
            "NO_APPROVED_INDEPENDENT_CONTRACT_LEVEL_PROBABILITY_MODEL",
            "NO_VERIFIED_ACCOUNT_EXECUTABLE_BETMGM_PRICES",
            "NO_PROVEN_POSITIVE_EV_OR_REAL_MONEY_BETS",
        ],
        "note":"Research output must never be treated as a profitable recommendation.",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=screen(json.loads(args.input.read_text(encoding="utf8")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"version":VERSION,"contracts":result["screened_contracts"],
                      "decision":result["decision"],"verified_edge":None}))
if __name__=="__main__":
    main()
