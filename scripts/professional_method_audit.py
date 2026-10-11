"""Professional-method market audit: exact-contract, point-in-time diagnostics.

Inspired by PUBLIC methods (Walters: independent prices/information; Andrews:
line shopping/closing lines; Koerner: granular props). No proprietary model is
reconstructed. Pure standard-library calculations. PASS is ALWAYS mandatory
until an independent betting-EV validation and account-price verification
process exists. This is not a wagering engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

OPPOSITES = {
    "GAME_SPREAD": {"HOME", "AWAY"},
    "GAME_MONEYLINE": {"HOME", "AWAY"},
    "GAME_TOTAL": {"OVER", "UNDER"},
    "PLAYER_RECEIVING_YARDS": {"OVER", "UNDER"},
    "PLAYER_RUSHING_YARDS": {"OVER", "UNDER"},
    "PLAYER_RECEPTIONS": {"OVER", "UNDER"},
    "PLAYER_ANYTIME_TD": {"YES", "NO"},
}
PROVENANCE = {"EDITORIAL", "AGGREGATOR", "PUBLIC_BOOK_PAGE",
              "ACCOUNT_OBSERVED_UNVERIFIED"}
CONTRACT = ("event_id", "kickoff_utc", "market", "period",
            "participant", "line")
SNAPSHOT = CONTRACT + ("bookmaker", "region", "snapshot_id",
                       "quote_utc", "source_type")


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("timestamp needs an explicit UTC offset")
    return ts.astimezone(timezone.utc)


def decimal_odds(value):
    if isinstance(value, bool):
        raise ValueError("invalid American odds")
    try:
        price = int(value)
    except (ValueError, TypeError):
        raise ValueError("invalid American odds") from None
    if str(value).strip() not in (str(price), f"+{price}"):
        raise ValueError("American odds must be an integer")
    if -100 < price < 100:
        raise ValueError("American odds must be <= -100 or >= +100")
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def _check_line(quote):
    m = quote["market"]
    line = quote["line"]
    if m == "GAME_MONEYLINE":
        if line != "NA":
            raise ValueError("moneyline must have line NA")
    else:
        try:
            x = float(line)
        except (ValueError, TypeError):
            raise ValueError("line must be numeric") from None
        if not math.isfinite(x):
            raise ValueError("line must be finite")
    # HOME spread is always expressed as a HOME-centric signed threshold,
    # even when the selected side is AWAY. Don't compare selections entered
    # in two different sign conventions.
    if m == "GAME_SPREAD" and quote.get("line_convention") != "HOME_SIGNED":
        raise ValueError("spread requires line_convention HOME_SIGNED")


def _validate_quote(quote):
    needed = set(SNAPSHOT) | {"selection", "american_odds",
                              "captured_utc", "source_url"}
    if any(k not in quote or quote[k] is None or quote[k] == "" for k in needed):
        raise ValueError("missing required quote provenance")
    if quote["market"] not in OPPOSITES:
        raise ValueError("unsupported market")
    if quote["source_type"] not in PROVENANCE:
        raise ValueError("unsupported quote source")
    if quote["selection"] not in OPPOSITES[quote["market"]]:
        raise ValueError("invalid selection")
    if not str(quote["source_url"]).startswith(("https://", "http://")):
        raise ValueError("source URL required")
    _check_line(quote)
    decimal_odds(quote["american_odds"])
    q, c, k = (_utc(quote[v]) for v in ("quote_utc", "captured_utc", "kickoff_utc"))
    if not q <= c < k:
        raise ValueError("quotes must be sourced and captured before kickoff")
    return q


def _key(quote, keys):
    return tuple(str(quote[k]) for k in keys)


def fair_snapshot(a, b):
    """No-vig shares CONDITIONAL on outcome not pushing/voiding.

    Caller must provide the same exact line, snapshot, book and period;
    HOME_SIGNED normalizes spread directions. No-vig proportions do not
    estimate absolute P(push).
    """
    ta = _validate_quote(a)
    tb = _validate_quote(b)
    if _key(a, SNAPSHOT) != _key(b, SNAPSHOT) or ta != tb:
        raise ValueError("not the same two-sided market snapshot")
    if {a["selection"], b["selection"]} != OPPOSITES[a["market"]]:
        raise ValueError("not complementary outcomes")
    ra = 1 / decimal_odds(a["american_odds"])
    rb = 1 / decimal_odds(b["american_odds"])
    return {
        "no_vig_conditional_nonpush": {a["selection"]: ra / (ra + rb),
                                      b["selection"]: rb / (ra + rb)},
        "overround": ra + rb - 1,
        "contract": {k: a[k] for k in CONTRACT},
        "snapshot_id": a["snapshot_id"],
        "bookmaker": a["bookmaker"],
        "source_type": a["source_type"],
        "quote_utc": a["quote_utc"],
        "captured_utc": a["captured_utc"],
    }


def _probability_audit(model, quote):
    if model is None:
        return None, "NO_INDEPENDENT_CALIBRATED_PROBABILITIES"
    if model.get("source_kind") != "INDEPENDENT_MODEL":
        return None, "EXPERT_OPINION_IS_NOT_A_CALIBRATED_PROBABILITY"
    try:
        probabilities = [float(model[k]) for k in ("p_win", "p_push", "p_loss")]
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in probabilities):
            raise ValueError
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError
        if _utc(model["forecast_utc"]) > _utc(quote["captured_utc"]):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return None, "INVALID_OR_FUTURE_MODEL_FORECAST"
    p_win, p_push, p_loss = probabilities
    profit = decimal_odds(quote["american_odds"]) - 1
    return {
        "p_win": p_win, "p_push": p_push, "p_loss": p_loss,
        "conditional_break_even_win": (1 - p_push) / (1 + profit),
        "diagnostic_ev_per_dollar": p_win * profit - p_loss,
        "model_validation_claim": model.get("validation_status", "UNVALIDATED"),
        "validation_evidence_id": model.get("validation_evidence_id"),
        "warning": "INPUT PROBABILITIES NOT VERIFIED BY THIS TOOL; NOT BET APPROVAL",
    }, None


def audit(entry_a, entry_b, *, selected, close_a=None, close_b=None, model=None, offers=None):
    """All failures are fail-closed PASS with explicit blockers."""
    result = {
        "version": "professional-method-audit-v1",
        "decision": "PASS", "bet_authorized": False,
        "execution_quote_verified": False,
        "edge_validated": False, "entry_market": None, "model": None,
        "closing_line_value": None, "price_shopping": None, "blockers": [],
        "professional_methods": [
            "independent_score_model_required",
            "asof_news_and_starter_verification_required",
            "exact_line_and_price_shopping",
            "granular_player_usage_model_required_for_props",
            "analyst_picks_are_features_to_test_not_forecasts",
        ],
    }
    try:
        fair = fair_snapshot(entry_a, entry_b)
        if selected not in {entry_a["selection"], entry_b["selection"]}:
            raise ValueError("selected side absent from snapshot")
        selected_entry = (entry_a if entry_a["selection"] == selected else entry_b)
        result["entry_market"] = fair
        result["selected"] = selected
        result["entry_decimal_odds"] = decimal_odds(selected_entry["american_odds"])
        model_result, blocker = _probability_audit(model, selected_entry)
        result["model"] = model_result
        if blocker:
            result["blockers"].append(blocker)
        if fair["source_type"] != "ACCOUNT_OBSERVED_UNVERIFIED":
            result["blockers"].append("EDITORIAL_OR_PUBLIC_PRICE_NOT_EXECUTABLE")
        else:
            result["blockers"].append("ACCOUNT_QUOTE_STILL_REQUIRES_INDEPENDENT_ACCEPTANCE_CHECK")
        if offers is not None:
            result["price_shopping"] = compare_same_contract_prices(offers, selected)
        if (close_a is None) != (close_b is None):
            raise ValueError("closing snapshot must provide BOTH market sides")
        if close_a is not None:
            closing = fair_snapshot(close_a, close_b)
            if _key(entry_a, CONTRACT) != _key(close_a, CONTRACT):
                raise ValueError("closing contract differs: CLV not computable")
            if _utc(closing["quote_utc"]) < _utc(fair["quote_utc"]):
                raise ValueError("closing quote earlier than entry quote")
            if _utc(closing["captured_utc"]) < _utc(fair["captured_utc"]):
                raise ValueError("closing capture earlier than entry capture")
            close_conditional = closing["no_vig_conditional_nonpush"][selected]
            # CLV is valid only for EXACT same selection and threshold.
            result["closing_line_value"] = {
                "price_ratio_minus_one": result["entry_decimal_odds"] *
                                         close_conditional - 1,
                "closing_fair_conditional_probability_nonpush": close_conditional,
                "close_snapshot": closing["snapshot_id"],
                "benchmark_not_profit_proof": True,
            }
    except (ValueError, KeyError, TypeError) as exc:
        result["blockers"].append(str(exc))
    result["blockers"].extend([
        "NO_VERIFIED_EXECUTABLE_US_BETMGM_ACCOUNT_QUOTE",
        "NO_PROSPECTIVE_UNTOUCHED_CALIBRATION_AND_PROFIT_VALIDATION",
    ])
    return result



def compare_same_contract_prices(offers, selected):
    """Rank archived prices ONLY when they are synchronous exact contracts.

    This is price-shopping RESEARCH, not evidence a BetMGM account can
    accept any price or that an independent model has positive EV.
    """
    if not isinstance(offers, list) or len(offers) < 2:
        raise ValueError("need at least two independently sourced offers")
    for offer in offers:
        _validate_quote(offer)
        if offer["selection"] != selected:
            raise ValueError("offers must name the identical selected side")
    base = offers[0]
    fingerprints = set()
    for offer in offers:
        if _key(offer, CONTRACT) != _key(base, CONTRACT):
            raise ValueError("cross-book offers are different contracts")
        if _utc(offer["quote_utc"]) != _utc(base["quote_utc"]):
            raise ValueError("cross-book offers are not simultaneous")
        ident = (offer["bookmaker"], offer["region"])
        if ident in fingerprints:
            raise ValueError("duplicate book-region offers")
        fingerprints.add(ident)
    ordered = sorted(offers, key=lambda x: decimal_odds(x["american_odds"]),
                     reverse=True)
    return {
        "status": "RESEARCH_ONLY_PRICE_RANKING",
        "selection": selected,
        "quote_utc": base["quote_utc"],
        "contract": {k: base[k] for k in CONTRACT},
        "ranked_offers": [
            {"bookmaker": o["bookmaker"], "region": o["region"],
             "source_type": o["source_type"], "american_odds": o["american_odds"],
             "decimal_odds": decimal_odds(o["american_odds"]),
             "source_url": o["source_url"]}
            for o in ordered
        ],
        "decimal_price_improvement_over_worst":
            decimal_odds(ordered[0]["american_odds"]) -
            decimal_odds(ordered[-1]["american_odds"]),
        "execution_verified": False,
        "edge_validated": False,
        "note": "Different prices alone do not establish mispricing or expected value",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="JSON with entry_a, entry_b, selected; optional close_a, close_b, model")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit(data["entry_a"], data["entry_b"], selected=data["selected"],
                   close_a=data.get("close_a"), close_b=data.get("close_b"),
                   model=data.get("model"), offers=data.get("offers"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS; market research audit only; no wager or positive-EV claim.")


if __name__ == "__main__":
    main()
