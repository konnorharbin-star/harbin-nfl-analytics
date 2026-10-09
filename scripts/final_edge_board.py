"""Deterministic NFL/CFB evidence-ranked edge board. No score or stake adjustments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

MAX_QUOTE_AGE_MINUTES = 120
FIELDS = (
    "sport", "rank", "game_id", "kickoff", "away_team", "home_team",
    "market", "side", "line", "american_odds", "book", "quoted_at",
    "quote_age_minutes", "model_probability", "raw_model_ev",
    "conservative_probability", "conservative_ev", "price_cushion_pp",
    "line_cushion_points", "historical_holdout_bets", "evidence_tier",
    "decision", "bet_approved", "reasons",
)
UNKNOWN_BOOK = re.compile(
    r"^(?:\s*|unknown|consensus|book\s*\d+|actionnetwork book\s*\d+)$",
    re.IGNORECASE,
)


def timestamp(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else None
    except (TypeError, ValueError):
        return None


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def yes(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def break_even(odds):
    if odds is None or odds == 0 or abs(odds) < 100:
        return None
    return 100 / (100 + odds) if odds > 0 else -odds / (100 - odds)


def conservative_ev(probability, odds):
    if probability is None or not 0 <= probability <= 1:
        return None
    implied = break_even(odds)
    if implied is None:
        return None
    profit = odds / 100 if odds > 0 else 100 / -odds
    return probability * profit - (1 - probability)


def inspect(row, sport, gate, as_of):
    if sport not in {"nfl", "cfb"}:
        raise ValueError("Unknown sport")
    nfl = sport == "nfl"
    value = lambda n, c: row.get(n if nfl else c)
    kickoff_str = value("kickoff", "date")
    kickoff = timestamp(kickoff_str)
    quote_str = value("quant_quote_at", "quote_at")
    quote = timestamp(quote_str)
    market = str(value("quant_market", "market") or "")
    side = str(value("quant_side", "side") or "")
    odds = number(value("quant_odds", "odds"))
    line = number(value("quant_price", "line"))
    book = str(value("quant_book", "book") or "").strip()
    probability = number(value("quant_probability", "probability"))
    raw_ev = number(value("quant_ev", "ev"))
    reasons = []
    if market not in {"spread", "moneyline", "total"} or not side:
        reasons.append("INVALID_SELECTION")
    if break_even(odds) is None or (market != "moneyline" and line is None):
        reasons.append("INVALID_PRICE")
    if UNKNOWN_BOOK.fullmatch(book):
        reasons.append("UNRESOLVED_BOOK")
    if kickoff is None:
        reasons.append("INVALID_KICKOFF")
    elif kickoff <= as_of:
        reasons.append("GAME_STARTED")
    if quote is None:
        reasons.append("QUOTE_TIME_MISSING")
        quote_age = None
    else:
        quote_age = (as_of - quote).total_seconds() / 60
        if quote_age < -5:
            reasons.append("FUTURE_QUOTE")
        elif quote_age > MAX_QUOTE_AGE_MINUTES:
            reasons.append("STALE_QUOTE")
        if kickoff is not None and quote >= kickoff:
            reasons.append("POST_KICKOFF_QUOTE")
    if probability is None or not 0 <= probability <= 1:
        reasons.append("INVALID_PROBABILITY")
    conservative_probability = None
    conservative_value = None
    price_cushion = None
    line_cushion = None
    holdout_bets = None
    if nfl:
        tier = str(row.get("edge_discovery_tier") or "UNSUPPORTED")
        supported = tier == "SUPPORTED_RESEARCH"
        conservative_value = number(row.get("conservative_ev"))
        if conservative_value is None:
            conservative_value = number(row.get("edge_shrunk_ev"))
        if not supported:
            reasons.append("NO_SUPPORTED_NFL_EDGE")
        if conservative_value is None or conservative_value <= 0:
            reasons.append("NO_POSITIVE_CONSERVATIVE_EV")
        if yes(row.get("context_freshness_veto")):
            reasons.append("STALE_PLAYER_CONTEXT")
        if yes(row.get("qb_certainty_veto")):
            reasons.append("UNCERTAIN_STARTING_QB")
        if not yes(row.get("regime_reliability_ready")):
            reasons.append("UNVALIDATED_REGIME")
        if not yes(row.get("probability_reliability_ready")):
            reasons.append("UNRELIABLE_PROBABILITIES")
        if str(row.get("betting_action") or "").upper() != "BET":
            reasons.append("MODEL_NOT_APPROVED")
    else:
        tier = str(row.get("edge_priority") or "UNSUPPORTED")
        supported = (
            tier == "ROBUST_CORE"
            and row.get("edge_reliability_status") == "SUPPORTED_SUBGROUP"
            and row.get("price_evidence_status") == "CONFIRMED"
        )
        if not supported:
            reasons.append("NOT_ROBUST_CORE")
        holdout_bets = number(row.get("subgroup_holdout_bets"))
        if not yes(row.get("subgroup_holdout_confirmed")) or not holdout_bets:
            reasons.append("HOLDOUT_NOT_CONFIRMED")
        conservative_probability = number(row.get("historical_price_wilson_lower"))
        price_cushion = number(row.get("conservative_price_margin"))
        line_cushion = number(row.get("line_cushion_points"))
        conservative_value = conservative_ev(conservative_probability, odds)
        if (
            price_cushion is None or price_cushion <= 0
            or conservative_value is None or conservative_value <= 0
        ):
            reasons.append("NO_CONSERVATIVE_PRICE_EDGE")
        if market == "spread" and (line_cushion is None or line_cushion < 0.5):
            reasons.append("NO_LINE_CUSHION")
    # New provider fields must be explicitly present, not inferred from
    # the collector's timestamp, a book name, or historical EV.
    if not yes(value("market_execution_verified", "market_execution_verified")):
        reasons.append("BOOK_EXECUTION_UNVERIFIED")
    if not yes(value("market_quote_timestamp_verified",
                     "market_quote_timestamp_verified")):
        reasons.append("PROVIDER_TIMESTAMP_UNVERIFIED")
    if nfl and not yes(row.get("market_quote_sanity_ok")):
        reasons.append("PRICE_SANITY_FAILED")
    if not yes(gate.get("production_eligible")):
        reasons.append("GLOBAL_RELEASE_BLOCKED")

    if not reasons:
        decision = "BET_READY"
    elif supported and "GAME_STARTED" not in reasons and "POST_KICKOFF_QUOTE" not in reasons:
        if any(r in reasons for r in (
            "STALE_QUOTE", "UNRESOLVED_BOOK", "QUOTE_TIME_MISSING",
            "INVALID_PRICE", "FUTURE_QUOTE",
        )):
            decision = "RECHECK_PRICE"
        else:
            decision = "RESEARCH_CORE"
    else:
        decision = "NO_BET"
    return {
        "sport": sport, "rank": None,
        "game_id": str(row.get("game_id") or ""),
        "kickoff": kickoff_str or "", "away_team": row.get("away_team") or "",
        "home_team": row.get("home_team") or "",
        "market": market, "side": side, "line": line, "american_odds": odds,
        "book": book, "quoted_at": quote_str or "",
        "quote_age_minutes": round(quote_age, 1) if quote_age is not None else None,
        "model_probability": probability, "raw_model_ev": raw_ev,
        "conservative_probability": conservative_probability,
        "conservative_ev": conservative_value,
        "price_cushion_pp": round(100 * price_cushion, 2)
        if price_cushion is not None else None,
        "line_cushion_points": line_cushion,
        "historical_holdout_bets": holdout_bets,
        "evidence_tier": tier, "decision": decision,
        "bet_approved": not reasons,
        "reasons": ";".join(dict.fromkeys(reasons)),
    }


def order(row):
    decisions = {"BET_READY": 0, "RESEARCH_CORE": 1, "RECHECK_PRICE": 2, "NO_BET": 3}
    cushion = row["price_cushion_pp"]
    return (
        decisions[row["decision"]],
        -cushion if cushion is not None else 9999.0,
        -(row["line_cushion_points"] or 0),
        str(row["game_id"]), str(row["market"]),
    )


def build(rows, *, sport, gate, as_of):
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Evaluation time must carry UTC offset")
    if "production_eligible" not in gate:
        raise ValueError("Missing release authority")
    examined = [inspect(row, sport, gate, as_of) for row in rows]
    best = {}
    for item in sorted(examined, key=order):
        key = item["game_id"]
        if not key:
            raise ValueError("Blank game ID")
        best.setdefault(key, item)
    ranked = sorted(best.values(), key=order)
    for n, item in enumerate(ranked, 1):
        item["rank"] = n
    counts = dict(Counter(x["decision"] for x in ranked))
    return ranked, {
        "version": 1, "sport": sport,
        "evaluated_at": as_of.astimezone(UTC).isoformat(),
        "release_state": gate.get("release_state", "UNKNOWN"),
        "production_eligible": yes(gate["production_eligible"]),
        "raw_market_candidates": len(examined), "distinct_games": len(ranked),
        "decision_counts": counts, "approved_bets": counts.get("BET_READY", 0),
        "release_blockers": gate.get("blockers", [])[:8],
        "method": "ONE_PER_GAME_CHRONOLOGICAL_CONSERVATIVE_RESEARCH_PRIORITY",
        "no_model_or_stake_change": True,
        "disclaimer": (
            "A research edge is not a proved profitable bet. "
            "Do not wager at an unverified or stale quote."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--as-of")
    args = p.parse_args()
    now = timestamp(args.as_of) if args.as_of else datetime.now(UTC)
    if now is None:
        raise ValueError("Invalid as-of timestamp")
    with args.source.open(newline="", encoding="utf-8-sig") as f:
        source = list(csv.DictReader(f))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    results, report = build(source, sport=args.sport, gate=gate, as_of=now)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
