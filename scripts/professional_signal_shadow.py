"""Forward-only market-controlled audit of individually published expert picks.

Research only; no actual bets, X scraping, paid data, proprietary coefficients,
or automatic adoption of public handicappers into the independent projection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

VERSION = "professional_signal_shadow_v1"
SUPPORTED = {
    "GAME_MONEYLINE": {"HOME", "AWAY"},
    "GAME_SPREAD": {"HOME", "AWAY"},
    "GAME_TOTAL": {"OVER", "UNDER"},
    "PLAYER_RECEIVING_YARDS": {"OVER", "UNDER"},
    "PLAYER_RUSHING_YARDS": {"OVER", "UNDER"},
    "PLAYER_RECEPTIONS": {"OVER", "UNDER"},
    "PLAYER_PASSING_YARDS": {"OVER", "UNDER"},
}
MIN_TRAIN_PICKS = 25
MIN_TRAIN_EVENTS = 20
MIN_SCORE_PICKS = 20
OFFSETS = tuple(n / 20 for n in range(-20, 21))
PENALTY = 0.15  # pre-registered stabilization, not learned from test outcomes


def utc(value):
    if not isinstance(value, str):
        raise ValueError("timestamp missing")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError("invalid timestamp") from e
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timezone-required timestamp")
    return t.astimezone(timezone.utc)


def decimal(odds):
    if isinstance(odds, bool):
        raise ValueError("invalid odds")
    try:
        x = int(odds)
    except (TypeError, ValueError) as e:
        raise ValueError("invalid odds") from e
    if str(odds).strip() not in {str(x), f"+{x}"} or abs(x) < 100:
        raise ValueError("invalid odds")
    return 1 + (x / 100 if x > 0 else 100 / abs(x))


def probability(value):
    if isinstance(value, bool):
        raise ValueError("invalid independent model probability")
    try:
        p = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError("invalid independent model probability") from e
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("invalid independent model probability")
    return p


def shifted(p, offset):
    logit = math.log(p / (1 - p)) + offset
    return 1 / (1 + math.exp(-logit))


def logloss(p, outcome):
    return -(outcome * math.log(p) + (1 - outcome) * math.log1p(-p))


def validate(record, league):
    required = ("record_id", "analyst", "league", "event_id", "kickoff_utc",
                "published_utc", "captured_utc", "source_url", "market",
                "period", "participant", "selection", "line", "market_pair", "outcome")
    if not isinstance(record, dict) or any(record.get(k) is None or record[k] == "" for k in required):
        raise ValueError("missing individual expert-pick evidence")
    if record["league"] != league or record["market"] not in SUPPORTED:
        raise ValueError("unsupported league or market")
    if record["period"] != "FULL_GAME" or record["selection"] not in SUPPORTED[record["market"]]:
        raise ValueError("period or selection mismatch")
    if record["market"].startswith("GAME_") and record["participant"] != "GAME":
        raise ValueError("game participant mismatch")
    if record["market"].startswith("PLAYER_") and record["participant"] == "GAME":
        raise ValueError("missing player identity")
    if not str(record["source_url"]).startswith("https://"):
        raise ValueError("unverifiable expert post URL")
    pub, cap, kickoff = (utc(record[key]) for key in
                         ("published_utc", "captured_utc", "kickoff_utc"))
    if not pub <= cap < kickoff:
        raise ValueError("expert pick was not captured before kickoff")
    line = record["line"]
    if record["market"] == "GAME_MONEYLINE":
        if line != "NA":
            raise ValueError("moneyline requires NA line")
    else:
        try:
            number = float(line)
        except (TypeError, ValueError) as e:
            raise ValueError("invalid contract threshold") from e
        if not math.isfinite(number) or abs((number * 2) - round(number * 2)) > 1e-9 or int(number * 2) % 2 != 1:
            raise ValueError("only half-point thresholds supported in binary shadow")
    pair = record["market_pair"]
    expected = ("bookmaker", "snapshot_id", "region", "source_url", "quote_utc",
                "captured_utc", "sides")
    if not isinstance(pair, dict) or any(not pair.get(k) for k in expected):
        raise ValueError("incomplete two-sided market evidence")
    if not str(pair["source_url"]).startswith("https://"):
        raise ValueError("missing market source URL")
    if not pub <= utc(pair["quote_utc"]) <= utc(pair["captured_utc"]) < kickoff:
        raise ValueError("market observation not contemporaneous post-publication")
    sides = pair["sides"]
    if not isinstance(sides, list) or len(sides) != 2:
        raise ValueError("exact two-sided market required")
    if {s.get("selection") for s in sides if isinstance(s, dict)} != SUPPORTED[record["market"]]:
        raise ValueError("two-sided market selections do not match")
    for s in sides:
        if s.get("line") != line:
            raise ValueError("market threshold mismatches published pick")
        decimal(s.get("american_odds"))
    if record["outcome"] not in {"WIN", "LOSS", "PUSH", "PENDING"}:
        raise ValueError("invalid outcome")
    if record["outcome"] == "PENDING":
        if record.get("settled_utc") is not None:
            raise ValueError("pending pick cannot have a settlement")
    elif record.get("settled_utc") is None or utc(record["settled_utc"]) <= kickoff:
        raise ValueError("settlement must follow kickoff")
    model_p = record.get("independent_model_p")
    if model_p is not None:
        probability(model_p)
        if not record.get("independent_model_frozen_utc") or utc(record["independent_model_frozen_utc"]) > pub:
            raise ValueError("independent model not frozen before expert publication")
    implied = {s["selection"]: 1 / decimal(s["american_odds"]) for s in sides}
    denom = sum(implied.values())
    if denom <= 1 or denom > 1.35:
        raise ValueError("suspicious same-book hold; manual review required")
    p = implied[record["selection"]] / denom
    return {**record, "market_p": p, "implied_market_hold": denom - 1,
            "_pub": pub, "_settled": utc(record["settled_utc"]) if record.get("settled_utc") else None}


def _train_shift(prior):
    if len(prior) < MIN_TRAIN_PICKS or len({r["event_id"] for r in prior}) < MIN_TRAIN_EVENTS:
        return 0.0, "INSUFFICIENT_PRIOR_RECORDS"
    baseline = sum(logloss(r["market_p"], int(r["outcome"] == "WIN")) for r in prior) / len(prior)
    scores = [(sum(logloss(shifted(r["market_p"], s), int(r["outcome"] == "WIN")) for r in prior) / len(prior)
               + PENALTY * s * s, abs(s), s) for s in OFFSETS]
    objective, _, shift = min(scores)
    return (shift, "PRIOR_MARKET_IMPROVEMENT") if objective + 1e-12 < baseline else (0.0, "NO_PRIOR_IMPROVEMENT")


def audit(records, league):
    if league not in {"NFL", "CFB"} or not isinstance(records, list):
        raise ValueError("league must be NFL/CFB and records a list")
    validated = [validate(r, league) for r in records]
    if len({r["record_id"] for r in validated}) != len(validated):
        raise ValueError("duplicate record_id")
    if len({(r["analyst"], r["event_id"], r["market"], r["participant"], r["selection"], str(r["line"])) for r in validated}) != len(validated):
        raise ValueError("duplicate expert contract")
    validated.sort(key=lambda r: (r["_pub"], r["record_id"]))
    rows = []
    for r in validated:
        prior = [p for p in validated if p["analyst"] == r["analyst"]
                 and p["_settled"] is not None and p["_settled"] <= r["_pub"]
                 and p["event_id"] != r["event_id"]
                 and p["outcome"] in {"WIN", "LOSS"}]
        shift, status = _train_shift(prior)
        p_expert = shifted(r["market_p"], shift)
        scored = r["outcome"] in {"WIN", "LOSS"}
        y = int(r["outcome"] == "WIN") if scored else None
        rows.append({
            "record_id": r["record_id"], "analyst": r["analyst"], "event_id": r["event_id"],
            "market": r["market"], "outcome": r["outcome"],
            "public_post_url": r["source_url"], "market_source_url": r["market_pair"]["source_url"],
            "market_p": round(r["market_p"], 8), "independent_model_p": r.get("independent_model_p"),
            "expert_shadow_p": round(p_expert, 8), "market_hold": round(r["implied_market_hold"], 6),
            "frozen_prior_picks": len(prior), "expert_offset": shift, "fit_gate": status,
            "market_logloss": logloss(r["market_p"], y) if scored else None,
            "expert_logloss": logloss(p_expert, y) if scored else None,
            "independent_model_logloss": logloss(probability(r["independent_model_p"]), y)
                if scored and r.get("independent_model_p") is not None else None,
            "expert_delta_brier": ((p_expert - y) ** 2 - (r["market_p"] - y) ** 2)
                if scored else None,
            "executable_account_price_verified": False, "bet_authorized": False,
        })
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["analyst"]].append(row)
    results = {}
    for analyst, cohort in sorted(grouped.items()):
        scored = [r for r in cohort if r["market_logloss"] is not None]
        live = [r for r in scored if r["fit_gate"] == "PRIOR_MARKET_IMPROVEMENT"]
        results[analyst] = {
            "submitted": len(cohort), "settled_binary": len(scored),
            "shadow_offset_decisions": len(live),
            "eligible_for_incremental_claim": len(live) >= MIN_SCORE_PICKS,
            "paired_market_logloss": round(sum(r["market_logloss"] for r in live) / len(live), 6) if live else None,
            "paired_expert_logloss": round(sum(r["expert_logloss"] for r in live) / len(live), 6) if live else None,
            "mean_expert_minus_market_brier": round(sum(r["expert_delta_brier"] for r in live) / len(live), 6) if live else None,
        }
    return {"version": VERSION, "league": league,
            "status": "RESEARCH_ONLY" if rows else "NO_REAL_SOURCE_RECORDS",
            "decision": "PASS", "bet_authorized": False, "model_weights_changed": False,
            "requires_complete_source_capture": True,
            "analysts": results, "records": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", choices=["NFL", "CFB"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit(data["records"], args.league)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "count": len(result["records"]),
                      "decision": result["decision"]}))


if __name__ == "__main__":
    main()
