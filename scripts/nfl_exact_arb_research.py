"""Fail-closed two-sided sportsbook price-gap research; NEVER a betting authorization.

No network calls, no sportsbook login, no claimed validated sportsbook odds.
Only matching two-way, half-point *NFL spread* contracts can be considered.
An observed mathematical gap is not a guaranteed executed return.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path

CENT = Decimal("0.01")
MIN_ODDS = Decimal("101")
MAX_AGE = timedelta(minutes=2)


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("TIME_MISSING")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("TIME_UNZONED")
    return dt.astimezone(UTC)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError("NUMBER_MISSING")
    try:
        number = Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation) as exc:
        raise ValueError("NUMBER_INVALID") from exc
    if not number.is_finite():
        raise ValueError("NUMBER_NONFINITE")
    return number


def _decimal_odds(american: object) -> Decimal:
    odds = _decimal(american)
    if odds >= 100:
        return Decimal(1) + odds / 100
    if odds <= -101:
        return Decimal(1) + 100 / (-odds)
    raise ValueError("INVALID_AMERICAN_PRICE")


def _base() -> dict:
    return {
        "status": "PASS_NO_QUALIFIED_ARBITRAGE",
        "action": "PASS_RESEARCH_ONLY",
        "wager_authorized": False,
        "guaranteed_profit_claimed": False,
        "actual_wagers_placed": 0,
        "executable_price_independently_confirmed": False,
        "estimated_real_bet_ev": None,
        "blockers": [],
        "synthetic_hold": None,
        "worst_case_profit": None,
        "stake_allocation": None,
    }


def screen_pair(
    a: dict, b: dict, *, now: datetime, bankroll: object = "200",
    fee_allowance: object = "0",
) -> dict:
    """Screen reported prices, always PASS until independent two-leg execution proof.

    Legal access and quotes supplied as self-reported flags cannot prove real
    executable account prices, subsequent acceptance, or independent settlement.
    Theoretical payouts cover HOME and AWAY for a no-push half-point NFL spread;
    all other markets are blocked (including NFL moneyline with possible ties).
    """
    result = _base()
    reasons: list[str] = []
    if now.tzinfo is None:
        raise ValueError("Reference clock must be timezone-aware")
    now = now.astimezone(UTC)
    if not isinstance(a, dict) or not isinstance(b, dict):
        result["blockers"] = ["MISSING_TWO_QUOTE_OBJECTS"]
        return result
    if a.get("market") != "NFL_FULL_GAME_SPREAD" or b.get("market") != "NFL_FULL_GAME_SPREAD":
        reasons.append("OUTCOME_SET_NOT_CERTIFIED_TWO_WAY_NO_PUSH")
    for key in ("event_id", "home_team", "away_team", "period", "settlement_rules_id", "kickoff_utc"):
        if not a.get(key) or not b.get(key) or a.get(key) != b.get(key):
            reasons.append("NONIDENTICAL_CONTRACT_" + key.upper())
    if {a.get("side"), b.get("side")} != {"home", "away"}:
        reasons.append("NOT_OPPOSITE_SIDES")
    if not a.get("book") or not b.get("book") or a.get("book") == b.get("book"):
        reasons.append("TWO_DISTINCT_BOOKS_REQUIRED")
    if a.get("period") != "full_game_including_overtime":
        reasons.append("FULL_GAME_OVERTIME_CONTRACT_REQUIRED")
    for quote in (a, b):
        for flag in ("account_access_legal_attested", "book_price_account_visible_attested",
                     "rules_compatible_manually_attested"):
            if quote.get(flag) is not True:
                reasons.append("UNATTESTED_" + flag.upper())
        if not quote.get("bookmaker_evidence_url"):
            reasons.append("MISSING_PRICE_EVIDENCE_URL")
    try:
        la, lb = _decimal(a.get("line")), _decimal(b.get("line"))
        if la + lb != 0 or la == 0:
            reasons.append("HANDICAP_NOT_EXACT_OPPOSITE")
        if ((la.copy_abs() * 2) % 2) != 1:
            reasons.append("SPREAD_HAS_POSSIBLE_PUSH_OR_UNSUPPORTED_QUARTER_LINE")
        d1, d2 = _decimal_odds(a.get("american_odds")), _decimal_odds(b.get("american_odds"))
        stake_cap1 = _decimal(a.get("max_stake")), _decimal(b.get("max_stake"))
        cap1, cap2 = stake_cap1
        total_budget, costs = _decimal(bankroll), _decimal(fee_allowance)
        if min(cap1, cap2, total_budget) <= 0 or costs < 0:
            reasons.append("INVALID_STAKES_OR_FEES")
        origin1, origin2 = _utc(a.get("provider_quote_at_utc")), _utc(b.get("provider_quote_at_utc"))
        capture1, capture2 = _utc(a.get("captured_at_utc")), _utc(b.get("captured_at_utc"))
        kickoff = _utc(a.get("kickoff_utc"))
        if not (origin1 <= capture1 <= now < kickoff and origin2 <= capture2 <= now < kickoff):
            reasons.append("NONPROSPECTIVE_OR_UNVERIFIABLE_QUOTE_TIME")
        if now - capture1 > MAX_AGE or now - capture2 > MAX_AGE:
            reasons.append("STALE_OBSERVED_PRICE")
        if capture1 - origin1 > MAX_AGE or capture2 - origin2 > MAX_AGE:
            reasons.append("STALE_PROVIDER_QUOTE_ORIGIN")
        if abs(capture1 - capture2) > MAX_AGE:
            reasons.append("NONCONTEMPORANEOUS_PRICES")
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        reasons.append("INVALID_QUOTE_INPUT_" + str(exc))
        result["blockers"] = sorted(set(reasons))
        return result
    if reasons:
        result["blockers"] = sorted(set(reasons))
        return result
    qsum = Decimal(1) / d1 + Decimal(1) / d2
    result["synthetic_hold"] = float(qsum - 1)
    if qsum >= 1:
        result["blockers"] = ["NO_POSITIVE_TWO_WAY_PRICE_GAP"]
        return result
    # Do not exceed an attested limit, and conservatively round down payouts.
    max_affordable = min(total_budget, cap1 * d1 * qsum, cap2 * d2 * qsum)
    total = max_affordable.quantize(CENT, rounding=ROUND_DOWN)
    s1 = (total * (Decimal(1) / d1) / qsum).quantize(CENT, rounding=ROUND_HALF_UP)
    s2 = total - s1
    if min(s1, s2) <= 0 or s1 > cap1 or s2 > cap2:
        result["blockers"] = ["STAKE_CAP_OR_ROUNDING_INFEASIBLE"]
        return result
    payout1 = (s1 * d1).quantize(CENT, rounding=ROUND_DOWN)
    payout2 = (s2 * d2).quantize(CENT, rounding=ROUND_DOWN)
    lower = min(payout1, payout2) - total - costs
    result["worst_case_profit"] = str(lower)
    result["stake_allocation"] = {a["book"]: str(s1), b["book"]: str(s2)}
    if lower <= 0:
        result["blockers"] = ["FEES_OR_ROUNDING_REMOVE_PRICE_GAP"]
    else:
        result["status"] = "CONDITIONAL_MATHEMATICAL_GAP_NOT_EXECUTABLE_PROOF"
        result["blockers"] = [
            "UNCONFIRMED_DUAL_ACCEPTED_TICKETS",
            "INDEPENDENT_SETTLEMENT_AND_VOID_RISK",
            "SELF_REPORTED_QUOTE_ATTESTATIONS_NOT_INDEPENDENT_PROOF",
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--at", required=True, help="research UTC observation timestamp")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    answer = screen_pair(
        payload.get("a"), payload.get("b"), now=_utc(args.at),
        bankroll=payload.get("bankroll", "200"),
        fee_allowance=payload.get("fee_allowance", "0"),
    )
    print(json.dumps(answer, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
