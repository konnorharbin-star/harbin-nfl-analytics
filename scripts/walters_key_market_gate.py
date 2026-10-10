"""Research-only bridge: published independent football margin -> exact book spread.

No sportsbook quotes are invented or collected. Even fully attested two-sided
quotes remain review-only: historical economic performance is unverified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import timedelta
from pathlib import Path

from scripts.forward_price_validation import decimal, first_commit_time, utc
from scripts.walters_key_number_distribution import distribution

QUOTE_SPEC = "walters_key_spread_quote_v1"
MODEL_SPEC = "walters_key_number_forward_v1"
FLAGS = (
    "book_identity_verified",
    "source_quote_time_verified",
    "executable_price_verified",
    "book_access_verified",
    "settlement_rules_verified",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def numeric(v):
    if isinstance(v, bool):
        raise ValueError("Boolean is not a numeric price or handicap")
    try:
        result = float(v)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid numeric value") from exc
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric value")
    return result


def half_point(v):
    result = numeric(v)
    if abs(result) > 80 or not (2 * result).is_integer():
        raise ValueError("Spread must be a half-point or integer")
    return result


def outcomes(pmf, home_line):
    """Two-sided win/push/loss on actual same-line spread, no continuity fiction."""
    line = half_point(home_line)
    if len(pmf) != 201 or not all(
        isinstance(p, (float, int)) and not isinstance(p, bool)
        and math.isfinite(p) and p >= 0 for p in pmf
    ) or not math.isclose(sum(pmf), 1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Invalid exact-margin distribution")
    home = [0.0, 0.0, 0.0]
    for i, p in enumerate(pmf):
        net = i - 100 + line
        home[0 if net > 0 else 1 if net == 0 else 2] += p
    return {"home": home, "away": [home[2], home[1], home[0]]}


def price_ev(p, american):
    """WIN net profit, PUSH zero, LOSS -1, for one unit at actual odds."""
    if len(p) != 3 or any(
        not math.isfinite(float(v)) or v < 0 for v in p
    ) or not math.isclose(sum(p), 1, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Bad win/push/loss masses")
    return p[0] * (decimal(american) - 1) - p[2]


def _model(f):
    if (f.get("spec") != MODEL_SPEC or f.get("release_state") != "SHADOW"
            or f.get("fixed_alpha") != 1.0
            or f.get("sportsbook_odds_used") is not False
            or f.get("season") != 2026):
        raise ValueError("Unfrozen or contaminated prediction")
    a = f.get("key_multipliers") or {}
    if set(a) != {"3", "7"}:
        raise ValueError("Unapproved key-number specification")
    scale = numeric(f["model_sigma"])
    center = numeric(f["model_mean"])
    fair_margin = numeric(f["independent_home_margin"])
    if scale <= 1 or abs(fair_margin) > 100:
        raise ValueError("Invalid score distribution")
    multipliers = {int(k): numeric(v) for k, v in a.items()}
    if any(not 0.6 <= v <= 2.2 for v in multipliers.values()):
        raise ValueError("Unfrozen margin multiplier")
    return {"mean": center, "sigma": scale, "multipliers": multipliers}, fair_margin


def blockers(q, f, quote_commit, model_commit, file_name):
    problems = []
    if q.get("spec") != QUOTE_SPEC or not SAFE_ID.fullmatch(file_name):
        problems.append("INVALID_QUOTE_SPEC")
    if q.get("quote_id") != file_name:
        problems.append("QUOTE_ID_FILENAME_MISMATCH")
    if any(q.get(k) != f.get(k) for k in
           ("game_id", "sport", "home_team", "away_team", "kickoff")):
        problems.append("WRONG_GAME_OR_KICKOFF")
    if not isinstance(q.get("book"), str) or not q["book"].strip():
        problems.append("NO_NAMED_BOOK")
    if not isinstance(q.get("source_url"), str) or not q["source_url"].startswith(
        "https://"
    ):
        problems.append("NO_DIRECT_BOOK_URL")
    if not isinstance(q.get("source_evidence_note"), str) or len(
        q["source_evidence_note"].strip()
    ) < 20:
        problems.append("NO_INDEPENDENT_SOURCE_REVIEW")
    for name in FLAGS:
        if q.get(name) is not True:
            problems.append("UNATTESTED_" + name.upper())
    try:
        home, away = half_point(q["home_spread"]), half_point(q["away_spread"])
        if home != -away:
            raise ValueError("Not same opposing handicap")
        decimal(q["home_american_odds"])
        decimal(q["away_american_odds"])
    except (ValueError, KeyError, TypeError):
        problems.append("INVALID_SAME_BOOK_PAIRED_PRICE")
    try:
        kickoff = utc(f["kickoff"])
        initial = utc(f["captured_at"])
        source = utc(q["source_quote_at"])
        observed = utc(q["captured_at"])
        if not source <= observed < kickoff:
            problems.append("QUOTE_NOT_PREGAME")
        if observed - source > timedelta(minutes=15):
            problems.append("PROVIDER_QUOTE_ORIGIN_STALE")
        if kickoff - observed > timedelta(days=7):
            problems.append("QUOTE_BEYOND_SEVEN_DAYS")
        if model_commit is None or not initial <= model_commit < observed:
            problems.append("MODEL_NOT_PUBLISHED_BEFORE_PRICE_OBSERVATION")
        if quote_commit is None or not observed <= quote_commit < kickoff:
            problems.append("QUOTE_NOT_PUBLISHED_PREGAME")
    except (KeyError, ValueError, TypeError):
        problems.append("INVALID_QUOTE_TIMESTAMPS")
    return sorted(set(problems))


def compare(f, q):
    model, margin = _model(f)
    home_line = half_point(q["home_spread"])
    if home_line != -half_point(q["away_spread"]):
        raise ValueError("Incompatible quote side lines")
    home_odds = q["home_american_odds"]
    away_odds = q["away_american_odds"]
    hraw = 1 / decimal(home_odds)
    araw = 1 / decimal(away_odds)
    market = {"home": hraw / (hraw + araw), "away": araw / (hraw + araw)}
    report = {}
    for name, alpha in (("discrete_gaussian", 0.0), ("key_number", 1.0)):
        p = outcomes(distribution(model, margin, alpha), home_line)
        report[name] = {}
        for side, odds in (("home", home_odds), ("away", away_odds)):
            win, push, loss = p[side]
            report[name][side] = {
                "win": win, "push": push, "loss": loss,
                "model_conditional_win_given_no_push": win / (win + loss),
                "market_conditional_no_vig_probability": market[side],
                "market_break_even_probability": 1 / decimal(odds),
                "unshrunk_model_ev_per_unit": price_ev(p[side], odds),
                "exact_american_odds": int(numeric(odds)),
            }
    return {
        "game_id": f["game_id"], "quote_id": q["quote_id"],
        "book": q["book"], "home_spread": home_line,
        "home_team": f["home_team"], "away_team": f["away_team"],
        "source_url": q["source_url"],
        "models": report,
        "status": "ATTESTED_REVIEW_ONLY_NO_ECONOMIC_EDGE_PROVEN",
        "wager_authorized": False,
    }


def audit(forecast_root, quote_root, *, commit_lookup=first_commit_time):
    forecasts, pub_dates, bad_forecasts = {}, {}, {}
    for path in sorted(Path(forecast_root).glob("*.json")):
        try:
            f = json.loads(path.read_text(encoding="utf-8"))
            _model(f)
            published = commit_lookup(path)
            if f["game_id"] != path.stem or f["game_id"] in forecasts:
                raise ValueError("Invalid forecast game identity")
            if published is None or not utc(f["captured_at"]) <= published < utc(
                f["kickoff"]
            ):
                raise ValueError("Not original pregame Git publication")
            forecasts[f["game_id"]] = f
            pub_dates[f["game_id"]] = published
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            bad_forecasts[path.name] = type(exc).__name__ + ": " + str(exc)
    rows, rejected = [], {}
    price_files = sorted(Path(quote_root).glob("*.json"))
    for path in price_files:
        try:
            q = json.loads(path.read_text(encoding="utf-8"))
            f = forecasts.get(q.get("game_id"))
            if f is None:
                rejected[path.name] = ["NO_PUBLISHED_INDEPENDENT_FORECAST"]
                continue
            reasons = blockers(q, f, commit_lookup(path), pub_dates[q["game_id"]],
                               path.stem)
            if reasons:
                rejected[path.name] = reasons
            else:
                rows.append(compare(f, q))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            rejected[path.name] = [type(exc).__name__ + ": " + str(exc)]
    return {
        "spec": "walters_key_spread_price_bridge_v1",
        "published_pregame_forecasts": len(forecasts),
        "blocked_forecasts": bad_forecasts,
        "quote_files": len(price_files),
        "attested_two_sided_quote_observations": len(rows),
        "blocked_quote_files": rejected,
        "comparisons": rows,
        "status": "RESEARCH_ONLY_NOT_A_VALIDATED_ECONOMIC_EDGE",
        "reported_ev_is_a_model_hypothesis_not_observed_profit": True,
        "book_execution_independently_confirmed_by_code": False,
        "historical_profitability_and_clv_verified": False,
        "betting_authorized": False,
        "actual_wagers": 0,
        "limits": [
            "Bookmaker quote source flags are human attestations not verification by code",
            "Bettable odds must be independently recorded and committed pregame",
            "No quotes or prices may be invented; missing quotes are missing EV",
            "Every exact price compares with win, push and loss (not a binary bet)",
            "Even source-attested disagreements are research, not live bet recommendations",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecasts", type=Path,
                        default=Path("history/walters_key_forward_v1/forecasts"))
    parser.add_argument("--quotes", type=Path,
                        default=Path("history/walters_key_forward_v1/book_quotes"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/walters_key_market_gate.json"))
    args = parser.parse_args()
    result = audit(args.forecasts, args.quotes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "published_forecasts": result["published_pregame_forecasts"],
        "attested_quote_observations": result["attested_two_sided_quote_observations"],
        "blocked_quote_files": len(result["blocked_quote_files"]),
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
