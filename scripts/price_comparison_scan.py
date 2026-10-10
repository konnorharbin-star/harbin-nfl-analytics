"""Exact-line, leave-one-book-out price diagnostics. Never emits BET."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from scripts.market_first_experiment import immutable
from scripts.recommendation_ledger import timestamp

MAX_AGE = timedelta(minutes=15)
MAX_SKEW = timedelta(minutes=2)
MIN_REFERENCE_BOOKS = 3
SPEC = "exact_line_price_scan_v1"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def decimal(value):
    odds = number(value)
    if odds is None or abs(odds) < 100:
        return None
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def book_key(label):
    if not isinstance(label, str) or not label.strip():
        return ""
    key = re.sub(r"[^a-z0-9]", "", str(label).lower())
    # Canonical identity deduplicates two aggregator labels; it does not certify availability.
    return {
        "actionnetworkbook15": "draftkings",
        "actionnetworkbook30": "fanduel",
        "actionnetworkbook75": "betmgm",
    }.get(key, key)


def normalize(game, q, observed):
    """Keep only complete same-book pairs; malformed markets stay missing."""
    base = {
        "game_id": str(game["game_id"]),
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "kickoff": game.get("kickoff") or game["date"],
        "book_label": q.get("provider"),
        "book_key": book_key(q.get("provider")),
        "source": q.get("source"),
        "source_url": q.get("source_url"),
        "original_request_url": None,
        "observed_at": observed,
        "reported_source_time": q.get("last_update"),
        "source_quote_time_verified": False,
        "book_identity_verified": False,
        "executable_price_verified": False,
    }
    records = []
    for market, line, first, second, keys in [
        ("moneyline", None, "home", "away", ("home_ml", "away_ml")),
        (
            "spread",
            number(q.get("home_spread")),
            "home",
            "away",
            ("home_spread_price", "away_spread_price"),
        ),
        ("total", number(q.get("market_total")), "over", "under", ("over_price", "under_price")),
    ]:
        prices = [number(q.get(k)) for k in keys]
        if market != "moneyline" and line is None:
            continue
        if all(decimal(p) is not None for p in prices):
            records.append(
                {**base, "market": market, "line": line, "sides": [first, second], "odds": prices}
            )
    return records


def load(sport):
    games = list(csv.DictReader(Path("docs/latest.csv").open()))
    by_id = {r["game_id"]: r for r in games}
    result = []
    if sport == "cfb":
        for game in by_id.values():
            for q in json.loads(game.get("market_quotes_json") or "[]"):
                if q.get("source") not in ("action_network", "espn_core", "espn"):
                    continue
                result.extend(normalize(game, q, q.get("captured_at")))
        return games, result
    fields = [
        "captured_at",
        "kickoff",
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "market_type",
        "provider",
        "book",
        "source_event_id",
        "first_side",
        "first_line",
        "first_american_odds",
        "second_side",
        "second_line",
        "second_american_odds",
    ]
    for row in csv.DictReader(Path("history/market_snapshots.csv").open()):
        if row["game_id"] not in by_id or row["provider"] not in ("espn", "action_network"):
            continue
        prices = {
            row["first_side"]: row["first_american_odds"],
            row["second_side"]: row["second_american_odds"],
        }
        lines = {
            row["first_side"]: number(row["first_line"]),
            row["second_side"]: number(row["second_line"]),
        }
        market = row["market_type"]
        if market == "spread" and (
            lines.get("home") is None or lines.get("away") != -lines["home"]
        ):
            continue
        if market == "total" and (
            lines.get("over") is None or lines.get("over") != lines.get("under")
        ):
            continue
        q = {
            "provider": row["book"],
            "source": row["provider"],
            "home_ml": prices.get("home") if market == "moneyline" else None,
            "away_ml": prices.get("away") if market == "moneyline" else None,
            "home_spread": lines.get("home"),
            "home_spread_price": prices.get("home") if market == "spread" else None,
            "away_spread_price": prices.get("away") if market == "spread" else None,
            "market_total": lines.get("over"),
            "over_price": prices.get("over"),
            "under_price": prices.get("under"),
        }
        normalized = normalize(by_id[row["game_id"]], q, row["captured_at"])
        identity = json.dumps(tuple(row.get(k) or "" for k in fields), separators=(",", ":"))
        sidecar = Path("history/market_snapshots.csv.events") / (
            hashlib.sha256(identity.encode()).hexdigest() + ".json"
        )
        if sidecar.exists():
            event = json.loads(sidecar.read_text())
            for item in normalized:
                item["reported_source_time"] = event.get("source_quote_at")
                item["source_quote_time_verified"] = event.get("source_quote_time_verified") is True
        result.extend(normalized)
    return games, result


def refresh(sport, games):
    """Existing free collectors only; optional paid/keyed source is never called."""
    now = datetime.now(UTC)
    targets = list(
        {r["game_id"]: r for r in games if timestamp(r.get("kickoff") or r["date"]) > now}.values()
    )
    result, errors = [], []
    if not targets:
        return result, errors
    if sport == "nfl":
        import polars as pl

        from nfl.action_network import ActionNetworkNFLClient

        weeks = {int(r["week"]) for r in targets}
        if len(weeks) != 1:
            raise ValueError("Current slate spans multiple weeks")
        client = ActionNetworkNFLClient(timeout_seconds=8)
        markets = client.current_markets(pl.DataFrame(targets), week=next(iter(weeks)))
        by_id = {r["game_id"]: r for r in targets}
        observed = datetime.now(UTC).isoformat()
        for market in markets:
            if market.provider not in ("espn", "action_network"):
                continue
            sides = {
                market.first_side: market.first_american_odds,
                market.second_side: market.second_american_odds,
            }
            lines = {market.first_side: market.first_line, market.second_side: market.second_line}
            q = {
                "provider": market.book,
                "source": market.provider,
                "source_url": client.last_diagnostic.get("endpoint"),
                "last_update": market.source_quote_at.isoformat()
                if market.source_quote_at
                else None,
                "home_ml": sides.get("home") if market.market_type == "moneyline" else None,
                "away_ml": sides.get("away") if market.market_type == "moneyline" else None,
                "home_spread": lines.get("home"),
                "home_spread_price": sides.get("home") if market.market_type == "spread" else None,
                "away_spread_price": sides.get("away") if market.market_type == "spread" else None,
                "market_total": lines.get("over"),
                "over_price": sides.get("over"),
                "under_price": sides.get("under"),
            }
            result.extend(normalize(by_id[market.game_id], q, observed))
        return result, errors
    from harbin.market_intel import MarketIntelligence, _fuzzy_team_match

    collector = MarketIntelligence()
    collector._load_action_network([SimpleNamespace(**r) for r in targets])
    errors.extend(collector.errors)
    observed = datetime.now(UTC).isoformat()
    for game in targets:
        matches = [
            x
            for x in collector.action_games
            if _fuzzy_team_match(game["home_team"], x["home_names"])
            and _fuzzy_team_match(game["away_team"], x["away_names"])
        ]
        if len(matches) != 1:
            continue
        try:
            if (
                abs((timestamp(matches[0]["start_time"]) - timestamp(game["date"])).total_seconds())
                > 300
            ):
                continue
        except (ValueError, TypeError):
            continue
        for q in matches[0]["quotes"]:
            q = {**q, "source_url": "https://api.actionnetwork.com/web/v2/scoreboard/ncaaf"}
            result.extend(normalize(game, q, observed))
    return result, errors


def scan(records, now):
    groups, counts = defaultdict(dict), Counter()
    for r in records:
        try:
            observed, kickoff = timestamp(r["observed_at"]), timestamp(r["kickoff"])
        except (ValueError, TypeError, KeyError):
            counts["missing_aware_timestamps"] += 1
            continue
        if not observed <= now < kickoff or now - observed > MAX_AGE:
            counts["stale_future_or_started"] += 1
            continue
        if (
            not r.get("book_key")
            or r.get("sides") not in (["home", "away"], ["over", "under"])
            or not all(decimal(x) for x in r["odds"])
        ):
            counts["invalid_pair"] += 1
            continue
        key = (r["game_id"], r["home_team"], r["away_team"], kickoff, r["market"], r["line"])
        previous = groups[key].get(r["book_key"])
        if previous is None or observed > timestamp(previous["observed_at"]):
            groups[key][r["book_key"]] = r
    candidates = []
    for group in groups.values():
        newest = max(timestamp(r["observed_at"]) for r in group.values())
        quotes = [r for r in group.values() if newest - timestamp(r["observed_at"]) <= MAX_SKEW]
        counts["fresh_exact_line_groups"] += 1
        if len(quotes) < MIN_REFERENCE_BOOKS + 1:
            counts["insufficient_same_line_books"] += 1
            continue
        for side in (0, 1):
            best = max(quotes, key=lambda r: decimal(r["odds"][side]))
            references = [r for r in quotes if r["book_key"] != best["book_key"]]
            probabilities = []
            for r in references:
                a, b = [1 / decimal(x) for x in r["odds"]]
                probabilities.append((a, b)[side] / (a + b))
            reference = median(probabilities)
            implied = 1 / decimal(best["odds"][side])
            gap = reference - implied
            if gap < 0.01:
                continue
            haircut = max(0.02, max(probabilities) - min(probabilities))
            candidates.append(
                {
                    "game_id": best["game_id"],
                    "home_team": best["home_team"],
                    "away_team": best["away_team"],
                    "kickoff": best["kickoff"],
                    "market": best["market"],
                    "side": best["sides"][side],
                    "line": -best["line"]
                    if best["market"] == "spread" and side == 1
                    else best["line"],
                    "book_label": best["book_label"],
                    "american_odds": best["odds"][side],
                    "quote_observed_at": best["observed_at"],
                    "reference_books": len(references),
                    "reference_probability": reference,
                    "price_implied_probability": implied,
                    "reference_price_gap_pp": 100 * gap,
                    "sensitivity_gap_pp": 100 * (reference - haircut - implied),
                    "reference_probability_range": [min(probabilities), max(probabilities)],
                    "status": "VERIFY_REQUIRED",
                    "betting_authorized": False,
                    "stake_units": 0,
                    "blockers": [
                        "Aggregator book identity and executable price unverified",
                        "Per-side source quote-origin timestamps unverified",
                        "No verified sharp benchmark",
                        "Reference probabilities are not calibrated model probabilities",
                        "Market-reference strategy lacks validated prospective economic advantage",
                        "User book access and settlement rules unverified",
                    ],
                }
            )
    candidates.sort(key=lambda r: (-r["sensitivity_gap_pp"], r["game_id"], r["market"], r["side"]))
    return {
        "spec": SPEC,
        "observed_at": now.isoformat(),
        "status": "PRICE_DIAGNOSTICS_ONLY",
        "betting_authorized": False,
        "qualified_bets": 0,
        "counts": dict(counts),
        "candidates": candidates,
        "limits": [
            "Exact line and two same-book prices required",
            "Exclude quoted book from reference and deduplicate aliases",
            "Non-push reference prices imply no true EV, ROI or fair price",
            "Sensitivity haircut is a fixed stress scenario, not a confidence interval",
            "Unresolved and legacy aggregator labels are not certified live sportsbooks",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--refresh-market", action="store_true")
    args = parser.parse_args()
    games, records = load(args.sport)
    errors = []
    if args.refresh_market:
        try:
            fresh, errors = refresh(args.sport, games)
            records.extend(fresh)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    now = datetime.now(UTC)
    report = scan(records, now)
    current = []
    for record in records:
        try:
            observed = timestamp(record["observed_at"])
            if observed <= now < timestamp(record["kickoff"]) and now - observed <= MAX_AGE:
                current.append(record)
        except (ValueError, TypeError):
            pass
    report["upcoming_games"] = len(
        {r["game_id"] for r in games if timestamp(r.get("kickoff") or r["date"]) > now}
    )
    report["fresh_priced_games"] = len({r["game_id"] for r in current})
    report.update(sport=args.sport, source_errors=errors, decision="NO_BET")
    identity = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:24]
    report["capture_id"] = identity
    immutable(
        Path("history/price_scan_v1/captures") / (identity + ".json"),
        {"report": report, "source_records": current},
    )
    Path("docs/price_comparison_scan.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "sport": args.sport,
                "capture_id": identity,
                "counts": report["counts"],
                "review_candidates": len(report["candidates"]),
                "qualified_bets": 0,
                "source_errors": errors,
            }
        )
    )


if __name__ == "__main__":
    main()
