"""Free non-executing market-price edge scout.

Finds only coherent theoretical cross-book two-outcome arbitrage observations.
It does not infer game outcomes, certify sportsbook fills, or submit wagers.
Fallback nflverse schedule markets and optional paid API quotes are excluded.
"""
from __future__ import annotations

import json
import math
from statistics import median
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from .book_identity import canonical_book_identity
from .espn_market import ESPNTwoWayMarket
from .market import american_to_decimal

MAX_QUOTE_AGE = timedelta(minutes=10)
MAX_SYNCHRONIZATION_GAP = timedelta(minutes=5)
MIN_THEORETICAL_MARGIN = 0.005
MIN_CONSENSUS_OTHER_BOOKS = 3
MIN_CONSENSUS_THEORETICAL_EV = 0.05
MAX_CONSENSUS_DISPERSION = 0.025
ALLOWED_FREE_PROVIDERS = frozenset({"espn", "action_network"})
SIDES = {
    "moneyline": frozenset({"home", "away"}),
    "spread": frozenset({"home", "away"}),
    "total": frozenset({"over", "under"}),
}

def _stamp(value: object) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(UTC)

def _quote_rows(
    quotes: Sequence[ESPNTwoWayMarket],
    *,
    as_of: datetime,
    kickoffs: dict[str, datetime],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    valid = []
    excluded = defaultdict(int)
    for market in quotes:
        kickoff = _stamp(kickoffs.get(market.game_id))
        captured = _stamp(market.captured_at)
        book = canonical_book_identity(market.book)
        if market.provider not in ALLOWED_FREE_PROVIDERS:
            excluded["source_not_free_and_verified"] += 1
            continue
        if market.market_type not in SIDES or not book or not market.source_event_id:
            excluded["unidentified_market"] += 1
            continue
        if (
            captured is None or kickoff is None or captured >= kickoff
            or captured > as_of or as_of >= kickoff
            or as_of - captured > MAX_QUOTE_AGE
        ):
            excluded["expired_future_or_postkickoff"] += 1
            continue
        first = str(market.first_side)
        second = str(market.second_side)
        if {first, second} != SIDES[market.market_type]:
            excluded["noncomplementary_pair"] += 1
            continue
        line_a, line_b = market.first_line, market.second_line
        if market.market_type == "moneyline":
            if line_a is not None or line_b is not None:
                excluded["invalid_point_pair"] += 1
                continue
            point = None
        else:
            try:
                valid_lines = all(
                    math.isfinite(float(x)) for x in (line_a, line_b)
                )
            except (ValueError, TypeError, OverflowError):
                valid_lines = False
            if not valid_lines:
                excluded["invalid_point_pair"] += 1
                continue
            if market.market_type == "spread":
                if abs(float(line_a) + float(line_b)) > 1e-8:
                    excluded["invalid_point_pair"] += 1
                    continue
                point = (
                    float(line_a) if first == "home" else float(line_b)
                )
            else:
                if abs(float(line_a) - float(line_b)) > 1e-8:
                    excluded["invalid_point_pair"] += 1
                    continue
                point = float(line_a)
        try:
            odds_a = int(market.first_american_odds)
            odds_b = int(market.second_american_odds)
            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            if abs(odds_a) < 100 or abs(odds_b) < 100:
                raise ValueError("invalid sportsbook price")
        except (TypeError, ValueError, OverflowError):
            excluded["invalid_price"] += 1
            continue
        for side, price, decimal in ((first, odds_a, dec_a), (second, odds_b, dec_b)):
            valid.append({
                "game_id": market.game_id,
                "market": market.market_type,
                "line": point,
                "side": side,
                "book": market.book,
                "book_key": book,
                "provider": market.provider,
                "source_event_id": market.source_event_id,
                "price": price,
                "decimal": decimal,
                "observed_at": captured,
                "quote_origin_time_verified": False,
            })
    return valid, dict(excluded)

def scan_price_edges(
    markets: Sequence[ESPNTwoWayMarket],
    *,
    kickoffs: dict[str, datetime],
    as_of: datetime,
) -> dict[str, object]:
    """Return reproducible quote-time research—not bet recommendations."""
    stamp = _stamp(as_of)
    if stamp is None:
        raise ValueError("as_of must be timezone-aware")
    quotes, excluded = _quote_rows(markets, as_of=stamp, kickoffs=kickoffs)
    groups = defaultdict(list)
    for q in quotes:
        groups[(q["game_id"], q["market"], q["line"])].append(q)

    opportunities = []
    disagreement_watchlist = []
    examined = 0
    for (game, kind, line), observations in sorted(groups.items()):
        sides = sorted(SIDES[kind])
        # Latest quote from each source/book/side is used, no mixing within book.
        distinct = {}
        for q in observations:
            k = (q["book_key"], q["side"])
            previous = distinct.get(k)
            if previous is None or q["observed_at"] > previous["observed_at"]:
                distinct[k] = q
        first = sorted(
            (q for q in distinct.values() if q["side"] == sides[0]),
            key=lambda q: (-q["decimal"], q["book_key"]),
        )
        second = sorted(
            (q for q in distinct.values() if q["side"] == sides[1]),
            key=lambda q: (-q["decimal"], q["book_key"]),
        )
        if not first or not second:
            continue
        examined += 1

        # Potential mispricing research: compare a sportsbook side ONLY to
        # leave-one-book-out no-vig prices from at least three other books.
        # This does NOT establish a true win probability or positive EV.
        complete_books = {}
        for quote in distinct.values():
            complete_books.setdefault(quote["book_key"], {})[quote["side"]] = quote
        complete_books = {
            key: sides_by_book for key, sides_by_book in complete_books.items()
            if set(sides_by_book) == set(sides)
        }
        for book, book_sides in complete_books.items():
            for side in sides:
                selected = book_sides[side]
                contemporaneous = []
                for other, other_sides in complete_books.items():
                    if other == book:
                        continue
                    if (
                        abs(selected["observed_at"] -
                            other_sides[side]["observed_at"])
                        > MAX_SYNCHRONIZATION_GAP
                        or abs(selected["observed_at"] -
                               other_sides[
                                   next(x for x in sides if x != side)
                               ]["observed_at"])
                        > MAX_SYNCHRONIZATION_GAP
                    ):
                        continue
                    counterpart = other_sides[next(x for x in sides if x != side)]
                    raw_p = 1 / other_sides[side]["decimal"]
                    raw_opposite = 1 / counterpart["decimal"]
                    contemporaneous.append(raw_p / (raw_p + raw_opposite))
                if len(contemporaneous) < MIN_CONSENSUS_OTHER_BOOKS:
                    continue
                estimated_probability = median(contemporaneous)
                spread = max(contemporaneous) - min(contemporaneous)
                if spread > MAX_CONSENSUS_DISPERSION:
                    continue
                theoretical_ev = estimated_probability * selected["decimal"] - 1
                if theoretical_ev <= MIN_CONSENSUS_THEORETICAL_EV:
                    continue
                disagreement_watchlist.append({
                    "game_id": game,
                    "market": kind, "line": line,
                    "side": side,
                    "book": selected["book"],
                    "american_odds": selected["price"],
                    "snapshot_at": selected["observed_at"].isoformat(),
                    "reference_excludes_candidate_book": True,
                    "other_books": len(contemporaneous),
                    "median_other_books_no_vig_probability":
                        round(estimated_probability, 6),
                    "reference_probability_range": round(spread, 6),
                    "market_based_theoretical_ev": round(theoretical_ev, 6),
                    "model_win_probability_used": False,
                    "true_edge_proven": False,
                    "source_quote_origin_time_verified": False,
                    "human_action": "MANUAL_RESEARCH_ONLY_NOT_AN_EXECUTABLE_BET",
                })

        eligible_pairs = []
        for a in first:
            for b in second:
                if a["book_key"] == b["book_key"]:
                    continue
                if abs(a["observed_at"]-b["observed_at"]) > MAX_SYNCHRONIZATION_GAP:
                    continue
                implied = 1/a["decimal"] + 1/b["decimal"]
                if implied <= 1 - MIN_THEORETICAL_MARGIN:
                    eligible_pairs.append((implied, a, b))
        if not eligible_pairs:
            continue
        # No multiplicity of identical same-event candidates.
        implied, a, b = min(eligible_pairs, key=lambda x:x[0])
        opportunity = {
            "game_id": game, "market": kind, "line": line,
            "first_side": a["side"], "first_book": a["book"],
            "first_american_odds": a["price"], "first_capture_at": a["observed_at"].isoformat(),
            "second_side": b["side"], "second_book": b["book"],
            "second_american_odds": b["price"], "second_capture_at": b["observed_at"].isoformat(),
            "inverse_decimal_odds_sum": round(implied, 8),
            "theoretical_two_way_margin": round(1-implied, 8),
            "theoretical_fixed_stake_yield": round(1/implied-1, 8),
            "quote_origin_time_verified": False,
            "prices_simultaneously_executable_verified": False,
            "actual_arbitrage_achieved": False,
            "human_action": "RECHECK_BOTH_BOOKS_MANUALLY_NOT_A_BET",
            "provider_note": "public captured snapshots, not bookmaker-certified timestamps",
        }
        opportunities.append(opportunity)
    opportunities.sort(key=lambda x: -x["theoretical_two_way_margin"])
    disagreement_watchlist.sort(
        key=lambda x: -x["market_based_theoretical_ev"]
    )
    return {
        "schema_version": 1,
        "generated_at": stamp.isoformat(),
        "mode": "FREE_THEORETICAL_SAME_POINT_CROSS_BOOK_ARBITRAGE_RESEARCH",
        "automatic_betting_enabled": False,
        "paid_api_enabled": False,
        "model_win_probability_used": False,
        "wager_execution_implemented": False,
        "source_quote_origin_verified": False,
        "summary": {
            "input_two_way_snapshots": len(markets),
            "admissible_side_quotes": len(quotes),
            "matched_two_way_market_lines": examined,
            "theoretical_arbitrage_observations": len(opportunities),
            "leave_one_book_out_consensus_disagreements": len(disagreement_watchlist),
            "excluded_snapshots": excluded,
        },
        "limits": (
            "Pairs are captured within 5 minutes across named independent sportsbooks "
            "and must be less than 10 minutes old, both strictly pre-kickoff. "
            "They may have moved or be unfillable; neither simultaneous availability, "
            "live liquidity, stake limits, taxes nor settle/push rules are certified. "
            "Prices must be manually rechecked. No EV or future profitability inferred."
        ),
        "opportunities": opportunities,
        "disagreement_watchlist": disagreement_watchlist,
    }

def write_price_edge_report(
    markets: Sequence[ESPNTwoWayMarket],
    *,
    kickoffs: dict[str, datetime],
    as_of: datetime,
    path: str | Path = "docs/price_edge_research.json",
) -> dict[str, object]:
    report = scan_price_edges(markets, kickoffs=kickoffs, as_of=as_of)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
