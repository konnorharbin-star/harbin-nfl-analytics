"""Generic, free NFL market-universe and bookmaker-disagreement research.

Covers game lines, alternate lines, halves/quarters, team totals and player
props *when* a permissible source actually supplies prices. Missing prices
remain COVERAGE_MISSING; source-capture time never becomes quote-origin time.
A consensus disagreement is NOT verified betting EV or an executable wager.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from math import isfinite
from statistics import median
from typing import Any

from .book_identity import canonical_book_identity
from .espn_market import ESPNTwoWayMarket
from .market import american_to_decimal

# These are discoverable product/market families, NOT claims of free live feeds.
MARKET_CATALOG: dict[str, tuple[str, ...]] = {
    "full_game": (
        "moneyline", "spread", "total", "team_total", "alt_spread", "alt_total",
    ),
    "first_half": ("moneyline", "spread", "total", "team_total"),
    "second_half": ("moneyline", "spread", "total", "team_total"),
    "first_quarter": ("moneyline", "spread", "total", "team_total"),
    "second_quarter": ("moneyline", "spread", "total"),
    "third_quarter": ("moneyline", "spread", "total"),
    "fourth_quarter": ("moneyline", "spread", "total"),
    "player": (
        "passing_yards", "passing_touchdowns", "passing_attempts",
        "pass_completions", "interceptions", "rushing_yards",
        "rushing_attempts", "receiving_yards", "receptions",
        "anytime_touchdown", "longest_reception", "longest_rush",
        "field_goals_made",
    ),
}
PERIOD_NAMES = frozenset(MARKET_CATALOG)
BINARY = {"moneyline": ("home", "away"), "spread": ("home", "away"),
          "alt_spread": ("home", "away")}
for markets in MARKET_CATALOG.values():
    for name in markets:
        if name not in BINARY:
            BINARY[name] = ("yes", "no") if name == "anytime_touchdown" else ("over", "under")
REQUIRES_SUBJECT = frozenset(("team_total", "anytime_touchdown",
                              "passing_yards", "passing_touchdowns",
                              "passing_attempts", "pass_completions",
                              "interceptions", "rushing_yards",
                              "rushing_attempts", "receiving_yards",
                              "receptions", "longest_reception",
                              "longest_rush", "field_goals_made"))
NO_POINT_LINE = frozenset(("moneyline", "anytime_touchdown"))
MAX_QUOTE_AGE = timedelta(minutes=10)
MAX_CAPTURE_GAP = timedelta(minutes=3)
MAX_REFERENCE_GAP = timedelta(minutes=5)
MIN_OTHER_BOOKS = 3
MAX_REFERENCE_RANGE = 0.025
MIN_PROXY_EV = .05


def aware(value: object) -> datetime | None:
    if isinstance(value, datetime):
        t = value
    elif isinstance(value, str):
        try:
            t = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return t.astimezone(UTC) if t.tzinfo else None


def number(value: object) -> float | None:
    try:
        n = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return n if isfinite(n) else None


def _is_offer_valid(item: dict[str, Any], now: datetime,
                    starts: dict[str, datetime]) -> tuple[dict[str, Any] | None, str]:
    game = str(item.get("game_id") or "").strip()
    period = str(item.get("period") or "")
    market = str(item.get("market") or "")
    side = str(item.get("side") or "").casefold()
    if not game or period not in PERIOD_NAMES or market not in MARKET_CATALOG[period]:
        return None, "UNSUPPORTED_OR_UNIDENTIFIED_MARKET"
    if side not in BINARY[market]:
        return None, "INVALID_MARKET_SIDE"
    subject = str(item.get("subject_id") or "").strip()
    if market in REQUIRES_SUBJECT and not subject:
        return None, "MISSING_SUBJECT_ID"
    if market not in REQUIRES_SUBJECT and subject:
        return None, "UNEXPECTED_SUBJECT_ID"
    pt = number(item.get("line"))
    if market in NO_POINT_LINE:
        if item.get("line") is not None:
            return None, "UNEXPECTED_POINT_LINE"
    elif pt is None or abs(pt) > 2500:
        return None, "MISSING_OR_INVALID_POINT_LINE"
    book = canonical_book_identity(item.get("book"))
    event = str(item.get("source_event_id") or "").strip()
    provider = str(item.get("provider") or "").strip()
    if not book or not event or provider not in ("espn", "action_network"):
        return None, "INVALID_FREE_PROVIDER_OR_BOOK"
    odds = number(item.get("american_odds"))
    if odds is None or not odds.is_integer() or abs(odds) < 100 or abs(odds) > 10000:
        return None, "INVALID_AMERICAN_ODDS"
    capture = aware(item.get("captured_at"))
    origin = aware(item.get("source_quote_at"))
    start = aware(starts.get(game))
    if start is None or capture is None or origin is None:
        return None, "MISSING_KICKOFF_CAPTURE_OR_ORIGIN"
    if not item.get("source_offer_timestamp_verified") is True:
        return None, "UNVERIFIED_BOOK_QUOTE_ORIGIN"
    if (origin > capture + timedelta(minutes=2)
        or capture-origin > MAX_QUOTE_AGE or now-origin > MAX_QUOTE_AGE
        or capture > now or origin > now or capture >= start or origin >= start
        or now >= start):
        return None, "STALE_OR_NONPREGAME_OFFER"
    try:
        decimal = american_to_decimal(int(odds))
    except (ValueError, TypeError, OverflowError):
        return None, "INVALID_AMERICAN_ODDS"
    if not isfinite(decimal) or decimal <= 1:
        return None, "INVALID_DECIMAL_ODDS"
    return {
        "game_id": game, "period": period, "market": market,
        "subject_id": subject, "line": pt, "side": side,
        "book": str(item["book"]), "book_key": book,
        "provider": provider, "source_event_id": event,
        "american_odds": int(odds), "decimal_odds": decimal,
        "captured_at": capture, "source_quote_at": origin,
    }, "ACCEPTED"


def from_game_markets(markets: list[ESPNTwoWayMarket]) -> list[dict[str, Any]]:
    """Convert already-collected 2-way game lines; do not forge price updates.

    For spread, line is the *home-team* handicap on BOTH sides, so opposite
    sides across books only compare at the exact same handicap.
    """
    offers = []
    for market in markets:
        if market.market_type not in ("moneyline", "spread", "total"):
            continue
        if market.market_type == "spread":
            home_line = (
                market.first_line if market.first_side == "home"
                else market.second_line
            )
        else:
            home_line = None
        for side, point, odds in (
            (market.first_side, market.first_line, market.first_american_odds),
            (market.second_side, market.second_line, market.second_american_odds),
        ):
            if market.market_type == "spread":
                point = home_line
            if market.market_type == "moneyline":
                point = None
            offers.append({
                "game_id": market.game_id, "period": "full_game",
                "market": market.market_type, "subject_id": "",
                "side": side, "line": point, "book": market.book,
                "american_odds": odds, "provider": market.provider,
                "source_event_id": market.source_event_id,
                "captured_at": market.captured_at,
                "source_quote_at": market.source_quote_at,
                "source_offer_timestamp_verified":
                    market.source_quote_time_verified,
            })
    return offers


def evaluate_market_universe(
    offers: list[dict[str, Any]], *,
    kickoffs: dict[str, datetime], as_of: datetime,
) -> dict[str, Any]:
    """Look at EVERY supplied market, not just a model-picked side.

    Only qualified, individual-book, complementary two-way quotes enter the
    leave-one-book-out comparison. This is a price proxy, not estimated truth.
    """
    now = aware(as_of)
    if now is None:
        raise ValueError("as_of requires timezone")
    exclusions = Counter()
    valid = []
    presence = Counter()
    for raw in offers:
        if isinstance(raw, dict):
            name = (str(raw.get("period") or ""), str(raw.get("market") or ""))
            if name[0] in MARKET_CATALOG and name[1] in MARKET_CATALOG[name[0]]:
                presence[name] += 1
            item, reason = _is_offer_valid(raw, now, kickoffs)
        else:
            item, reason = None, "MALFORMED_OFFER"
        if item is None:
            exclusions[reason] += 1
        else:
            valid.append(item)

    # Require a unique, non-conflicting latest two-side quote per sportsbook.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        key = (row["game_id"], row["period"], row["market"],
               row["subject_id"], row["line"])
        groups[key].append(row)

    candidates = []
    eligible_markets = 0
    for key, market_rows in sorted(groups.items()):
        game, period, market, subject, line = key
        # Each book must have both complementary sides and origin timestamps
        # corresponding to near-contemporaneous offers.
        by_book = defaultdict(list)
        for row in market_rows:
            by_book[row["book_key"]].append(row)
        pairs = {}
        for book, records in by_book.items():
            q = {}
            valid_pair = True
            for side in BINARY[market]:
                found = sorted(
                    [r for r in records if r["side"] == side],
                    key=lambda x: (x["source_quote_at"],x["captured_at"]),
                    reverse=True,
                )
                if not found:
                    valid_pair = False
                    break
                best = found[0]
                if any(
                    x["source_quote_at"] == best["source_quote_at"]
                    and x["american_odds"] != best["american_odds"]
                    for x in found[1:]
                ):
                    valid_pair = False
                    break
                q[side] = best
            if not valid_pair:
                exclusions["INCOMPLETE_OR_CONFLICTING_BOOK_PAIR"] += 1
                continue
            a,b=(q[x] for x in BINARY[market])
            if abs(a["source_quote_at"]-b["source_quote_at"]) > MAX_CAPTURE_GAP:
                exclusions["NONCONTEMPORANEOUS_TWO_SIDES"] += 1
                continue
            pairs[book] = q
        if len(pairs) < MIN_OTHER_BOOKS + 1:
            continue
        eligible_markets += 1
        for book, pair in pairs.items():
            for side in BINARY[market]:
                target = pair[side]
                comparisons = []
                for other, opponent in pairs.items():
                    if other == book:
                        continue
                    a,b = (opponent[x] for x in BINARY[market])
                    if abs(a["source_quote_at"] -
                           target["source_quote_at"]) > MAX_REFERENCE_GAP:
                        continue
                    if abs(b["source_quote_at"] -
                           target["source_quote_at"]) > MAX_REFERENCE_GAP:
                        continue
                    p_a = 1 / a["decimal_odds"]
                    p_b = 1 / b["decimal_odds"]
                    total = p_a + p_b
                    # Invalid/crossed or absent overround is suspicious, not
                    # a reliable estimate of a no-vig market probability.
                    if not .98 <= total <= 1.25:
                        continue
                    probability = p_a / total if side == BINARY[market][0] else p_b / total
                    comparisons.append(probability)
                if len(comparisons) < MIN_OTHER_BOOKS:
                    continue
                if max(comparisons)-min(comparisons) > MAX_REFERENCE_RANGE:
                    continue
                center = median(comparisons)
                conservative = min(comparisons)
                proxy_ev = center * target["decimal_odds"] - 1
                lower_proxy_ev = conservative * target["decimal_odds"] - 1
                if lower_proxy_ev < MIN_PROXY_EV:
                    continue
                candidates.append({
                    "game_id": game, "period": period, "market": market,
                    "subject_id": subject, "line": line, "side": side,
                    "book": target["book"], "american_odds": target["american_odds"],
                    "source_quote_at": target["source_quote_at"].isoformat(),
                    "other_books": len(comparisons),
                    "median_other_books_no_vig_probability": round(center, 6),
                    "lowest_other_books_no_vig_probability": round(conservative, 6),
                    "no_vig_reference_range": round(max(comparisons)-min(comparisons), 6),
                    "theoretical_consensus_proxy_ev": round(proxy_ev, 6),
                    "conservative_consensus_proxy_ev": round(lower_proxy_ev, 6),
                    "status": "RESEARCH_ONLY_RECHECK_PRICE",
                    "model_calibration_verified": False,
                    "true_probability_verified": False,
                    "price_executable_verified": False,
                    "bet_placed": False,
                })
    candidates.sort(key=lambda v: (-v["conservative_consensus_proxy_ev"],
                                   v["game_id"],v["market"],v["side"]))
    catalog = [
        {
            "period": period, "market": market,
            "supplied_offer_count": presence[(period, market)],
            "coverage": ("HAS_PRICES" if presence[(period, market)] else
                         "NO_FREE_SOURCE_OBSERVED"),
        }
        for period, families in MARKET_CATALOG.items() for market in families
    ]
    return {
        "schema_version": 1,
        "mode": "ALL_MARKETS_PRICE_DISCOVERY_RESEARCH",
        "generated_at_utc": now.isoformat(),
        "automatic_betting_enabled": False,
        "paid_feeds_required": False,
        "current_price_execution_verified": False,
        "true_market_mispricing_proven": False,
        "summary": {
            "market_families_cataloged": len(catalog),
            "supplied_offers": len(offers),
            "origin_valid_offers": len(valid),
            "multi_book_two_way_markets": eligible_markets,
            "research_price_discrepancies": len(candidates),
            "markets_without_supplied_source": sum(
                entry["coverage"]=="NO_FREE_SOURCE_OBSERVED" for entry in catalog
            ),
        },
        "exclusions": dict(sorted(exclusions.items())),
        "market_coverage": catalog,
        "research_candidates": candidates,
        "limitations": (
            "A cross-book median is an imperfect correlated-price reference, "
            "not an independently validated probability. Never call its proxy "
            "EV a true edge. Push/tie handling and book-specific settlement "
            "rules may differ. No fabricated player props or period markets: "
            "unprovided prices are reported missing, never filled by guesses. "
            "Per-book source updates still may not certify executable prices."
        ),
    }


def write_market_universe(
    offers: list[dict[str, Any]], *,
    kickoffs: dict[str, datetime], as_of: datetime,
    output: str = "docs/market_universe.json",
) -> dict[str, Any]:
    """Publish all covered and missing markets with no bet execution."""
    import json
    from pathlib import Path

    report = evaluate_market_universe(
        offers, kickoffs=kickoffs, as_of=as_of
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report
