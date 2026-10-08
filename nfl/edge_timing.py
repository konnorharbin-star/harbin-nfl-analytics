"""NFL timing intelligence v2: timestamp-safe same-book, research-only guidance.

Never overrides fair-score projections, market/regime evidence, production
release gates, portfolio signals or stakes. Snapshot capture timestamps are
observation times, NOT proof of sportsbook publication or executable fills.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from .book_identity import canonical_book_identity
from .edge_discovery import EVIDENCE_SUPPORTED
from .execution_market import validate_execution_row
from .market import american_implied_probability

SCHEMA_VERSION = 1
ACTIONS = ("BET_NOW_RESEARCH", "WAIT_MONITOR", "PASS")
MOVEMENTS = ("WORSENED", "IMPROVED", "MIXED_LINE_PRICE", "STABLE", "NO_HISTORY")
MAX_LOOKBACK_HOURS = 12
MIN_OBSERVATION_GAP_MINUTES = 10
MIN_LINE_MOVE_POINTS = 0.5
MIN_PRICE_MOVE_PP = 1.0
MIN_SAME_LINE_EV_CUSHION = 0.01
MIN_SAME_LINE_PROB_CUSHION = 0.02
EXTRA_COLUMNS = (
    "edge_timing_action", "edge_timing_reason", "edge_timing_status",
    "edge_timing_evidence_source", "edge_timing_prior_captured_at",
    "edge_timing_prior_book", "edge_timing_prior_line",
    "edge_timing_prior_odds", "edge_timing_quote_age_minutes",
    "edge_timing_observation_gap_minutes", "edge_timing_line_change_points",
    "edge_timing_price_change_pp", "edge_timing_market_move",
    "edge_timing_reference_line", "edge_timing_same_line_min_american_odds",
    "edge_timing_same_line_break_even_ceiling",
    "edge_timing_bet_to_line", "edge_timing_staking_authorized",
)

def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _parse(value: object) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value
    elif value is None or value == "":
        return None
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    # Unzoned historical capture values cannot establish reliable time order.
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(UTC)


def _american(value: object) -> int | None:
    number = _number(value)
    if number is None or int(number) != number:
        return None
    result = int(number)
    return result if result >= 100 or result <= -100 else None


def _snap_side(snapshot: Mapping[str, object], side: str, market: str):
    if str(snapshot.get("first_side") or "") == side:
        line = _number(snapshot.get("first_line"))
        odds = _american(snapshot.get("first_american_odds"))
    elif str(snapshot.get("second_side") or "") == side:
        line = _number(snapshot.get("second_line"))
        odds = _american(snapshot.get("second_american_odds"))
    else:
        return None
    if odds is None or (market != "moneyline" and line is None):
        return None
    if market == "moneyline" and line is not None:
        return None
    return line, odds


def _history_index(snapshots: pl.DataFrame):
    indexed: dict[tuple[str, str, str], list[tuple[datetime, dict[str, object]]]] = {}
    required = {"game_id", "market_type", "book", "captured_at", "kickoff"}
    if snapshots.is_empty() or not required.issubset(snapshots.columns):
        return indexed
    for snapshot in snapshots.iter_rows(named=True):
        captured = _parse(snapshot.get("captured_at"))
        kickoff = _parse(snapshot.get("kickoff"))
        book = canonical_book_identity(snapshot.get("book"))
        game = str(snapshot.get("game_id") or "")
        market = str(snapshot.get("market_type") or "").lower()
        if (captured is None or kickoff is None or captured >= kickoff
                or not game or not book or market not in {"moneyline", "spread", "total"}):
            continue
        item = dict(snapshot)
        item["_kickoff_utc"] = kickoff
        key = (game, market, book)
        indexed.setdefault(key, []).append((captured, item))
    for collection in indexed.values():
        collection.sort(key=lambda item: item[0])
    return indexed


def _prior(row: Mapping[str, object], index, as_of: datetime):
    now_quote = _parse(row.get("quant_quote_at"))
    kickoff = _parse(row.get("kickoff"))
    if now_quote is None or kickoff is None or now_quote >= kickoff:
        return None
    key = (
        str(row.get("game_id") or ""),
        str(row.get("quant_market") or "").lower(),
        canonical_book_identity(row.get("quant_book")),
    )
    minimum = now_quote - timedelta(hours=MAX_LOOKBACK_HOURS)
    maximum = now_quote - timedelta(minutes=MIN_OBSERVATION_GAP_MINUTES)
    market = key[1]
    side = str(row.get("quant_side") or "")
    for captured, snap in reversed(index.get(key, ())):
        if captured > maximum:
            continue
        if captured < minimum:
            break
        if captured > as_of or captured >= kickoff:
            continue
        if abs((snap["_kickoff_utc"] - kickoff).total_seconds()) > 60:
            continue
        found = _snap_side(snap, side, market)
        if found is not None:
            return captured, snap, found
    return None


def _movement(market: str, side: str, old_line, old_odds, new_line, new_odds):
    line_change = None
    if market == "spread" and old_line is not None and new_line is not None:
        line_change = new_line - old_line
    elif market == "total" and old_line is not None and new_line is not None:
        line_change = old_line - new_line if side == "over" else new_line - old_line
    pp = 100 * (
        american_implied_probability(old_odds) - american_implied_probability(new_odds)
    )
    line_good = line_change is not None and line_change >= MIN_LINE_MOVE_POINTS
    line_bad = line_change is not None and line_change <= -MIN_LINE_MOVE_POINTS
    price_good = pp >= MIN_PRICE_MOVE_PP
    price_bad = pp <= -MIN_PRICE_MOVE_PP
    if (line_good and price_bad) or (line_bad and price_good):
        status = "MIXED_LINE_PRICE"
    elif line_good or price_good:
        status = "IMPROVED"
    elif line_bad or price_bad:
        status = "WORSENED"
    else:
        status = "STABLE"
    return status, line_change, pp


def _same_line_minimum_odds(probability: object):
    """Require +1% model EV and +2pp cushion over implied break-even, same line.

    The selected market's held-out-shrunk probability is valid ONLY at its
    current line. Changing the spread/total requires a fresh football
    probability; never extrapolate it to a different betting line.
    """
    p = _number(probability)
    if p is None or not 0 < p < 1 or p <= MIN_SAME_LINE_PROB_CUSHION:
        return None, None
    max_implied = min(
        p - MIN_SAME_LINE_PROB_CUSHION,
        p / (1 + MIN_SAME_LINE_EV_CUSHION),
    )
    if not 0 < max_implied < 1:
        return None, None
    decimal_min = 1 / max_implied
    if decimal_min >= 2:
        american = max(100, math.ceil(100 * (decimal_min - 1) - 1e-10))
    else:
        american = min(-100, math.ceil(-100 / (decimal_min - 1) - 1e-10))
    return int(american), max_implied


def _guide(
    row: Mapping[str, object],
    index,
    policy: Mapping[str, object],
    as_of: datetime,
):
    action = "PASS"
    reason = "NO_ELIGIBLE_OBSERVED_EDGE"
    market = str(row.get("quant_market") or "").lower()
    book = canonical_book_identity(row.get("quant_book"))
    quote_at = _parse(row.get("quant_quote_at"))
    kickoff = _parse(row.get("kickoff"))
    current_line = _number(row.get("quant_price"))
    current_odds = _american(row.get("quant_odds"))
    quote_age = (as_of - quote_at).total_seconds() / 60 if quote_at else None
    min_odds, max_implied = (None, None)
    alpha = _number(row.get("edge_validated_shrinkage_alpha"))
    if (
        alpha is not None and alpha > 0
        and str(row.get("edge_discovery_tier") or "") == EVIDENCE_SUPPORTED
    ):
        min_odds, max_implied = _same_line_minimum_odds(
            row.get("edge_shrunk_probability")
        )
    result = {
        "edge_timing_action": action,
        "edge_timing_reason": reason,
        "edge_timing_status": "RESEARCH_ONLY",
        "edge_timing_evidence_source": "observed_capture_not_official_close",
        "edge_timing_prior_captured_at": None,
        "edge_timing_prior_book": None,
        "edge_timing_prior_line": None,
        "edge_timing_prior_odds": None,
        "edge_timing_quote_age_minutes": quote_age,
        "edge_timing_observation_gap_minutes": None,
        "edge_timing_line_change_points": None,
        "edge_timing_price_change_pp": None,
        "edge_timing_market_move": "NO_HISTORY",
        "edge_timing_reference_line": current_line,
        "edge_timing_same_line_min_american_odds": min_odds,
        "edge_timing_same_line_break_even_ceiling": max_implied,
        "edge_timing_bet_to_line": None,
        "edge_timing_staking_authorized": False,
    }
    # Observe a historical movement even if the live signal is disallowed,
    # provided both captured prices have valid, pre-kickoff time provenance.
    matched = _prior(row, index, as_of)
    if matched is not None and current_odds is not None:
        old_at, snap, (old_line, old_odds) = matched
        status, line_change, price_change = _movement(
            market, str(row.get("quant_side") or ""),
            old_line, old_odds, current_line, current_odds,
        )
        result.update({
            "edge_timing_prior_captured_at": old_at.isoformat(),
            "edge_timing_prior_book": snap.get("book"),
            "edge_timing_prior_line": old_line,
            "edge_timing_prior_odds": old_odds,
            "edge_timing_observation_gap_minutes": (
                (quote_at - old_at).total_seconds() / 60 if quote_at else None
            ),
            "edge_timing_line_change_points": line_change,
            "edge_timing_price_change_pp": price_change,
            "edge_timing_market_move": status,
        })
    limits = policy.get("portfolio")
    cfg = limits if isinstance(limits, Mapping) else {}
    verified, quote_reason = validate_execution_row(row, limits=cfg, now=as_of)
    if (
        not verified or row.get("market_execution_verified") is not True
        or row.get("market_quote_timestamp_verified") is not True
        or row.get("market_quote_sanity_ok") is not True
        or current_odds is None or not book or kickoff is None
    ):
        reason = "NO_FRESH_VERIFIED_SANE_OFFER: " + quote_reason
    elif row.get("edge_discovery_tier") != EVIDENCE_SUPPORTED:
        reason = "NO_VALIDATED_NFL_EDGE: "
        reason += str(row.get("edge_discovery_tier") or "MISSING_EVIDENCE")
    elif (
        row.get("qb_certainty_veto") is True
        or row.get("context_freshness_veto") is True
        or row.get("context_veto") is True
        or row.get("context_injuries_personnel_fresh") is not True
        or row.get("probability_reliability_ready") is not True
        or row.get("regime_reliability_ready") is not True
    ):
        reason = "NFL_CONTEXT_OR_PROBABILITY_VETO"
    elif (
        row.get("market_dispersion_high") is True
        or str(row.get("market_disagreement_severity") or "").upper() == "HIGH"
    ):
        reason = "HIGH_CROSS_BOOK_DISAGREEMENT"
    elif min_odds is None or max_implied is None:
        reason = "NO_REPRICED_HOLDOUT_SUPPORTED_SAME_LINE_PRICE_LIMIT"
    elif (
        american_implied_probability(current_odds) > max_implied + 1e-10
        or (current_line is None and market != "moneyline")
    ):
        reason = "CURRENT_SAME_LINE_PRICE_BREACHES_CONSERVATIVE_LIMIT"
    elif matched is None:
        reason = "NO_RECENT_PRIOR_SAME_BOOK_OBSERVATION"
    elif result["edge_timing_market_move"] == "MIXED_LINE_PRICE":
        reason = "MIXED_LINE_ODDS_MOVEMENT_CANNOT_SCORE_DIRECTION"
    elif result["edge_timing_market_move"] == "WORSENED":
        action, reason = "BET_NOW_RESEARCH", "SAME_BOOK_OBSERVED_PRICE_DETERIORATION"
    elif result["edge_timing_market_move"] == "IMPROVED":
        action, reason = "WAIT_MONITOR", "SAME_BOOK_OBSERVED_PRICE_IMPROVEMENT"
    else:
        reason = "NO_MATERIAL_SAME_BOOK_MOVEMENT"
    result["edge_timing_action"] = action
    result["edge_timing_reason"] = reason
    return result


def enrich_edge_timing(
    frame: pl.DataFrame,
    snapshots: pl.DataFrame,
    *,
    policy: Mapping[str, object],
    as_of: datetime | None = None,
):
    """Compute shadow advice, never modify any existing financial or model field."""
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Timing as_of requires timezone")
    now = now.astimezone(UTC)
    index = _history_index(snapshots)
    rows = []
    for original in frame.iter_rows(named=True):
        row = dict(original)
        row.update(_guide(row, index, policy, now))
        rows.append(row)
    enriched = pl.DataFrame(rows) if rows else frame.clone()
    counts = Counter(row["edge_timing_action"] for row in rows)
    moves = Counter(row["edge_timing_market_move"] for row in rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "RESEARCH_ONLY",
        "league": "NFL",
        "as_of": now.isoformat(),
        "rows": len(rows),
        "actions": {key: counts.get(key, 0) for key in ACTIONS},
        "observed_movements": {key: moves.get(key, 0) for key in MOVEMENTS},
        "same_book_prior_coverage": (
            sum(row["edge_timing_prior_captured_at"] is not None for row in rows) / len(rows)
            if rows else 0.0
        ),
        "qualified_research_signals": counts.get("BET_NOW_RESEARCH", 0)
        + counts.get("WAIT_MONITOR", 0),
        "policy_and_stake_unchanged": True,
        "production_stake_authorized": False,
        "validated_timing_strategy": False,
        "observed_quote_not_guaranteed_fill": True,
        "not_official_closing_line": True,
        "limits": {
            "lookback_hours": MAX_LOOKBACK_HOURS,
            "minimum_observation_gap_minutes": MIN_OBSERVATION_GAP_MINUTES,
            "material_line_points": MIN_LINE_MOVE_POINTS,
            "material_implied_break_even_move_pp": MIN_PRICE_MOVE_PP,
            "minimum_same_line_EV_cushion": MIN_SAME_LINE_EV_CUSHION,
            "minimum_same_line_probability_cushion": MIN_SAME_LINE_PROB_CUSHION,
        },
        "notes": [
            "A prior snapshot is a capture timestamp, not verified original book publish time.",
            "Prices are matched only within the same game, market, side and canonical book.",
            "No bet-to spread or total is inferred by carrying probability across lines.",
            "Mixed adverse/favorable spread and odds moves are unscored.",
            "An absent prior quote never establishes a BET_NOW timing advantage.",
            "NFL holdout alpha=0 and unreliable regime markets fail closed.",
            "Forward validation is required; observed price movement is not economic value.",
        ],
    }
    return enriched, report


def write_edge_timing(
    frame: pl.DataFrame,
    report: Mapping[str, object],
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
):
    columns = (
        "season", "week", "game_id", "kickoff", "away_team", "home_team",
        "quant_market", "quant_side", "quant_book", "quant_price", "quant_odds",
        "quant_quote_at", "quant_probability", "quant_ev", "quant_edge",
        "edge_discovery_tier", "research_signal", "quant_signal",
        "portfolio_action", "portfolio_stake_units", *EXTRA_COLUMNS,
    )
    rows = frame.to_dicts() if not frame.is_empty() else []
    rows.sort(key=lambda r: (
        ACTIONS.index(str(r.get("edge_timing_action")))
        if str(r.get("edge_timing_action")) in ACTIONS else 99,
        str(r.get("game_id") or ""), str(r.get("quant_market") or ""),
    ))
    for folder in (Path(output_dir), Path(docs_dir)):
        folder.mkdir(parents=True, exist_ok=True)
        for name, selection in (
            ("edge_timing.csv", rows),
            ("edge_timing_signals.csv", [
                r for r in rows if r["edge_timing_action"] != "PASS"
            ]),
        ):
            with (folder / name).open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in selection:
                    writer.writerow({field: row.get(field) for field in columns})
        (folder / "edge_timing_report.json").write_text(
            json.dumps(dict(report), sort_keys=True, indent=2)
        )
