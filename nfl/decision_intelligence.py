"""Downstream NFL market-consensus, disagreement, and execution-timing diagnostics.

This layer is deliberately downstream of the independent football projection. Sportsbook
prices are used to diagnose model/market disagreement and execution conditions; they are
never fed back into fair-score ratings or historical football features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from statistics import median, pstdev

import polars as pl

from .book_identity import canonical_book_identity
from .espn_market import ESPNTwoWayMarket
from .execution_market import validate_execution_row
from .market import american_to_decimal, remove_two_way_vig
from .schedule_market import is_research_only_market


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _parse_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value
    elif value in {None, ""}:
        return None
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def _side_quote(
    market: ESPNTwoWayMarket,
    side: str,
) -> tuple[float | None, int] | None:
    if market.first_side == side:
        return market.first_line, market.first_american_odds
    if market.second_side == side:
        return market.second_line, market.second_american_odds
    return None


def _home_no_vig_probability(market: ESPNTwoWayMarket) -> float | None:
    first, second = remove_two_way_vig(
        market.first_american_odds,
        market.second_american_odds,
    )
    if market.first_side == "home":
        return first
    if market.second_side == "home":
        return second
    return None


def _canonical_market_value(market: ESPNTwoWayMarket) -> float | None:
    """Return a common market coordinate suitable for cross-book consensus."""

    if market.market_type == "moneyline":
        return _home_no_vig_probability(market)
    if market.market_type == "spread":
        home = _side_quote(market, "home")
        if home is None or home[0] is None:
            return None
        # A home spread of -3.5 implies a market home margin of +3.5.
        return -float(home[0])
    if market.market_type == "total":
        if market.first_line is not None:
            return float(market.first_line)
        if market.second_line is not None:
            return float(market.second_line)
    return None


def _thresholds(
    market_type: str,
    config: Mapping[str, object],
) -> tuple[float, float, float]:
    defaults = {
        "moneyline": (0.05, 0.09, 0.035),
        "spread": (2.0, 3.5, 0.75),
        "total": (2.5, 4.5, 1.0),
    }
    medium, high, dispersion = defaults.get(market_type, (1.0, 2.0, 1.0))
    if market_type == "moneyline":
        medium = float(config.get("moneyline_disagreement_warn_probability", medium))
        high = float(config.get("moneyline_disagreement_high_probability", high))
        dispersion = float(config.get("moneyline_dispersion_warn_probability", dispersion))
    elif market_type == "spread":
        medium = float(config.get("spread_disagreement_warn_points", medium))
        high = float(config.get("spread_disagreement_high_points", high))
        dispersion = float(config.get("spread_dispersion_warn_points", dispersion))
    elif market_type == "total":
        medium = float(config.get("total_disagreement_warn_points", medium))
        high = float(config.get("total_disagreement_high_points", high))
        dispersion = float(config.get("total_dispersion_warn_points", dispersion))
    return max(0.0, medium), max(medium, high), max(0.0, dispersion)


def consensus_diagnostics(
    market_type: str,
    markets: Sequence[ESPNTwoWayMarket],
    *,
    model_home_margin: float,
    model_total: float,
    model_home_probability: float,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize verified cross-book consensus without altering the model projection."""

    cfg = config or {}
    verified = [
        market
        for market in markets
        if market.market_type == market_type and not is_research_only_market(market)
    ]
    observations: list[tuple[float, str]] = []
    for market in verified:
        value = _canonical_market_value(market)
        if value is None or not isfinite(value):
            continue
        book = canonical_book_identity(market.book or market.provider)
        observations.append((value, book))

    values = [value for value, _ in observations]
    books = sorted({book for _, book in observations if book})
    metric = {
        "moneyline": "home_no_vig_probability",
        "spread": "market_home_margin",
        "total": "market_total",
    }.get(market_type, "unknown")
    if not values:
        return {
            "market_consensus_metric": metric,
            "market_consensus_value": None,
            "market_consensus_books": 0,
            "market_consensus_verified": False,
            "market_dispersion": None,
            "market_dispersion_high": False,
            "market_disagreement": None,
            "market_disagreement_abs": None,
            "market_disagreement_severity": "UNKNOWN",
        }

    consensus = float(median(values))
    dispersion = float(pstdev(values)) if len(values) > 1 else 0.0
    if market_type == "moneyline":
        model_value = float(model_home_probability)
    elif market_type == "spread":
        model_value = float(model_home_margin)
    else:
        model_value = float(model_total)
    disagreement = model_value - consensus
    disagreement_abs = abs(disagreement)
    medium, high, dispersion_warn = _thresholds(market_type, cfg)
    severity = (
        "HIGH"
        if disagreement_abs >= high
        else "MEDIUM"
        if disagreement_abs >= medium
        else "LOW"
    )
    return {
        "market_consensus_metric": metric,
        "market_consensus_value": consensus,
        "market_consensus_books": len(books),
        "market_consensus_verified": True,
        "market_dispersion": dispersion,
        "market_dispersion_high": dispersion >= dispersion_warn,
        "market_disagreement": disagreement,
        "market_disagreement_abs": disagreement_abs,
        "market_disagreement_severity": severity,
    }


def _snapshot_side_quote(
    snapshot: Mapping[str, object],
    side: str,
) -> tuple[float | None, int] | None:
    if str(snapshot.get("first_side") or "") == side:
        line = _number(snapshot.get("first_line"))
        odds = _number(snapshot.get("first_american_odds"))
    elif str(snapshot.get("second_side") or "") == side:
        line = _number(snapshot.get("second_line"))
        odds = _number(snapshot.get("second_american_odds"))
    else:
        return None
    if odds is None:
        return None
    return line, int(round(odds))


def _latest_prior_snapshot(
    row: Mapping[str, object],
    snapshots: pl.DataFrame,
) -> dict[str, object] | None:
    if snapshots.is_empty():
        return None
    required = {"game_id", "market_type", "book", "captured_at"}
    if not required.issubset(snapshots.columns):
        return None
    current_at = _parse_utc(row.get("quant_quote_at"))
    if current_at is None:
        return None
    game_id = str(row.get("game_id") or "")
    market_type = str(row.get("quant_market") or "")
    book = str(row.get("quant_book") or "")
    side = str(row.get("quant_side") or "")
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for snapshot in snapshots.iter_rows(named=True):
        if str(snapshot.get("game_id") or "") != game_id:
            continue
        if str(snapshot.get("market_type") or "") != market_type:
            continue
        if str(snapshot.get("book") or "") != book:
            continue
        captured = _parse_utc(snapshot.get("captured_at"))
        if captured is None or captured >= current_at:
            continue
        if _snapshot_side_quote(snapshot, side) is None:
            continue
        candidates.append((captured, snapshot))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _movement(
    row: Mapping[str, object],
    prior: Mapping[str, object] | None,
) -> dict[str, object]:
    empty = {
        "timing_prior_quote_at": None,
        "timing_prior_line": None,
        "timing_prior_odds": None,
        "timing_line_value_move": None,
        "timing_odds_value_move": None,
        "timing_market_move": "NO_HISTORY",
    }
    if prior is None:
        return empty

    side = str(row.get("quant_side") or "")
    prior_quote = _snapshot_side_quote(prior, side)
    if prior_quote is None:
        return empty
    prior_line, prior_odds = prior_quote
    current_line = _number(row.get("quant_price"))
    current_odds_number = _number(row.get("quant_odds"))
    current_odds = int(round(current_odds_number)) if current_odds_number is not None else None
    market = str(row.get("quant_market") or "").lower()

    line_value_move: float | None = None
    if market == "spread" and current_line is not None and prior_line is not None:
        line_value_move = current_line - prior_line
    elif market == "total" and current_line is not None and prior_line is not None:
        if side == "over":
            line_value_move = prior_line - current_line
        elif side == "under":
            line_value_move = current_line - prior_line

    odds_value_move: float | None = None
    if current_odds is not None:
        try:
            odds_value_move = (
                american_to_decimal(current_odds)
                - american_to_decimal(prior_odds)
            )
        except ValueError:
            odds_value_move = None

    primary = line_value_move if line_value_move is not None else odds_value_move
    if primary is None or abs(primary) < 1e-12:
        direction = "STABLE"
    elif primary > 0:
        direction = "BETTOR_IMPROVED"
    else:
        direction = "BETTOR_WORSENED"
    return {
        "timing_prior_quote_at": str(prior.get("captured_at") or "") or None,
        "timing_prior_line": prior_line,
        "timing_prior_odds": prior_odds,
        "timing_line_value_move": line_value_move,
        "timing_odds_value_move": odds_value_move,
        "timing_market_move": direction,
    }


def _execution_action(
    row: Mapping[str, object],
    signal: str,
    *,
    config: Mapping[str, object],
    now: datetime | None,
    movement: Mapping[str, object],
) -> tuple[str, str]:
    normalized = signal.upper()
    if normalized == "PASS":
        return "PASS", "policy signal is PASS"

    executable, reason = validate_execution_row(row, limits=config, now=now)
    if not executable:
        return "PASS", reason

    severity = str(row.get("market_disagreement_severity") or "UNKNOWN").upper()
    if severity == "HIGH":
        return "WAIT", "high model-vs-consensus disagreement requires review"
    if bool(row.get("market_dispersion_high", False)):
        return "WAIT", "cross-book market dispersion is elevated"

    if normalized == "LEAN":
        return "WAIT", "LEAN signal remains watchlist-only for timing diagnostics"

    meaningful_line = max(
        0.0,
        float(config.get("meaningful_line_move_points", 0.5) or 0.5),
    )
    meaningful_odds = max(
        0.0,
        float(config.get("meaningful_moneyline_decimal_move", 0.05) or 0.05),
    )
    line_move = _number(movement.get("timing_line_value_move"))
    odds_move = _number(movement.get("timing_odds_value_move"))
    primary = line_move if line_move is not None else odds_move
    threshold = meaningful_line if line_move is not None else meaningful_odds

    if normalized == "STRONG":
        return "BET_NOW", "STRONG edge clears freshness and disagreement checks"
    if primary is None:
        return "BET_NOW", "BET edge is executable; no prior same-book quote is available"
    if primary <= -threshold:
        return "BET_NOW", "price has moved against the bettor while model edge remains"
    if primary >= threshold:
        return "WAIT", "price has been improving for the bettor; timing remains shadow-only"
    return "BET_NOW", "BET edge is executable and market movement is not materially favorable"


def attach_decision_intelligence(
    candidates: pl.DataFrame,
    *,
    policy: Mapping[str, object],
    snapshots: pl.DataFrame | None = None,
    now: datetime | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Attach BET_NOW/WAIT/PASS diagnostics without changing fair projections."""

    config = policy.get("decision_intelligence")
    if not isinstance(config, Mapping):
        config = {}
    portfolio = policy.get("portfolio")
    execution_limits = portfolio if isinstance(portfolio, Mapping) else {}
    history = snapshots if snapshots is not None else pl.DataFrame()
    enforced = bool(config.get("enforce_execution_timing", False))

    if candidates.is_empty():
        return candidates, {
            "status": "READY",
            "rows": 0,
            "enforced": enforced,
            "execution_actions": {},
            "research_execution_actions": {},
            "movement_coverage": 0.0,
            "high_disagreement_rows": 0,
        }

    rows: list[dict[str, object]] = []
    action_counts: dict[str, int] = {"BET_NOW": 0, "WAIT": 0, "PASS": 0}
    research_counts: dict[str, int] = {"BET_NOW": 0, "WAIT": 0, "PASS": 0}
    movement_rows = 0
    high_disagreement = 0
    for source in candidates.to_dicts():
        row = dict(source)
        prior = _latest_prior_snapshot(row, history)
        movement = _movement(row, prior)
        row.update(movement)
        if movement["timing_market_move"] != "NO_HISTORY":
            movement_rows += 1
        if str(row.get("market_disagreement_severity") or "").upper() == "HIGH":
            high_disagreement += 1

        quant_action, quant_reason = _execution_action(
            row,
            str(row.get("quant_signal") or "PASS"),
            config=execution_limits,
            now=now,
            movement=movement,
        )
        research_action, research_reason = _execution_action(
            row,
            str(row.get("research_signal") or "PASS"),
            config=execution_limits,
            now=now,
            movement=movement,
        )
        row.update(
            {
                "execution_action": quant_action,
                "execution_action_reason": quant_reason,
                "research_execution_action": research_action,
                "research_execution_action_reason": research_reason,
                "execution_timing_enforced": enforced,
            }
        )
        action_counts[quant_action] = action_counts.get(quant_action, 0) + 1
        research_counts[research_action] = research_counts.get(research_action, 0) + 1
        rows.append(row)

    return pl.DataFrame(rows), {
        "status": "READY",
        "rows": len(rows),
        "enforced": enforced,
        "execution_actions": action_counts,
        "research_execution_actions": research_counts,
        "movement_coverage": movement_rows / len(rows),
        "high_disagreement_rows": high_disagreement,
        "meaning": (
            "shadow execution timing and model/market diagnostics; market prices remain "
            "downstream of the independent fair-score projection"
        ),
    }
