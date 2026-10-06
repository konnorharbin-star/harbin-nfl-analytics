"""CFB-style execution, bankroll, and concentration controls for NFL candidates."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import polars as pl

from .execution_market import recommendation_freshness, validate_execution_row
from .performance_feedback import build_performance_feedback, performance_feedback_for_row
from .policy import load_policy


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if isfinite(result) else float(default)


def _read_json(path: str | Path, fallback: dict[str, object]) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return dict(fallback)
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return dict(fallback)
    return value if isinstance(value, dict) else dict(fallback)


def build_bankroll_risk_state(
    live_bets_path: str | Path = "reports/live_graded_bets.csv",
    limits: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the same unit-based drawdown throttle used conceptually by CFB."""

    config = limits or {}
    soft = max(0.0, _number(config.get("drawdown_soft_stop_units"), 8.0))
    hard = max(soft + 1e-9, _number(config.get("drawdown_hard_stop_units"), 15.0))
    floor = min(1.0, max(0.0, _number(config.get("drawdown_floor_multiplier"), 0.25)))
    trail_n = max(10, int(_number(config.get("trailing_window_bets"), 50)))
    min_trail = max(
        10,
        int(_number(config.get("min_trailing_bets_for_throttle"), 30)),
    )
    roi_trigger = _number(config.get("trailing_roi_throttle"), -0.10)
    clv_trigger = _number(config.get("trailing_clv_throttle"), 0.0)
    adverse = min(
        1.0,
        max(0.0, _number(config.get("adverse_run_multiplier"), 0.50)),
    )

    source = Path(live_bets_path)
    empty = {
        "history_available": False,
        "graded_bets": 0,
        "cumulative_units": 0.0,
        "current_drawdown_units": 0.0,
        "max_drawdown_units": 0.0,
        "trailing_window_bets": 0,
        "trailing_roi": None,
        "trailing_avg_clv": None,
        "risk_multiplier": 1.0,
        "hard_stop": False,
        "reason": "no independent graded NFL live/shadow history",
    }
    if not source.exists():
        return empty

    try:
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {**empty, "reason": "live betting ledger unreadable"}

    graded = []
    for row in rows:
        value = row.get("net_units", row.get("profit"))
        try:
            units = float(value) if value not in {None, ""} else None
        except (TypeError, ValueError):
            units = None
        if units is not None and isfinite(units):
            graded.append((row, units))
    if not graded:
        return empty

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for _, units in graded:
        cumulative += units
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    current_drawdown = max(0.0, peak - cumulative)

    tail = graded[-trail_n:]
    trailing_roi = sum(units for _, units in tail) / len(tail)
    clv_values: list[float] = []
    for row, _ in tail:
        for key in ("execution_clv", "clv_proxy", "probability_clv", "clv"):
            raw = row.get(key)
            try:
                value = float(raw) if raw not in {None, ""} else None
            except (TypeError, ValueError):
                value = None
            if value is not None and isfinite(value):
                clv_values.append(value)
                break
    trailing_clv = sum(clv_values) / len(clv_values) if clv_values else None

    hard_stop = current_drawdown >= hard - 1e-12
    multiplier = 0.0 if hard_stop else 1.0
    reasons: list[str] = []
    if hard_stop:
        reasons.append(f"drawdown {current_drawdown:.2f}u reached hard stop {hard:.2f}u")
    elif current_drawdown > soft:
        fraction = min(1.0, (current_drawdown - soft) / (hard - soft))
        multiplier = 1.0 - fraction * (1.0 - floor)
        reasons.append("drawdown throttle active")

    if (
        not hard_stop
        and len(tail) >= min_trail
        and trailing_roi <= roi_trigger
        and trailing_clv is not None
        and trailing_clv <= clv_trigger
    ):
        multiplier *= adverse
        reasons.append("recent ROI and CLV are jointly adverse")

    return {
        "history_available": True,
        "graded_bets": len(graded),
        "cumulative_units": round(cumulative, 6),
        "current_drawdown_units": round(current_drawdown, 6),
        "max_drawdown_units": round(max_drawdown, 6),
        "trailing_window_bets": len(tail),
        "trailing_roi": round(trailing_roi, 6),
        "trailing_avg_clv": None if trailing_clv is None else round(trailing_clv, 6),
        "risk_multiplier": round(min(1.0, max(0.0, multiplier)), 6),
        "hard_stop": hard_stop,
        "reason": "; ".join(reasons) if reasons else "no bankroll throttle active",
    }


def _kickoff_bucket(value: object, hours: int) -> str:
    if value is None:
        return "unknown"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    seconds = max(1, hours) * 3600
    epoch = int(stamp.timestamp())
    return str(epoch - (epoch % seconds))


def _team_keys(row: Mapping[str, object]) -> list[str]:
    market = str(row.get("quant_market") or "").lower()
    side = str(row.get("quant_side") or "")
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    if market in {"moneyline", "spread"}:
        if side == "home":
            return [home] if home else []
        if side == "away":
            return [away] if away else []
    if market == "total":
        return [team for team in (home, away) if team]
    return []


def _parse_utc(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def load_committed_production_exposure(
    path: str | Path,
    candidate_rows: list[dict[str, object]],
    *,
    now: datetime | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load previously authorized, still-open BET decisions for target weeks."""

    target_periods: set[tuple[int, int]] = set()
    for row in candidate_rows:
        try:
            target_periods.add((int(row["season"]), int(row["week"])))
        except (KeyError, TypeError, ValueError):
            continue

    source = Path(path)
    base = {
        "ledger_available": False,
        "integrity_ok": False,
        "open_bets": 0,
        "open_units": 0.0,
        "invalid_rows": 0,
        "target_periods": [list(value) for value in sorted(target_periods)],
        "reason": "committed-exposure ledger is unavailable",
    }
    if not source.exists():
        return [], base

    try:
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return [], {**base, "reason": "committed-exposure ledger is unreadable"}

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    reference = reference.astimezone(UTC)

    open_rows: list[dict[str, object]] = []
    invalid_rows = 0
    for row in rows:
        if str(row.get("portfolio_action") or "").upper() != "BET":
            continue
        units = _number(row.get("portfolio_stake_units"))
        if units <= 0:
            continue
        try:
            period = (int(row["season"]), int(row["week"]))
        except (KeyError, TypeError, ValueError):
            invalid_rows += 1
            continue
        if target_periods and period not in target_periods:
            continue
        kickoff = _parse_utc(row.get("kickoff"))
        if kickoff is None:
            invalid_rows += 1
            continue
        if kickoff <= reference:
            continue
        item = dict(row)
        item["_committed_units"] = units
        open_rows.append(item)

    integrity_ok = invalid_rows == 0
    return open_rows, {
        "ledger_available": True,
        "integrity_ok": integrity_ok,
        "open_bets": len(open_rows),
        "open_units": round(
            sum(float(row["_committed_units"]) for row in open_rows),
            6,
        ),
        "invalid_rows": invalid_rows,
        "target_periods": [list(value) for value in sorted(target_periods)],
        "reason": (
            "committed production exposure loaded"
            if integrity_ok
            else "committed-exposure ledger contains invalid open BET rows"
        ),
    }


def apply_portfolio_controls(
    candidates: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
    release_gate: dict[str, object] | None = None,
    live_bets_path: str | Path = "reports/live_graded_bets.csv",
    decision_ledger_path: str | Path = "history/portfolio_decisions_v1.csv",
    now: datetime | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Allocate cap-constrained paper/shadow/production units to NFL opportunities."""

    active = policy or load_policy()
    limits = active.get("portfolio")
    if not isinstance(limits, dict):
        limits = {}
    gate = release_gate or _read_json(
        "outputs/release_gate.json",
        {"release_state": "PAPER", "production_eligible": False},
    )
    gate_state = str(gate.get("release_state", "PAPER")).upper()
    policy_mode = str(active.get("deployment_mode", "paper")).lower()
    bankroll = build_bankroll_risk_state(live_bets_path, limits)
    feedback = build_performance_feedback(live_bets_path, limits)
    require_history = bool(limits.get("require_live_history_for_production", True))
    history_ok = bool(bankroll["history_available"]) or not require_history
    candidate_rows = candidates.to_dicts() if not candidates.is_empty() else []
    committed_rows, committed_exposure = load_committed_production_exposure(
        decision_ledger_path,
        candidate_rows,
        now=now,
    )
    require_exposure_ledger = bool(
        limits.get("require_open_exposure_ledger_for_production", True)
    )
    exposure_ok = (
        bool(committed_exposure["ledger_available"])
        and bool(committed_exposure["integrity_ok"])
    ) or not require_exposure_ledger
    production_gate_open = (
        bool(gate.get("production_eligible"))
        and policy_mode == "production"
    )
    production_allowed = (
        production_gate_open
        and history_ok
        and exposure_ok
        and not bool(bankroll["hard_stop"])
    )
    if production_gate_open and not history_ok:
        production_block_reason = (
            "production release gate is open but the independent live betting "
            "ledger is unavailable"
        )
    elif production_gate_open and not exposure_ok:
        production_block_reason = str(
            committed_exposure.get("reason")
            or "committed production exposure cannot be verified"
        )
    elif production_gate_open and bool(bankroll["hard_stop"]):
        production_block_reason = str(
            bankroll.get("reason") or "bankroll hard stop is active"
        )
    else:
        production_block_reason = ""

    if production_gate_open and not production_allowed:
        effective_mode = "halted"
    elif production_allowed:
        effective_mode = "production"
    elif gate_state == "SHADOW":
        effective_mode = "shadow"
    else:
        effective_mode = "paper"

    if candidates.is_empty():
        return candidates, {
            "mode": effective_mode,
            "policy_mode": policy_mode,
            "release_state": gate_state,
            "production_gate_open": production_gate_open,
            "production_eligible": production_allowed,
            "production_block_reason": production_block_reason,
            "approved_bets": 0,
            "approved_units": 0.0,
            "bankroll_risk": bankroll,
            "committed_exposure": committed_exposure,
        }

    rows = candidate_rows
    for row in rows:
        row_feedback = performance_feedback_for_row(row, feedback)
        row["performance_multiplier"] = float(row_feedback["multiplier"])
        row["performance_feedback_reason"] = str(row_feedback["reason"])
    rows.sort(
        key=lambda row: (
            _number(row.get("quant_ev")) * _number(row.get("performance_multiplier"), 1.0),
            _number(row.get("quant_edge")),
            _number(row.get("quant_probability")),
        ),
        reverse=True,
    )

    multiplier = float(bankroll["risk_multiplier"])
    caps = {
        "slate": _number(limits.get("max_slate_units"), 5.0) * multiplier,
        "game": _number(limits.get("max_game_units"), 1.0) * multiplier,
        "team": _number(limits.get("max_team_units"), 1.5) * multiplier,
        "market": _number(limits.get("max_market_units"), 2.5) * multiplier,
        "window": _number(limits.get("max_kickoff_window_units"), 2.0) * multiplier,
        "book": _number(limits.get("max_book_units"), 2.0) * multiplier,
    }
    min_allocation = max(0.0, _number(limits.get("min_allocation_units"), 0.05))
    max_bets = max(1, int(_number(limits.get("max_bets"), 20)))
    bucket_hours = max(1, int(_number(limits.get("kickoff_window_hours"), 3)))

    slate = 0.0
    by_game: dict[str, float] = {}
    by_team: dict[str, float] = {}
    by_market: dict[str, float] = {}
    by_book: dict[str, float] = {}
    by_window: dict[str, float] = {}
    committed_keys: set[tuple[str, str]] = set()
    active_committed = committed_rows if production_gate_open else []
    for committed in active_committed:
        units = _number(committed.get("_committed_units"))
        game = str(committed.get("game_id") or "unknown")
        market = str(committed.get("quant_market") or "unknown").lower()
        book = str(committed.get("quant_book") or "unattributed")
        window = _kickoff_bucket(
            committed.get("kickoff", committed.get("date")),
            bucket_hours,
        )
        slate += units
        by_game[game] = by_game.get(game, 0.0) + units
        by_market[market] = by_market.get(market, 0.0) + units
        by_book[book] = by_book.get(book, 0.0) + units
        by_window[window] = by_window.get(window, 0.0) + units
        for team in _team_keys(committed):
            by_team[team] = by_team.get(team, 0.0) + units
        committed_keys.add((game, market))

    committed_bets = len(active_committed)
    committed_units = slate
    allocated = 0
    blocked = 0

    for row in rows:
        research_stake = max(
            0.0,
            _number(row.get("research_stake_units", row.get("stake_units"))),
        )
        production_stake = max(0.0, _number(row.get("stake_units")))
        if production_gate_open:
            signal = str(row.get("quant_signal") or "PASS").upper()
            proposed_base = production_stake
        else:
            signal = str(
                row.get("research_signal", row.get("quant_signal")) or "PASS"
            ).upper()
            proposed_base = research_stake
        bankroll_adjusted = proposed_base * multiplier
        performance_multiplier = max(0.0, min(1.0, _number(row.get("performance_multiplier"), 1.0)))
        proposed = bankroll_adjusted * performance_multiplier
        row["portfolio_signal"] = signal
        row["paper_stake_units"] = research_stake
        row["bankroll_adjusted_units"] = round(bankroll_adjusted, 6)
        row["performance_adjusted_units"] = round(proposed, 6)
        row["portfolio_candidate_units"] = 0.0
        row["portfolio_stake_units"] = 0.0
        row["portfolio_action"] = "PASS"
        row["portfolio_limit_reason"] = ""

        freshness = recommendation_freshness(row, limits=limits, now=now)
        row.update(freshness)
        executable, reason = validate_execution_row(row, limits=limits, now=now)
        row["execution_ready"] = executable and freshness["recommendation_status"] == "ACTIVE"
        if not row["execution_ready"]:
            blocked += 1
            row["portfolio_limit_reason"] = (
                reason
                if not executable
                else str(freshness["recommendation_freshness_reason"])
            )
            # A non-executable quote must not consume portfolio capacity in any
            # mode. PAPER/SHADOW retain the research signal for diagnostics, but
            # portfolio allocation is reserved for timestamp-valid pre-kickoff
            # opportunities so the decision ledger remains gradeable.
            continue

        if production_gate_open and not production_allowed:
            row["portfolio_limit_reason"] = (
                production_block_reason or "production safety halt is active"
            )
            continue

        if (
            signal == "PASS"
            or proposed <= 0
            or committed_bets + allocated >= max_bets
            or multiplier <= 0
        ):
            if committed_bets + allocated >= max_bets and signal != "PASS":
                row["portfolio_limit_reason"] = "max bet count"
            continue

        game = str(row.get("game_id") or "unknown")
        market = str(row.get("quant_market") or "unknown").lower()
        book = str(row.get("quant_book") or "unattributed")
        window = _kickoff_bucket(
            row.get("kickoff", row.get("gameday")),
            bucket_hours,
        )
        teams = _team_keys(row)

        if production_allowed and (game, market) in committed_keys:
            row["portfolio_limit_reason"] = (
                "production stake already committed for game/market"
            )
            continue

        residuals = [
            caps["slate"] - slate,
            caps["game"] - by_game.get(game, 0.0),
            caps["market"] - by_market.get(market, 0.0),
            caps["book"] - by_book.get(book, 0.0),
            caps["window"] - by_window.get(window, 0.0),
        ]
        residuals.extend(caps["team"] - by_team.get(team, 0.0) for team in teams)
        allocation = max(0.0, min([proposed, *residuals]))
        if allocation < min_allocation:
            row["portfolio_limit_reason"] = (
                row["portfolio_limit_reason"]
                or "portfolio cap below minimum allocation"
            )
            continue

        row["portfolio_candidate_units"] = round(allocation, 6)
        if production_allowed and executable:
            row["portfolio_action"] = "BET"
            row["portfolio_stake_units"] = round(allocation, 6)
        elif gate_state == "SHADOW":
            row["portfolio_action"] = "SHADOW"
        else:
            row["portfolio_action"] = "PAPER"

        slate += allocation
        by_game[game] = by_game.get(game, 0.0) + allocation
        by_market[market] = by_market.get(market, 0.0) + allocation
        by_book[book] = by_book.get(book, 0.0) + allocation
        by_window[window] = by_window.get(window, 0.0) + allocation
        for team in teams:
            by_team[team] = by_team.get(team, 0.0) + allocation
        allocated += 1

    output = pl.DataFrame(rows)
    return output, {
        "mode": effective_mode,
        "policy_mode": policy_mode,
        "release_state": gate_state,
        "production_gate_open": production_gate_open,
        "production_eligible": production_allowed,
        "production_block_reason": production_block_reason,
        "bankroll_risk": bankroll,
        "performance_feedback": feedback,
        "committed_exposure": {
            **committed_exposure,
            "reserved_open_bets": committed_bets,
            "reserved_open_units": round(committed_units, 6),
            "total_open_bets_after_allocation": committed_bets + allocated,
            "total_open_units_after_allocation": round(slate, 6),
        },
        "proposed_units": round(
            sum(_number(row.get("paper_stake_units")) for row in rows),
            6,
        ),
        "risk_adjusted_proposed_units": round(
            sum(_number(row.get("bankroll_adjusted_units")) for row in rows), 6
        ),
        "performance_adjusted_proposed_units": round(
            sum(_number(row.get("performance_adjusted_units")) for row in rows), 6
        ),
        "approved_units": round(
            sum(_number(row.get("portfolio_stake_units")) for row in rows), 6
        ),
        "paper_or_shadow_allocated_units": round(
            sum(_number(row.get("portfolio_candidate_units")) for row in rows), 6
        ),
        "bets": allocated,
        "approved_bets": sum(
            1 for row in rows if row.get("portfolio_action") == "BET"
        ),
        "execution_blocked_bets": blocked,
        "active_recommendations": sum(
            str(row.get("recommendation_status") or "").upper() == "ACTIVE"
            for row in rows
        ),
        "expired_recommendations": sum(
            str(row.get("recommendation_status") or "").upper() == "EXPIRED"
            for row in rows
        ),
    }
