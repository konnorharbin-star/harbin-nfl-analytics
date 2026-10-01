"""CFB-style execution, bankroll, and concentration controls for NFL candidates."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import polars as pl

from .execution_market import validate_execution_row
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


def apply_portfolio_controls(
    candidates: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
    release_gate: dict[str, object] | None = None,
    live_bets_path: str | Path = "reports/live_graded_bets.csv",
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
    require_history = bool(limits.get("require_live_history_for_production", True))
    history_ok = bool(bankroll["history_available"]) or not require_history
    production_allowed = (
        bool(gate.get("production_eligible"))
        and policy_mode == "production"
        and history_ok
        and not bool(bankroll["hard_stop"])
    )
    if production_allowed:
        effective_mode = "production"
    elif gate_state == "SHADOW":
        effective_mode = "shadow"
    else:
        effective_mode = "paper"

    if candidates.is_empty():
        return candidates, {
            "mode": effective_mode,
            "approved_bets": 0,
            "approved_units": 0.0,
            "bankroll_risk": bankroll,
        }

    rows = candidates.to_dicts()
    rows.sort(
        key=lambda row: (
            _number(row.get("quant_ev")),
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
    allocated = 0
    blocked = 0

    for row in rows:
        proposed = max(0.0, _number(row.get("stake_units"))) * multiplier
        signal = str(row.get("quant_signal") or "PASS").upper()
        row["paper_stake_units"] = max(0.0, _number(row.get("stake_units")))
        row["bankroll_adjusted_units"] = round(proposed, 6)
        row["portfolio_candidate_units"] = 0.0
        row["portfolio_stake_units"] = 0.0
        row["portfolio_action"] = "PASS"
        row["portfolio_limit_reason"] = ""

        executable, reason = validate_execution_row(row, limits=limits, now=now)
        row["execution_ready"] = executable
        if not executable:
            blocked += 1
            row["portfolio_limit_reason"] = reason

        if signal == "PASS" or proposed <= 0 or allocated >= max_bets or multiplier <= 0:
            if allocated >= max_bets and signal != "PASS":
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
        "production_eligible": production_allowed,
        "bankroll_risk": bankroll,
        "proposed_units": round(
            sum(_number(row.get("paper_stake_units")) for row in rows),
            6,
        ),
        "risk_adjusted_proposed_units": round(
            sum(_number(row.get("bankroll_adjusted_units")) for row in rows), 6
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
    }
