"""Audit NFL portfolio caps, execution provenance, and production-stake gating."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .contracts import require_columns
from .policy import load_policy


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _teams(row: dict[str, object]) -> list[str]:
    market = str(row.get("quant_market") or "").lower()
    side = str(row.get("quant_side") or "").lower()
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


def _window(value: object, hours: int) -> str:
    stamp = _parse(value)
    if stamp is None:
        return "unknown"
    seconds = max(1, hours) * 3600
    epoch = int(stamp.timestamp())
    return str(epoch - epoch % seconds)


def audit_portfolio(
    frame: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
    release_gate: dict[str, object] | None = None,
    bankroll_multiplier: float = 1.0,
) -> dict[str, object]:
    """Verify cap-constrained allocations and fail-closed production execution."""

    active = policy or load_policy()
    limits = active.get("portfolio")
    if not isinstance(limits, dict):
        limits = {}
    gate = release_gate or {"production_eligible": False, "release_state": "PAPER"}
    if frame.is_empty():
        return {
            "status": "WARN",
            "errors": 0,
            "warnings": 1,
            "issues": [{"severity": "WARNING", "code": "empty_portfolio"}],
            "candidate_units": 0.0,
            "real_stake_units": 0.0,
        }

    require_columns(
        frame,
        {
            "game_id",
            "home_team",
            "away_team",
            "quant_market",
            "quant_side",
            "quant_book",
            "quant_odds",
            "quant_quote_at",
            "market_book_count",
            "kickoff",
            "portfolio_candidate_units",
            "portfolio_stake_units",
            "portfolio_action",
            "execution_ready",
        },
        "portfolio_output",
    )

    multiplier = max(0.0, min(1.0, float(bankroll_multiplier)))
    caps = {
        "slate": _number(limits.get("max_slate_units"), 5.0) * multiplier,
        "game": _number(limits.get("max_game_units"), 1.0) * multiplier,
        "team": _number(limits.get("max_team_units"), 1.5) * multiplier,
        "market": _number(limits.get("max_market_units"), 2.5) * multiplier,
        "book": _number(limits.get("max_book_units"), 2.0) * multiplier,
        "window": _number(limits.get("max_kickoff_window_units"), 2.0) * multiplier,
    }
    max_bets = max(1, int(_number(limits.get("max_bets"), 20)))
    minimum_books = max(
        1,
        int(_number(limits.get("min_market_book_count_for_execution"), 1)),
    )
    bucket_hours = max(1, int(_number(limits.get("kickoff_window_hours"), 3)))

    issues: list[dict[str, object]] = []
    candidate_total = 0.0
    real_total = 0.0
    by_game: dict[str, float] = {}
    by_team: dict[str, float] = {}
    by_market: dict[str, float] = {}
    by_book: dict[str, float] = {}
    by_window: dict[str, float] = {}
    allocated_rows = 0

    for row in frame.iter_rows(named=True):
        candidate = _number(row.get("portfolio_candidate_units"))
        real = _number(row.get("portfolio_stake_units"))
        if candidate < -1e-9 or real < -1e-9:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "negative_portfolio_units",
                    "game_id": row["game_id"],
                }
            )
            continue
        if real > candidate + 1e-9:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "real_stake_exceeds_candidate_allocation",
                    "game_id": row["game_id"],
                }
            )
        if real > 0:
            if not bool(gate.get("production_eligible", False)):
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "real_stake_without_production_gate",
                        "game_id": row["game_id"],
                    }
                )
            if str(row.get("portfolio_action") or "").upper() != "BET":
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "real_stake_without_bet_action",
                        "game_id": row["game_id"],
                    }
                )
            if not bool(row.get("execution_ready")):
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": "real_stake_without_execution_ready",
                        "game_id": row["game_id"],
                    }
                )
        if candidate <= 0:
            continue

        allocated_rows += 1
        candidate_total += candidate
        real_total += real
        game = str(row["game_id"])
        market = str(row["quant_market"])
        book = str(row["quant_book"])
        window = _window(row.get("kickoff"), bucket_hours)
        by_game[game] = by_game.get(game, 0.0) + candidate
        by_market[market] = by_market.get(market, 0.0) + candidate
        by_book[book] = by_book.get(book, 0.0) + candidate
        by_window[window] = by_window.get(window, 0.0) + candidate
        for team in _teams(row):
            by_team[team] = by_team.get(team, 0.0) + candidate

        quote = _parse(row.get("quant_quote_at"))
        kickoff = _parse(row.get("kickoff"))
        if quote is None or kickoff is None or quote >= kickoff:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "allocated_quote_not_pre_kickoff",
                    "game_id": game,
                }
            )
        books = _number(row.get("market_book_count"), -1.0)
        if books < minimum_books:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "allocated_market_book_count_below_minimum",
                    "game_id": game,
                    "book_count": books,
                    "minimum": minimum_books,
                }
            )

    def cap_issues(values: dict[str, float], cap: float, code: str) -> None:
        for key, value in values.items():
            if value > cap + 1e-9:
                issues.append(
                    {
                        "severity": "ERROR",
                        "code": code,
                        "key": key,
                        "units": value,
                        "cap": cap,
                    }
                )

    if candidate_total > caps["slate"] + 1e-9:
        issues.append(
            {
                "severity": "ERROR",
                "code": "slate_cap_exceeded",
                "units": candidate_total,
                "cap": caps["slate"],
            }
        )
    if allocated_rows > max_bets:
        issues.append(
            {
                "severity": "ERROR",
                "code": "max_bet_count_exceeded",
                "bets": allocated_rows,
                "cap": max_bets,
            }
        )
    cap_issues(by_game, caps["game"], "game_cap_exceeded")
    cap_issues(by_team, caps["team"], "team_cap_exceeded")
    cap_issues(by_market, caps["market"], "market_cap_exceeded")
    cap_issues(by_book, caps["book"], "book_cap_exceeded")
    cap_issues(by_window, caps["window"], "kickoff_window_cap_exceeded")

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
        "candidate_units": candidate_total,
        "real_stake_units": real_total,
        "allocated_bets": allocated_rows,
        "caps": caps,
        "release_state": gate.get("release_state", "PAPER"),
        "production_eligible": bool(gate.get("production_eligible", False)),
    }


def write_portfolio_audit(
    frame: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
    release_gate: dict[str, object] | None = None,
    bankroll_multiplier: float = 1.0,
    output: str | Path = "reports/portfolio_audit.json",
    fail_on_error: bool = True,
) -> dict[str, object]:
    report = audit_portfolio(
        frame,
        policy=policy,
        release_gate=release_gate,
        bankroll_multiplier=bankroll_multiplier,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if fail_on_error and report["errors"]:
        raise RuntimeError("portfolio audit failed")
    return report
