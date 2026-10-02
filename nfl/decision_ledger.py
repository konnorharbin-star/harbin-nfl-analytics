"""Append-only forward NFL portfolio-decision ledger."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import polars as pl

SIGNATURE_FIELDS = (
    "quant_market",
    "quant_side",
    "quant_book",
    "quant_price",
    "quant_odds",
    "quant_quote_at",
    "portfolio_candidate_units",
    "portfolio_stake_units",
    "execution_ready",
    "portfolio_action",
)

LEDGER_FIELDS = (
    "game_id",
    "season",
    "week",
    "date",
    "kickoff",
    "away_team",
    "home_team",
    "model_margin_home",
    "model_total",
    "calibrated_home_probability",
    "quant_signal",
    "research_signal",
    "portfolio_signal",
    "quant_market",
    "quant_side",
    "quant_book",
    "quant_price",
    "quant_odds",
    "quant_quote_at",
    "quant_probability",
    "quant_ev",
    "quant_edge",
    "market_book_count",
    "paper_stake_units",
    "bankroll_adjusted_units",
    "portfolio_candidate_units",
    "portfolio_stake_units",
    "execution_ready",
    "portfolio_action",
    "portfolio_limit_reason",
)


def _normal(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 8) if isfinite(value) else None
    return value


def _signature(row: dict[str, object]) -> str:
    payload = {field: _normal(row.get(field)) for field in SIGNATURE_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def append_portfolio_decisions(
    decisions: pl.DataFrame,
    *,
    path: str | Path = "history/portfolio_decisions_v1.csv",
    decision_at: str | None = None,
) -> dict[str, object]:
    """Persist cap-constrained PAPER/SHADOW/BET decisions for independent grading."""

    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    if decisions.is_empty():
        return {"path": str(source), "eligible_rows": 0, "appended_rows": 0}

    rows = []
    for row in decisions.to_dicts():
        units = float(row.get("portfolio_candidate_units") or 0.0)
        action = str(row.get("portfolio_action") or "PASS").upper()
        if units > 0 and action in {"PAPER", "SHADOW", "BET"}:
            rows.append(row)
    if not rows:
        return {"path": str(source), "eligible_rows": 0, "appended_rows": 0}

    old: list[dict[str, object]] = []
    if source.exists() and source.stat().st_size:
        try:
            with source.open(newline="") as handle:
                old = list(csv.DictReader(handle))
        except OSError:
            old = []

    latest: dict[tuple[str, str], str] = {}
    for row in old:
        key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""))
        latest[key] = str(row.get("decision_signature") or "")

    stamp = decision_at or datetime.now(UTC).isoformat()
    additions: list[dict[str, object]] = []
    for row in rows:
        signature = _signature(row)
        key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""))
        if latest.get(key) == signature:
            continue
        record: dict[str, object] = {"decision_at": stamp}
        for field in LEDGER_FIELDS:
            record[field] = row.get(field)
        record["decision_signature"] = signature
        additions.append(record)
        latest[key] = signature

    fieldnames = ["decision_at", *LEDGER_FIELDS, "decision_signature"]
    combined = [*old, *additions]
    if additions or not source.exists():
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined)
    return {
        "path": str(source),
        "eligible_rows": len(rows),
        "appended_rows": len(additions),
        "total_rows": len(combined),
    }
