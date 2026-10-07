"""Append-only audit ledger for blocked NFL model candidates and quote outliers."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import polars as pl

FIELDS = (
    "game_id",
    "season",
    "week",
    "date",
    "kickoff",
    "away_team",
    "home_team",
    "model_margin_home",
    "model_total",
    "quant_market",
    "quant_side",
    "quant_book",
    "quant_price",
    "quant_odds",
    "quant_quote_at",
    "quant_probability",
    "quant_ev",
    "quant_edge",
    "model_candidate_signal",
    "model_candidate_stake_units",
    "quant_signal",
    "research_signal",
    "candidate_state",
    "candidate_watch",
    "candidate_block_reason",
    "selected_quote_consensus_value",
    "market_consensus_value",
    "selected_quote_consensus_distance",
    "selected_quote_outlier_threshold",
    "selected_quote_outlier",
    "market_disagreement",
    "market_disagreement_severity",
    "market_dispersion",
    "market_dispersion_high",
    "probability_reliability_veto",
    "qb_certainty_veto",
    "qb_certainty_veto_reason",
    "context_freshness_veto",
    "context_freshness_veto_reason",
    "context_veto",
    "context_veto_reason",
    "regime_reliability_ready",
    "regime_reliability_status",
    "execution_action",
    "execution_action_reason",
)


def _normal(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 8) if isfinite(value) else None
    return value


def _signature(row: dict[str, object]) -> str:
    payload = {field: _normal(row.get(field)) for field in FIELDS}
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def append_candidate_observations(
    candidates: pl.DataFrame,
    *,
    path: str | Path = "history/candidate_observations_v1.csv",
    observed_at: str | None = None,
) -> dict[str, object]:
    """Persist changing blocked-candidate/outlier states without grading them as bets."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if candidates.is_empty():
        return {"path": str(target), "eligible_rows": 0, "appended_rows": 0}

    eligible = [
        row
        for row in candidates.to_dicts()
        if str(row.get("model_candidate_signal") or "PASS").upper() != "PASS"
        or bool(row.get("selected_quote_outlier", False))
    ]
    if not eligible:
        return {"path": str(target), "eligible_rows": 0, "appended_rows": 0}

    old: list[dict[str, object]] = []
    if target.exists() and target.stat().st_size:
        try:
            with target.open(newline="") as handle:
                old = list(csv.DictReader(handle))
        except OSError:
            old = []

    latest: dict[tuple[str, str], str] = {}
    for row in old:
        key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""))
        latest[key] = str(row.get("candidate_signature") or "")

    stamp = observed_at or datetime.now(UTC).isoformat()
    additions: list[dict[str, object]] = []
    for row in eligible:
        signature = _signature(row)
        key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""))
        if latest.get(key) == signature:
            continue
        record: dict[str, object] = {"observed_at": stamp}
        for field in FIELDS:
            record[field] = row.get(field)
        record["candidate_signature"] = signature
        additions.append(record)
        latest[key] = signature

    fieldnames = ["observed_at", *FIELDS, "candidate_signature"]
    combined = [*old, *additions]
    if additions or not target.exists():
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined)

    return {
        "path": str(target),
        "eligible_rows": len(eligible),
        "appended_rows": len(additions),
        "total_rows": len(combined),
    }
