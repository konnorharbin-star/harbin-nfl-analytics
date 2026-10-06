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
    "research_signal",
    "portfolio_signal",
    "portfolio_candidate_units",
    "portfolio_stake_units",
    "performance_multiplier",
    "performance_feedback_reason",
    "home_expected_qb_id",
    "away_expected_qb_id",
    "home_expected_qb_confidence",
    "away_expected_qb_confidence",
    "home_expected_qb_decision_ready",
    "away_expected_qb_decision_ready",
    "qb_certainty_veto",
    "context_injuries_personnel_fresh",
    "context_freshness_veto",
    "probability_model_family",
    "probability_reliability_ready",
    "probability_reliability_veto",
    "regime_reliability_ready",
    "regime_reliability_status",
    "regime_reliability_blocked_segments",
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
    "quote_age_minutes",
    "recommendation_status",
    "recommendation_valid_until",
    "recommendation_minutes_remaining",
    "recommendation_freshness_reason",
    "quant_probability",
    "quant_ev",
    "quant_edge",
    "market_book_count",
    "market_consensus_quality",
    "data_quality_score",
    "context_risk",
    "risk_multiplier",
    "market_disagreement",
    "home_expected_qb_id",
    "home_expected_qb_name",
    "home_expected_qb_source",
    "home_expected_qb_confidence",
    "home_expected_qb_confidence_label",
    "home_expected_qb_injury_status",
    "home_expected_qb_injury_severity",
    "home_expected_qb_changed_from_last_observed",
    "home_expected_qb_decision_ready",
    "away_expected_qb_id",
    "away_expected_qb_name",
    "away_expected_qb_source",
    "away_expected_qb_confidence",
    "away_expected_qb_confidence_label",
    "away_expected_qb_injury_status",
    "away_expected_qb_injury_severity",
    "away_expected_qb_changed_from_last_observed",
    "away_expected_qb_decision_ready",
    "qb_certainty_veto",
    "qb_certainty_veto_reason",
    "context_injuries_personnel_fresh",
    "context_freshness_reason",
    "context_freshness_veto",
    "context_freshness_veto_reason",
    "probability_model_family",
    "probability_reliability_ready",
    "probability_reliability_veto",
    "probability_reliability_veto_reason",
    "regime_reliability_ready",
    "regime_reliability_status",
    "regime_reliability_reason",
    "regime_reliability_blocked_segments",
    "regime_reliability_missing_segments",
    "model_margin_sigma",
    "model_total_sigma",
    "injury_feed_freshness_status",
    "injury_feed_age_days",
    "home_injury_freshness_status",
    "away_injury_freshness_status",
    "home_depth_freshness_status",
    "away_depth_freshness_status",
    "home_depth_age_days",
    "away_depth_age_days",
    "home_roster_freshness_status",
    "away_roster_freshness_status",
    "home_roster_week_gap",
    "away_roster_week_gap",
    "home_personnel_freshness_status",
    "away_personnel_freshness_status",
    "performance_multiplier",
    "performance_feedback_reason",
    "performance_adjusted_units",
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
