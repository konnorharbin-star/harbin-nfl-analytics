"""Append-only forward NFL portfolio-decision ledger."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

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
    decisions,
    path="history/portfolio_decisions_v1.csv",
    decision_at: str | None = None,
) -> dict:
    """Append research decisions without rewriting any historical bytes.

    This is the legacy research ledger, not a receipt proving a user recommendation.
    Lock concurrent writers and deduplicate all prior signatures per game/market.
    Fail closed on unreadable or incompatible history rather than replacing it.
    """
    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    records = decisions.to_dicts()
    rows = [
        r
        for r in records
        if float(r.get("portfolio_candidate_units") or 0) > 0
        and str(r.get("portfolio_action") or "PASS").upper() in {"PAPER", "SHADOW", "BET"}
    ]
    if not rows:
        return {"path": str(source), "eligible_rows": 0, "appended_rows": 0}
    stamp = decision_at or datetime.now(UTC).isoformat()
    expected = ["decision_at", *LEDGER_FIELDS, "decision_signature"]
    with source.open("a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or expected
        if reader.fieldnames and not {
            "decision_at",
            "game_id",
            "quant_market",
            "decision_signature",
        }.issubset(fields):
            raise ValueError("Incompatible decision ledger schema; history preserved")
        old = list(reader)
        seen = {
            (
                str(r.get("game_id") or ""),
                str(r.get("quant_market") or ""),
                str(r.get("decision_signature") or ""),
            )
            for r in old
        }
        additions = []
        for row in rows:
            signature = _signature(row)
            key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""), signature)
            if not key[0] or not key[1]:
                raise ValueError("Decision requires game and market identity")
            if key in seen:
                continue
            additions.append(
                {
                    "decision_at": stamp,
                    **{k: row.get(k) for k in LEDGER_FIELDS},
                    "decision_signature": signature,
                }
            )
            seen.add(key)
        # Legacy headers stay byte-for-byte intact. Freeze the complete modern
        # row in a sidecar so new context fields are never silently discarded.
        if set(LEDGER_FIELDS) - set(fields):
            events = source.with_suffix(".events")
            events.mkdir(exist_ok=True)
            for record in additions:
                identity = {
                    k: record.get(k) for k in ("game_id", "quant_market", "decision_signature")
                }
                event_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                event_path = events / (event_id + ".json")
                if not event_path.exists():
                    with event_path.open("x") as event:
                        json.dump(record, event, sort_keys=True, default=str)
                        event.flush()
                        os.fsync(event.fileno())
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(additions)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(source),
        "eligible_rows": len(rows),
        "appended_rows": len(additions),
        "total_rows": len(old) + len(additions),
    }
