"""Canonical NFL reporting outputs aligned with the CFB publication layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .render import write_weekly_publication


def build_canonical_report(
    current: pl.DataFrame,
    *,
    meta: dict[str, object],
    portfolio: dict[str, object],
    monitoring: dict[str, object],
    release_gate: dict[str, object],
    health: dict[str, object],
    evidence: dict[str, object],
    model_card: dict[str, object],
    ledger: dict[str, object] | None = None,
    line_capture: dict[str, object] | None = None,
    publication: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "league": "NFL",
        "release_state": release_gate.get("release_state", "RESEARCH"),
        "production_eligible": bool(release_gate.get("production_eligible", False)),
        "meta": meta,
        "portfolio": portfolio,
        "monitoring": monitoring,
        "release_gate": release_gate,
        "health": health,
        "evidence": evidence,
        "model_card": model_card,
        "decision_ledger": ledger or {},
        "line_capture": line_capture or {},
        "publication": publication or {},
        "games": current.to_dicts(),
    }


def write_canonical_report(
    current: pl.DataFrame,
    *,
    meta: dict[str, object],
    portfolio: dict[str, object],
    monitoring: dict[str, object],
    release_gate: dict[str, object],
    health: dict[str, object],
    evidence: dict[str, object],
    model_card: dict[str, object],
    ledger: dict[str, object] | None = None,
    line_capture: dict[str, object] | None = None,
    output_json: str | Path = "outputs/current_model.json",
    report_json: str | Path = "reports/current_model.json",
    output_csv: str | Path = "outputs/current_predictions.csv",
) -> dict[str, object]:
    release_state = str(release_gate.get("release_state", "RESEARCH"))
    publication = write_weekly_publication(
        current,
        week=int(meta.get("week", 0) or 0),
        updated_at=str(meta.get("generated_at") or datetime.now(UTC).isoformat()),
        release_state=release_state,
    )
    payload = build_canonical_report(
        current,
        meta=meta,
        portfolio=portfolio,
        monitoring=monitoring,
        release_gate=release_gate,
        health=health,
        evidence=evidence,
        model_card=model_card,
        ledger=ledger,
        line_capture=line_capture,
        publication=publication,
    )
    for path in (output_json, report_json):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    csv_target = Path(output_csv)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    if not current.is_empty():
        current.write_csv(csv_target)
    return payload
