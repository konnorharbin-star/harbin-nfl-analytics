"""Canonical NFL reporting outputs aligned with the CFB publication layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .publication import validate_publication_files, write_publication_bundle
from .publication_integrity import write_publication_manifest
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


def _write_json_copies(
    payload: dict[str, object],
    *,
    output_json: str | Path,
    report_json: str | Path,
) -> None:
    for path in (output_json, report_json):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


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
    _write_json_copies(payload, output_json=output_json, report_json=report_json)

    csv_target = Path(output_csv)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    if current.columns:
        # Even a zero-game slate must replace any old predictions with a
        # header-only current file, not keep last week's apparent picks.
        current.write_csv(csv_target)
    else:
        csv_target.write_text(
            "season,week,game_id,home_team,away_team,quant_market,"
            "quant_side,quant_signal\n",
            encoding="utf-8",
        )

    bundle = write_publication_bundle(current, payload)
    publication["bundle"] = bundle
    payload["publication"] = publication
    _write_json_copies(payload, output_json=output_json, report_json=report_json)

    docs_latest = Path("docs/latest.json")
    docs_latest.parent.mkdir(parents=True, exist_ok=True)
    docs_latest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    # write_publication_bundle() validated a provisional JSON without its final
    # publication.bundle metadata. Re-fingerprint the EXACT final bytes now,
    # after both canonical and public JSON copies are atomically reconciled.
    write_publication_manifest()
    validation = validate_publication_files()
    for path in ("outputs/publication_validation.json", "docs/publication_validation.json"):
        Path(path).write_text(json.dumps(validation, indent=2, sort_keys=True, default=str))
    if validation["status"] == "FAIL":
        details = "; ".join(str(item["detail"]) for item in validation["errors"])
        raise RuntimeError(f"NFL publication reconciliation failed after final report: {details}")
    return payload
