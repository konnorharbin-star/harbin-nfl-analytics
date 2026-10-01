"""NFL source/model health summary aligned with the CFB operational health layer."""

from __future__ import annotations

import json
from pathlib import Path


def build_health(
    meta: dict[str, object],
    monitor: dict[str, object],
    gate: dict[str, object],
) -> dict[str, object]:
    data_quality = meta.get("data_quality")
    if not isinstance(data_quality, dict):
        data_quality = {}
    sources = meta.get("sources")
    if not isinstance(sources, dict):
        sources = {}

    source_failures = [name for name, value in sources.items() if str(value).upper() in {"FAIL", "ERROR"}]
    data_ok = str(data_quality.get("status", "UNKNOWN")).upper() not in {"FAIL", "ERROR"}
    readiness = float(monitor.get("live_readiness_score", 0.0) or 0.0)
    status = "FAIL" if source_failures or not data_ok else "WARN" if readiness < 80 else "OK"
    return {
        "status": status,
        "release_state": gate.get("release_state", "RESEARCH"),
        "production_eligible": bool(gate.get("production_eligible", False)),
        "live_readiness_score": readiness,
        "data_quality": data_quality,
        "sources": sources,
        "source_failures": source_failures,
        "monitor_alerts": monitor.get("alerts", []),
        "release_blockers": gate.get("blockers", []),
        "meaning": "operational health; not a profitability or staking signal",
    }


def write_health(
    meta: dict[str, object],
    monitor: dict[str, object],
    gate: dict[str, object],
    *,
    output: str | Path = "outputs/health.json",
) -> dict[str, object]:
    report = build_health(meta, monitor, gate)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
