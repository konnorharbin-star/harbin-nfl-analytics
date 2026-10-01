"""Machine-readable NFL model card, structurally aligned with CFB reporting."""

from __future__ import annotations

import json
from pathlib import Path


def build_model_card(
    meta: dict[str, object],
    gate: dict[str, object],
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = evidence or {}
    return {
        "name": "Harbin NFL Analytics",
        "league": "NFL",
        "release_state": gate.get("release_state", "RESEARCH"),
        "production_eligible": bool(gate.get("production_eligible", False)),
        "purpose": (
            "Leakage-safe NFL fair-score, probability, market-research, portfolio-risk, "
            "grading, and monitoring platform."
        ),
        "core_model": {
            "fair_score": "ridge-regularized team offense/defense scoring model",
            "sportsbook_prices_in_score_model": False,
            "prior_season_weight": 0.10,
            "quarterback_layer": "separate NFL-specific research/shadow subsystem",
        },
        "probability": {
            "method": "chronologically fitted Gaussian score residual distribution",
            "sportsbook_lines_as_inputs": False,
            "holdout": meta.get("probability", {}),
        },
        "markets": {
            "historical_primary": "free nflverse archive",
            "current_primary": "ESPN public endpoints",
            "optional_enrichment": "The Odds API or other verified multi-book source",
            "clv_policy": "proxy labels remain explicit unless timestamped forward snapshots exist",
        },
        "risk": {
            "staking": "capped fractional Kelly",
            "portfolio_caps": True,
            "drawdown_throttle": True,
            "production_requires_release_gate": True,
        },
        "evidence": {
            "historical_status": evidence.get("status", "UNKNOWN"),
            "historical": evidence.get("overall", {}),
            "promotion_sample": evidence.get("promotion_sample", {}),
        },
        "non_negotiables": [
            "No target-week/future leakage.",
            "No sportsbook line in the independent fair-score engine.",
            "NFL parameters are validated independently from CFB.",
            "Missing quotes/context are not invented.",
            "Historical archive fallbacks do not become verified opening entries by relabeling.",
            "No production staking until every hard release gate passes.",
        ],
    }


def write_model_card(
    meta: dict[str, object],
    gate: dict[str, object],
    evidence: dict[str, object] | None = None,
    *,
    output: str | Path = "outputs/model_card.json",
    report: str | Path = "reports/model_card.json",
) -> dict[str, object]:
    card = build_model_card(meta, gate, evidence)
    for path in (output, report):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(card, indent=2, sort_keys=True, default=str))
    return card
