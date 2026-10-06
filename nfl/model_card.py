"""Machine-readable NFL model card, structurally aligned with CFB reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .research_status import build_research_status


def _read_optional_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_model_card(
    meta: dict[str, object],
    gate: dict[str, object],
    evidence: dict[str, object] | None = None,
    research_status: dict[str, object] | None = None,
) -> dict[str, object]:
    evidence = evidence or {}
    production_policy = _read_optional_json("reports/production_policy.json")
    research = research_status
    if research is None:
        research = _read_optional_json("reports/research_status.json")
    if not research:
        research = build_research_status(
            candidate_benchmark=_read_optional_json(
                "reports/candidate_benchmark.json"
            ),
            market_shrinkage=_read_optional_json(
                "reports/market_edge_shrinkage.json"
            ),
            forward_shadow=_read_optional_json(
                "reports/forward_shadow_summary.json"
            ),
        )
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
            "quarterback_layer": (
                "historical QB shadow subsystem plus current expected-starter "
                "identity/certainty gate"
            ),
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
            "expected_starting_qb_required": True,
            "uncertain_or_changed_qb_blocks_betting": True,
            "production_policy_mode": production_policy.get(
                "deployment_mode", "paper"
            ),
            "production_policy_ready": bool(
                gate.get("production_policy_ready", False)
            ),
        },
        "evidence": {
            "historical_status": evidence.get("status", "UNKNOWN"),
            "historical": evidence.get("overall", {}),
            "promotion_sample": evidence.get("promotion_sample", {}),
        },
        "research_status": {
            "status": research.get("status", "MISSING"),
            "research_decision": research.get("research_decision"),
            "active_forward_candidates": research.get(
                "active_forward_candidates",
                [],
            ),
            "promotion_ready_forward_candidates": research.get(
                "promotion_ready_forward_candidates",
                [],
            ),
            "canonical_model_change_enabled": bool(
                research.get("canonical_model_change_enabled", False)
            ),
            "canonical_market_change_enabled": bool(
                research.get("canonical_market_change_enabled", False)
            ),
        },
        "non_negotiables": [
            "No target-week/future leakage.",
            "No sportsbook line in the independent fair-score engine.",
            "NFL parameters are validated independently from CFB.",
            "Missing quotes/context are not invented.",
            (
                "Current betting requires an identified, decision-ready expected "
                "starting QB for both teams."
            ),
            (
                "A starter change relative to the last-observed QB remains "
                "betting-blocked until a validated replacement-QB adjustment exists."
            ),
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
