"""Machine-readable registry of NFL research decisions and forward evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .qb_validated import QB_MARGIN_ENABLED, QB_TOTAL_SHADOW_ENABLED

DEFAULT_PATHS = {
    "candidate_benchmark": Path("reports/candidate_benchmark.json"),
    "market_shrinkage": Path("reports/market_edge_shrinkage.json"),
    "forward_shadow": Path("reports/forward_shadow_summary.json"),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_summary(report: dict[str, Any]) -> dict[str, object]:
    margin = report.get("margin_selection")
    total = report.get("total_selection")
    if not isinstance(margin, dict):
        margin = {}
    if not isinstance(total, dict):
        total = {}
    return {
        "status": report.get("status", "MISSING"),
        "margin_architecture": margin.get("selected"),
        "margin_architecture_state": margin.get("release_state"),
        "margin_robustness_score": margin.get("robustness_score"),
        "total_architecture": total.get("selected"),
        "total_architecture_state": total.get("release_state"),
        "total_robustness_score": total.get("robustness_score"),
        "canonical_score_adjustment_enabled": bool(
            report.get("canonical_score_adjustment_enabled", False)
        ),
        "fixed_qb_margin_enabled": bool(QB_MARGIN_ENABLED),
        "fixed_qb_total_shadow_enabled": bool(QB_TOTAL_SHADOW_ENABLED),
        "interpretation": (
            "Stage 14 architecture selection is development evidence only. "
            "Stage 15 is authoritative for fixed QB specifications."
        ),
    }


def _market_summary(report: dict[str, Any]) -> dict[str, object]:
    markets = report.get("markets")
    if not isinstance(markets, dict):
        markets = {}
    alphas: dict[str, object] = {}
    incremental: dict[str, object] = {}
    for market in ("moneyline", "spread", "total"):
        row = markets.get(market)
        if not isinstance(row, dict):
            continue
        alphas[market] = row.get("alpha")
        incremental[market] = bool(
            row.get("incremental_model_value_vs_market", False)
        )
    return {
        "status": report.get("status", "MISSING"),
        "selected_alpha": alphas,
        "incremental_model_value": incremental,
        "incremental_model_value_markets": report.get(
            "incremental_model_value_markets"
        ),
        "canonical_market_probability_change_enabled": bool(
            report.get("canonical_market_probability_change_enabled", False)
        ),
        "betting_policy_change_enabled": bool(
            report.get("betting_policy_change_enabled", False)
        ),
        "interpretation": (
            "Archive probability shrinkage is downstream research only; "
            "market data never enters the independent fair-score model."
        ),
    }


def _forward_summary(report: dict[str, Any]) -> dict[str, object]:
    candidates = report.get("candidates")
    if not isinstance(candidates, dict):
        candidates = {}
    candidate_status = {
        name: {
            "status": value.get("status"),
            "graded_games": value.get("graded_games"),
            "minimum_games": value.get("minimum_games"),
            "promotion_eligible": bool(value.get("promotion_eligible", False)),
        }
        for name, value in candidates.items()
        if isinstance(value, dict)
    }
    canonical = report.get("canonical_betting")
    if not isinstance(canonical, dict):
        canonical = {}
    return {
        "phase_status": report.get("phase_status", "MISSING"),
        "candidate_status": candidate_status,
        "candidate_promotion_evidence_count": report.get(
            "candidate_promotion_evidence_count", 0
        ),
        "canonical_betting": {
            "status": canonical.get("status"),
            "graded_bets": canonical.get("graded_bets"),
            "minimum_bets": canonical.get("minimum_bets"),
            "forward_betting_gate_passed": bool(
                canonical.get("forward_betting_gate_passed", False)
            ),
        },
        "canonical_model_change_enabled": bool(
            report.get("canonical_model_change_enabled", False)
        ),
        "production_release_authority": bool(
            report.get("production_release_authority", False)
        ),
    }


def build_research_status(
    *,
    candidate_benchmark: dict[str, Any] | None = None,
    market_shrinkage: dict[str, Any] | None = None,
    forward_shadow: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Consolidate historical development and prospective evidence boundaries."""

    candidate = candidate_benchmark or {}
    market = market_shrinkage or {}
    forward = forward_shadow or {}

    candidate_summary = _candidate_summary(candidate)
    market_summary = _market_summary(market)
    forward_summary = _forward_summary(forward)

    active_forward = [
        name
        for name, value in forward_summary["candidate_status"].items()
        if value["status"] == "ACCUMULATING_FORWARD_EVIDENCE"
    ]
    promoted_forward = [
        name
        for name, value in forward_summary["candidate_status"].items()
        if bool(value["promotion_eligible"])
    ]

    historical_model_change_ready = False
    market_edge_change_ready = False
    forward_model_change_ready = bool(promoted_forward)

    return {
        "version": 1,
        "status": "TRACKING",
        "historical_fair_score_research": candidate_summary,
        "market_edge_research": market_summary,
        "forward_evidence": forward_summary,
        "active_forward_candidates": active_forward,
        "promotion_ready_forward_candidates": promoted_forward,
        "historical_model_change_ready": historical_model_change_ready,
        "market_edge_change_ready": market_edge_change_ready,
        "forward_model_change_ready": forward_model_change_ready,
        "canonical_model_change_enabled": False,
        "canonical_market_change_enabled": False,
        "research_decision": (
            "Keep the canonical fair-score and market-probability paths unchanged. "
            "Stage 14's QB-margin architecture was superseded by the fixed-spec "
            "Stage 15 margin rejection; Stage 30 found no incremental probability "
            "value versus the no-vig market. Continue only the already-frozen "
            "prospective 2026 candidate ledgers until their gates mature."
        ),
    }


def write_research_status(
    *,
    candidate_path: str | Path = DEFAULT_PATHS["candidate_benchmark"],
    market_path: str | Path = DEFAULT_PATHS["market_shrinkage"],
    forward_path: str | Path = DEFAULT_PATHS["forward_shadow"],
    report_path: str | Path = "reports/research_status.json",
    docs_path: str | Path = "docs/research_status.json",
) -> dict[str, object]:
    status = build_research_status(
        candidate_benchmark=_read_json(candidate_path),
        market_shrinkage=_read_json(market_path),
        forward_shadow=_read_json(forward_path),
    )
    payload = json.dumps(status, indent=2, sort_keys=True, default=str)
    for path in (report_path, docs_path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return status
