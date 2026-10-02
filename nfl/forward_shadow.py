"""Consolidated Phase 5 forward-shadow evidence.

This module does not create or reconstruct predictions. It only reads the persisted,
independently graded forward reports produced by the canonical betting ledger and the
frozen recent-form, QB-total, and probability shadow systems.

The summary is an evidence index and phase gate. It never changes the canonical score
or probability model and it never opens PRODUCTION on its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

FORWARD_BET_MINIMUM = 300
SHADOW_MINIMUM = 128

DEFAULT_REPORT_PATHS = {
    "live": Path("reports/live_performance.json"),
    "recent_form_total": Path("reports/recent_form_forward.json"),
    "qb_total": Path("reports/qb_total_forward.json"),
    "probability": Path("reports/probability_forward.json"),
}


def _integer(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric_subset(report: dict[str, Any], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: report.get(key) for key in keys if key in report}


def _shadow_candidate(
    *,
    name: str,
    report: dict[str, Any],
    source_report: str,
    games_field: str = "graded_games",
    nested: dict[str, Any] | None = None,
    canonical_change_enabled: bool = False,
) -> dict[str, object]:
    target = nested if isinstance(nested, dict) else report
    games = _integer(report.get(games_field))
    minimum = _integer(report.get("minimum_games"), SHADOW_MINIMUM) or SHADOW_MINIMUM
    source_status = str(target.get("status") or report.get("status") or "MISSING_REPORT")
    reported_promotion = bool(target.get("promotion_eligible", False))

    if games < minimum:
        status = "ACCUMULATING_FORWARD_EVIDENCE"
        promotion_eligible = False
    elif reported_promotion:
        status = "PROMOTION_EVIDENCE"
        promotion_eligible = True
    elif source_status in {
        "SHADOW_FAILING",
        "SHADOW_INCONCLUSIVE",
        "PROMOTION_EVIDENCE",
    }:
        status = source_status
        promotion_eligible = False
    else:
        status = "FORWARD_EVIDENCE_INCONCLUSIVE"
        promotion_eligible = False

    return {
        "name": name,
        "source_report": source_report,
        "source_status": source_status,
        "ledger_rows": _integer(report.get("ledger_rows")),
        "graded_games": games,
        "minimum_games": minimum,
        "status": status,
        "promotion_eligible": promotion_eligible,
        "canonical_change_enabled": bool(canonical_change_enabled),
        "prospective_only": True,
        "retrospective_reconstruction_allowed": False,
    }


def _canonical_betting(report: dict[str, Any]) -> dict[str, object]:
    overall = report.get("overall")
    if not isinstance(overall, dict):
        overall = {}

    graded_bets = _integer(report.get("graded_bets"), _integer(overall.get("bets")))
    roi = _number(report.get("roi"))
    if roi is None:
        roi = _number(overall.get("roi"))
    avg_clv = _number(report.get("avg_clv"))
    if avg_clv is None:
        avg_clv = _number(overall.get("avg_clv"))
    verified = bool(report.get("portfolio_verified", False))

    if not verified:
        status = "INVALID_FORWARD_LEDGER"
        passed = False
    elif graded_bets < FORWARD_BET_MINIMUM:
        status = "ACCUMULATING_FORWARD_EVIDENCE"
        passed = False
    elif roi is None or avg_clv is None:
        status = "FORWARD_EVIDENCE_INCONCLUSIVE"
        passed = False
    elif roi >= 0.0 and avg_clv > 0.0:
        status = "FORWARD_EVIDENCE_READY"
        passed = True
    else:
        status = "FORWARD_EVIDENCE_FAILING"
        passed = False

    return {
        "source_report": "reports/live_performance.json",
        "evidence_source": report.get("evidence_source"),
        "portfolio_verified": verified,
        "graded_bets": graded_bets,
        "minimum_bets": FORWARD_BET_MINIMUM,
        "roi": roi,
        "avg_clv": avg_clv,
        "max_drawdown": _number(overall.get("max_drawdown")),
        "roi_ci_95": overall.get("roi_ci_95"),
        "status": status,
        "forward_betting_gate_passed": passed,
        "prospective_only": True,
        "retrospective_reconstruction_allowed": False,
    }


def build_forward_shadow_summary(
    *,
    live_report: dict[str, Any] | None = None,
    recent_form_report: dict[str, Any] | None = None,
    qb_total_report: dict[str, Any] | None = None,
    probability_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build the Phase 5 status from already-persisted forward evidence only."""

    live = live_report or {}
    recent = recent_form_report or {}
    qb = qb_total_report or {}
    probability = probability_report or {}

    canonical = _canonical_betting(live)

    recent_candidate = _shadow_candidate(
        name="recent_form_total",
        report=recent,
        source_report="reports/recent_form_forward.json",
        canonical_change_enabled=False,
    )
    recent_candidate["metrics"] = _metric_subset(
        recent,
        (
            "baseline_total_mae",
            "shadow_total_mae",
            "total_mae_improvement",
            "baseline_total_rmse",
            "shadow_total_rmse",
            "total_rmse_improvement",
        ),
    )

    qb_candidate = _shadow_candidate(
        name="qb_total",
        report=qb,
        source_report="reports/qb_total_forward.json",
        canonical_change_enabled=bool(qb.get("canonical_score_adjustment_enabled", False)),
    )
    qb_candidate["spec_version"] = qb.get("spec_version")
    qb_candidate["metrics"] = _metric_subset(
        qb,
        (
            "baseline_total_mae",
            "qb_total_shadow_mae",
            "total_mae_improvement",
            "baseline_total_rmse",
            "qb_total_shadow_rmse",
            "total_rmse_improvement",
        ),
    )

    home = probability.get("home_win")
    home_report = home if isinstance(home, dict) else {}
    home_games = _integer(
        probability.get("non_tied_games"),
        _integer(probability.get("graded_games")),
    )
    probability_home_input = dict(probability)
    probability_home_input["non_tied_games"] = home_games
    home_candidate = _shadow_candidate(
        name="probability_home_win",
        report=probability_home_input,
        source_report="reports/probability_forward.json",
        games_field="non_tied_games",
        nested=home_report,
        canonical_change_enabled=bool(
            probability.get("canonical_probability_change_enabled", False)
        ),
    )
    home_candidate["spec_version"] = probability.get("spec_version")
    home_candidate["metrics"] = _metric_subset(
        home_report,
        (
            "baseline_brier",
            "shadow_brier",
            "brier_improvement",
            "baseline_log_loss",
            "shadow_log_loss",
            "log_loss_improvement",
            "baseline_ece",
            "shadow_ece",
        ),
    )

    total_distribution = probability.get("total_distribution")
    total_report = total_distribution if isinstance(total_distribution, dict) else {}
    total_candidate = _shadow_candidate(
        name="probability_total_distribution",
        report=probability,
        source_report="reports/probability_forward.json",
        nested=total_report,
        canonical_change_enabled=bool(
            probability.get("canonical_probability_change_enabled", False)
        ),
    )
    total_candidate["spec_version"] = probability.get("spec_version")
    total_candidate["metrics"] = _metric_subset(
        total_report,
        (
            "baseline_nll",
            "shadow_nll",
            "nll_improvement",
        ),
    )

    candidates = {
        "recent_form_total": recent_candidate,
        "qb_total": qb_candidate,
        "probability_home_win": home_candidate,
        "probability_total_distribution": total_candidate,
    }
    promotion_candidates = [
        name
        for name, candidate in candidates.items()
        if bool(candidate["promotion_eligible"])
    ]

    forward_ready = bool(canonical["forward_betting_gate_passed"])
    phase_status = (
        "READY_FOR_RELEASE_REVIEW"
        if forward_ready
        else "ACCUMULATING_FORWARD_EVIDENCE"
    )

    return {
        "version": 1,
        "phase": "Phase 5 — Forward Shadow Validation",
        "phase_status": phase_status,
        "evidence_policy": {
            "prospective_only": True,
            "first_persisted_pregame_snapshot_only": True,
            "retrospective_reconstruction_allowed": False,
            "candidate_retuning_on_forward_outcomes_allowed": False,
        },
        "canonical_betting": canonical,
        "candidates": candidates,
        "candidate_promotion_evidence": promotion_candidates,
        "candidate_promotion_evidence_count": len(promotion_candidates),
        "forward_betting_gate_passed": forward_ready,
        "production_release_authority": False,
        "canonical_model_change_enabled": False,
        "meaning": (
            "Phase 5 consolidates independently persisted prospective evidence. "
            "Candidate promotion evidence requires its own frozen forward gate, while "
            "production remains controlled separately by the hard release gate."
        ),
    }


def write_forward_shadow_summary(
    *,
    live_path: str | Path = DEFAULT_REPORT_PATHS["live"],
    recent_form_path: str | Path = DEFAULT_REPORT_PATHS["recent_form_total"],
    qb_total_path: str | Path = DEFAULT_REPORT_PATHS["qb_total"],
    probability_path: str | Path = DEFAULT_REPORT_PATHS["probability"],
    report_path: str | Path = "reports/forward_shadow_summary.json",
    docs_path: str | Path = "docs/forward_shadow_summary.json",
) -> dict[str, object]:
    """Persist a consolidated read-only summary of existing forward reports."""

    summary = build_forward_shadow_summary(
        live_report=_load_json(live_path),
        recent_form_report=_load_json(recent_form_path),
        qb_total_report=_load_json(qb_total_path),
        probability_report=_load_json(probability_path),
    )
    payload = json.dumps(summary, indent=2, sort_keys=True, default=str)
    for path in (report_path, docs_path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return summary
