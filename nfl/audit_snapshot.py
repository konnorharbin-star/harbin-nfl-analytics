"""Canonical machine-readable audit snapshot across NFL model evidence layers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .backtest_audit import audit_backtest_bets
from .dataset import FEATURE_NAMES
from .feature_audit import leakage_columns
from .grading_audit import audit_decisions, audit_graded_bets
from .policy import load_policy
from .portfolio_audit import audit_portfolio


def _read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    try:
        return pl.read_csv(source, try_parse_dates=False)
    except Exception:
        return pl.DataFrame()


def _feature_contract() -> dict[str, object]:
    model_columns = [f"{name}_matchup_advantage" for name in FEATURE_NAMES]
    leaking = leakage_columns(model_columns)
    return {
        "status": "FAIL" if leaking else "PASS",
        "errors": 1 if leaking else 0,
        "warnings": 0,
        "model_columns": model_columns,
        "leakage_columns": leaking,
        "score_model_market_inputs": False,
        "note": (
            "Static contract audit. Dynamic training/live drift is reported separately "
            "when reports/feature_audit.json exists."
        ),
    }


def build_audit_snapshot(
    *,
    reports_dir: str | Path = "reports",
    outputs_dir: str | Path = "outputs",
    history_dir: str | Path = "history",
) -> dict[str, object]:
    reports = Path(reports_dir)
    outputs = Path(outputs_dir)
    history = Path(history_dir)

    backtest = audit_backtest_bets(_read_csv(reports / "free_market_bets.csv"))
    decisions = audit_decisions(_read_csv(history / "portfolio_decisions_v1.csv"))
    graded = audit_graded_bets(_read_csv(reports / "live_graded_bets.csv"))

    current = _read_csv(outputs / "current_predictions.csv")
    current_model = _read_json(outputs / "current_model.json")
    release_gate = _read_json(outputs / "release_gate.json")
    portfolio_meta = current_model.get("portfolio")
    if not isinstance(portfolio_meta, dict):
        portfolio_meta = {}
    bankroll = portfolio_meta.get("bankroll_risk")
    if not isinstance(bankroll, dict):
        bankroll = {}
    portfolio = audit_portfolio(
        current,
        policy=load_policy(),
        release_gate=release_gate,
        bankroll_multiplier=float(bankroll.get("risk_multiplier", 1.0) or 1.0),
    )

    dynamic_feature = _read_json(reports / "feature_audit.json")
    feature = _feature_contract()
    if dynamic_feature:
        feature["dynamic"] = dynamic_feature
        feature["errors"] = int(feature["errors"]) + int(dynamic_feature.get("errors", 0) or 0)
        feature["warnings"] = int(feature["warnings"]) + int(
            dynamic_feature.get("warnings", 0) or 0
        )
        feature["status"] = (
            "FAIL"
            if feature["errors"]
            else "WARN"
            if feature["warnings"]
            else "PASS"
        )

    health = _read_json(outputs / "health.json")
    sections = {
        "feature": feature,
        "backtest": backtest,
        "grading_decisions": decisions,
        "grading_results": graded,
        "portfolio": portfolio,
    }
    errors = sum(int(section.get("errors", 0) or 0) for section in sections.values())
    warnings = sum(
        int(section.get("warnings", 0) or 0) for section in sections.values()
    )
    status = "FAIL" if errors else "WARN" if warnings else "PASS"
    production_eligible = bool(release_gate.get("production_eligible", False))
    if production_eligible and status != "PASS":
        status = "FAIL"
        errors += 1
        sections["cross_check"] = {
            "status": "FAIL",
            "errors": 1,
            "warnings": 0,
            "code": "production_gate_open_with_nonpassing_audit",
        }

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "release_state": release_gate.get("release_state", "UNKNOWN"),
        "production_eligible": production_eligible,
        "health_status": health.get("status", "UNKNOWN"),
        "sections": sections,
        "meaning": (
            "Consolidated engineering/evidence audit. PASS is necessary but not sufficient "
            "for a betting edge or production release."
        ),
    }


def write_audit_snapshot(
    *,
    reports_dir: str | Path = "reports",
    outputs_dir: str | Path = "outputs",
    history_dir: str | Path = "history",
    output: str | Path = "reports/audit_snapshot.json",
    fail_on_error: bool = True,
) -> dict[str, object]:
    report = build_audit_snapshot(
        reports_dir=reports_dir,
        outputs_dir=outputs_dir,
        history_dir=history_dir,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if fail_on_error and report["errors"]:
        raise RuntimeError("canonical NFL audit snapshot failed")
    return report
