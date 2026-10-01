"""Audit forward NFL decision/grading evidence for timing and portfolio integrity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .contracts import require_columns


def _parse(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def audit_decisions(decisions: pl.DataFrame) -> dict[str, object]:
    """Audit cap-constrained decision rows before independent grading."""

    if decisions.is_empty():
        return {
            "status": "WARN",
            "errors": 0,
            "warnings": 1,
            "eligible_decisions": 0,
            "timing_excluded_rows": 0,
            "issues": [{"severity": "WARNING", "code": "no_forward_decisions"}],
        }
    require_columns(
        decisions,
        {
            "decision_at",
            "kickoff",
            "game_id",
            "quant_market",
            "portfolio_candidate_units",
            "portfolio_action",
            "execution_ready",
        },
        "portfolio_decisions",
    )

    issues: list[dict[str, object]] = []
    timing_excluded = 0
    eligible = 0
    execution_excluded = 0
    for row in decisions.iter_rows(named=True):
        units = float(row.get("portfolio_candidate_units") or 0.0)
        action = str(row.get("portfolio_action") or "PASS").upper()
        if units <= 0 or action not in {"PAPER", "SHADOW", "BET"}:
            continue
        decision = _parse(row.get("decision_at"))
        kickoff = _parse(row.get("kickoff"))
        if decision is None or kickoff is None or decision >= kickoff:
            timing_excluded += 1
            continue
        if not bool(row.get("execution_ready")):
            execution_excluded += 1
            continue
        eligible += 1

    if timing_excluded:
        issues.append(
            {
                "severity": "ERROR",
                "code": "non_pre_kickoff_decisions",
                "rows": timing_excluded,
            }
        )
    if execution_excluded:
        issues.append(
            {
                "severity": "WARNING",
                "code": "non_execution_ready_decisions",
                "rows": execution_excluded,
            }
        )

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "eligible_decisions": eligible,
        "timing_excluded_rows": timing_excluded,
        "execution_excluded_rows": execution_excluded,
        "portfolio_verified": True,
        "evidence_source": "portfolio_decisions_v1",
        "issues": issues,
    }


def audit_graded_bets(graded: pl.DataFrame) -> dict[str, object]:
    """Audit already graded forward bets and close-snapshot chronology."""

    if graded.is_empty():
        return {
            "status": "WARN",
            "errors": 0,
            "warnings": 1,
            "graded_bets": 0,
            "issues": [{"severity": "WARNING", "code": "no_graded_forward_bets"}],
        }
    require_columns(
        graded,
        {
            "game_id",
            "quant_market",
            "decision_at",
            "kickoff",
            "portfolio_verified",
            "result",
            "net_units",
        },
        "live_graded_bets",
    )

    issues: list[dict[str, object]] = []
    duplicate_groups = (
        graded.group_by(["game_id", "quant_market"]).len().filter(pl.col("len") > 1).height
    )
    if duplicate_groups:
        issues.append(
            {
                "severity": "ERROR",
                "code": "duplicate_graded_game_market",
                "groups": duplicate_groups,
            }
        )

    not_verified = graded.filter(~pl.col("portfolio_verified").cast(pl.Boolean)).height
    if not_verified:
        issues.append(
            {
                "severity": "ERROR",
                "code": "unverified_portfolio_evidence",
                "rows": not_verified,
            }
        )

    bad_entry_timing = 0
    bad_close_timing = 0
    for row in graded.iter_rows(named=True):
        decision = _parse(row.get("decision_at"))
        kickoff = _parse(row.get("kickoff"))
        if decision is None or kickoff is None or decision >= kickoff:
            bad_entry_timing += 1
        close = _parse(row.get("closing_snapshot_at"))
        if close is not None and (
            decision is None
            or kickoff is None
            or close <= decision
            or close >= kickoff
        ):
            bad_close_timing += 1
    if bad_entry_timing:
        issues.append(
            {
                "severity": "ERROR",
                "code": "graded_entry_not_pre_kickoff",
                "rows": bad_entry_timing,
            }
        )
    if bad_close_timing:
        issues.append(
            {
                "severity": "ERROR",
                "code": "invalid_closing_snapshot_timing",
                "rows": bad_close_timing,
            }
        )

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "graded_bets": graded.height,
        "portfolio_verified": not not_verified,
        "issues": issues,
    }


def audit_grading_files(
    *,
    decisions_path: str | Path = "history/portfolio_decisions_v1.csv",
    graded_path: str | Path = "reports/live_graded_bets.csv",
    output: str | Path = "reports/grading_audit.json",
    fail_on_error: bool = True,
) -> dict[str, object]:
    decisions_file = Path(decisions_path)
    graded_file = Path(graded_path)
    decisions = (
        pl.read_csv(decisions_file, try_parse_dates=False)
        if decisions_file.exists() and decisions_file.stat().st_size
        else pl.DataFrame()
    )
    graded = (
        pl.read_csv(graded_file, try_parse_dates=False)
        if graded_file.exists() and graded_file.stat().st_size
        else pl.DataFrame()
    )
    decision_report = audit_decisions(decisions)
    graded_report = audit_graded_bets(graded)
    errors = int(decision_report["errors"]) + int(graded_report["errors"])
    warnings = int(decision_report["warnings"]) + int(graded_report["warnings"])
    report = {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "decisions": decision_report,
        "graded": graded_report,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if fail_on_error and errors:
        raise RuntimeError("grading audit failed")
    return report
