"""Grade completed NFL paper/shadow/production portfolio decisions."""

from __future__ import annotations

import json
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.grading import grade_live_from_files, load_decision_ledger
from nfl.grading_audit import audit_decisions, audit_graded_bets


def main() -> None:
    decisions = load_decision_ledger()
    if decisions.is_empty():
        payload = {
            "evidence_source": "portfolio_decisions_v1",
            "portfolio_verified": True,
            "graded_bets": 0,
            "status": "NO_DECISIONS",
            "grading_audit": audit_decisions(decisions),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    decision_audit = audit_decisions(decisions)
    if decision_audit["errors"]:
        raise RuntimeError("decision-ledger timing audit failed")

    seasons = sorted(
        {int(value) for value in decisions.get_column("season").drop_nulls().to_list()}
    )
    client = NFLDataClient()
    schedules = client.load_schedules(seasons, refresh=True)
    graded, summary = grade_live_from_files(schedules)
    graded_audit = audit_graded_bets(graded)
    if graded_audit["errors"]:
        raise RuntimeError("graded-bet timing/integrity audit failed")

    summary["status"] = "READY"
    summary["grading_audit"] = {
        "status": (
            "WARN"
            if decision_audit["warnings"] or graded_audit["warnings"]
            else "PASS"
        ),
        "decisions": decision_audit,
        "graded": graded_audit,
    }

    reports = Path("reports/grading_audit.json")
    reports.parent.mkdir(parents=True, exist_ok=True)
    reports.write_text(
        json.dumps(summary["grading_audit"], indent=2, sort_keys=True, default=str)
    )
    docs = Path("docs/live_performance.json")
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
