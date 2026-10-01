"""Grade completed NFL paper/shadow/production portfolio decisions."""

from __future__ import annotations

import json
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.grading import grade_live_from_files, load_decision_ledger


def main() -> None:
    decisions = load_decision_ledger()
    if decisions.is_empty():
        payload = {
            "evidence_source": "portfolio_decisions_v1",
            "portfolio_verified": True,
            "graded_bets": 0,
            "status": "NO_DECISIONS",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    seasons = sorted({int(value) for value in decisions.get_column("season").drop_nulls().to_list()})
    client = NFLDataClient()
    schedules = client.load_schedules(seasons, refresh=True)
    _, summary = grade_live_from_files(schedules)
    summary["status"] = "READY"

    docs = Path("docs/live_performance.json")
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
