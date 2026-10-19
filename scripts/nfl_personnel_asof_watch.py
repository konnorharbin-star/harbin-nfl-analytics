"""Record as-of player/QB uncertainty alongside an existing independent forecast.

The default run is append-only and does not change scores or approve bets.
"""
import argparse
from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json

from nfl.coupled_forward_grading import clock
from nfl.personnel_asof_watch import watch, append, read_journal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast", type=Path, default=Path("outputs/independent_forecasts.json"))
    p.add_argument("--context", type=Path, default=Path("reports/nfl_web_context_research.json"))
    p.add_argument("--journal", type=Path, default=Path("history/personnel_asof_v1_2026/observations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/nfl_personnel_asof_watch.json"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--check-pregame", action="store_true")
    a = p.parse_args()
    at = datetime.now(UTC)
    if a.check_pregame:
        for row in read_journal(a.journal):
            if row.get("source_run_id") == a.run_id and at >= clock(row["kickoff_utc"]):
                raise ValueError("PERSONNEL_OBSERVATION_BECAME_POSTKICKOFF_BEFORE_COMMIT")
        print(json.dumps({"status": "PREGAME_CHECK_PASSED", "betting_authorized": False}))
        return

    forecast_bytes = a.forecast.read_bytes()
    context_bytes = a.context.read_bytes()
    bundle = json.loads(forecast_bytes)
    context = json.loads(context_bytes)
    report = watch(bundle, context, captured=at,
                   bundle_sha=hashlib.sha256(forecast_bytes).hexdigest(),
                   research_sha=hashlib.sha256(context_bytes).hexdigest(),
                   run_id=a.run_id)
    archive = append(report, a.journal)
    report["immutable_journal"] = str(a.journal)
    report["append_receipt"] = archive
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "pregame_games": report["eligible_pregame_games"],
        "model_qb_starters_confirmed": 0,
        "new_immutable_context_records": archive["new_observations"],
        "betting_authorized": False,
        "forecasts_changed": False,
    }))


if __name__ == "__main__":
    main()
