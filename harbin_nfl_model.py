"""Run the canonical Harbin NFL model and operational controls."""

from __future__ import annotations

import argparse
import json

from nfl.pipeline import run_operational_pipeline
from nfl.policy import load_policy
from nfl.portfolio_audit import write_portfolio_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--history-seasons", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-line-capture", action="store_true")
    parser.add_argument("--no-decision-ledger", action="store_true")
    args = parser.parse_args()

    current, report = run_operational_pipeline(
        args.season,
        args.week,
        history_seasons=args.history_seasons,
        refresh=args.refresh,
        capture_lines=not args.no_line_capture,
        persist_decisions=not args.no_decision_ledger,
    )
    bankroll = report["portfolio"].get("bankroll_risk", {})
    multiplier = (
        float(bankroll.get("risk_multiplier", 1.0))
        if isinstance(bankroll, dict)
        else 1.0
    )
    portfolio_audit = write_portfolio_audit(
        current,
        policy=load_policy(),
        release_gate=report["release_gate"],
        bankroll_multiplier=multiplier,
    )
    publication = report.get("publication", {})
    if not isinstance(publication, dict):
        publication = {}

    summary = {
        "status": "READY",
        "season": report["meta"]["season"],
        "week": report["meta"]["week"],
        "release_state": report["release_state"],
        "production_eligible": report["production_eligible"],
        "games": current.get_column("game_id").n_unique() if not current.is_empty() else 0,
        "market_candidates": current.height,
        "portfolio": report["portfolio"],
        "portfolio_audit": portfolio_audit["status"],
        "release_blockers": report["release_gate"].get("blockers", []),
        "outputs": {
            "json": "outputs/current_model.json",
            "csv": "outputs/current_predictions.csv",
            "health": "outputs/health.json",
            "release_gate": "outputs/release_gate.json",
            "model_card": "outputs/model_card.json",
            "portfolio_audit": "reports/portfolio_audit.json",
            "weekly_html": publication.get("html"),
            "weekly_png_pages": publication.get("png_pages", []),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
