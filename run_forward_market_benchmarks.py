"""Generate NFL Step 5 research baselines from the frozen, graded 2026 cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nfl.market_benchmarks import build_market_benchmarks, write_market_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market-only vs frozen-model prospective research comparisons"
    )
    parser.add_argument(
        "--graded", default="reports/forward_edge_graded.csv"
    )
    parser.add_argument(
        "--forward-report", default="reports/forward_edge_validation.json"
    )
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    graded_path = Path(args.graded)
    if not graded_path.exists() or not graded_path.stat().st_size:
        raise SystemExit(
            "Frozen forward grading artifact absent; run grade_forward_edge.py first"
        )
    graded = pl.read_csv(
        graded_path, try_parse_dates=False, infer_schema_length=2000
    )
    source = Path(args.forward_report)
    if not source.exists() or not source.stat().st_size:
        raise SystemExit("Frozen forward grading report absent; fail closed")
    report = json.loads(source.read_text(encoding="utf-8"))
    research = build_market_benchmarks(graded, forward_report=report)
    if research["status"] == "BLOCKED_SOURCE_INTEGRITY":
        raise SystemExit(
            "Market benchmark source-integrity failure; no output promoted: "
            + json.dumps(research["integrity"], sort_keys=True, default=str)
        )
    write_market_benchmarks(
        research,
        reports_dir=args.reports_dir,
        outputs_dir=args.outputs_dir,
        docs_dir=args.docs_dir,
    )
    print(json.dumps({
        "status": research["status"],
        "frozen": research["frozen"],
        "decided": research["decided"],
        "forecast_methods": list(research["predeclared_forecast_weights"]),
        "paper_rules": research["predeclared_flat_unit_rules"],
        "staking_authorized": research["staking_authorized"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
