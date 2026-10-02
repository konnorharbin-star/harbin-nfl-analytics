from __future__ import annotations

import argparse
import json

from nfl.forward_shadow import write_forward_shadow_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate all persisted NFL Phase 5 forward-shadow evidence."
    )
    parser.add_argument("--live", default="reports/live_performance.json")
    parser.add_argument("--recent-form", default="reports/recent_form_forward.json")
    parser.add_argument("--qb-total", default="reports/qb_total_forward.json")
    parser.add_argument("--probability", default="reports/probability_forward.json")
    parser.add_argument("--output", default="reports/forward_shadow_summary.json")
    parser.add_argument("--docs-output", default="docs/forward_shadow_summary.json")
    args = parser.parse_args()

    summary = write_forward_shadow_summary(
        live_path=args.live,
        recent_form_path=args.recent_form,
        qb_total_path=args.qb_total,
        probability_path=args.probability,
        report_path=args.output,
        docs_path=args.docs_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
