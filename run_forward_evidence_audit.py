from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.forward_evidence import audit_forward_evidence
from nfl.qb_total_forward import load_qb_total_forward_predictions
from nfl.recent_form_forward import load_recent_form_forward_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit integrity of the frozen 2026 forward-shadow ledgers."
    )
    parser.add_argument(
        "--recent-form-ledger",
        default="history/recent_form_shadow_predictions_v1.csv",
    )
    parser.add_argument(
        "--qb-total-ledger",
        default="history/qb_total_shadow_predictions_v1.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/forward_evidence_integrity.json",
    )
    args = parser.parse_args()

    recent = load_recent_form_forward_predictions(args.recent_form_ledger)
    qb = load_qb_total_forward_predictions(args.qb_total_ledger)
    audit = audit_forward_evidence(
        recent,
        qb,
        recent_form_path=args.recent_form_ledger,
        qb_total_path=args.qb_total_ledger,
    )
    payload = audit.to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if audit.overall_status != "PASS":
        raise SystemExit("forward evidence integrity audit failed")


if __name__ == "__main__":
    main()
