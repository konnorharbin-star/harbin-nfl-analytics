"""Build the canonical NFL engineering/evidence audit snapshot."""

from __future__ import annotations

import argparse
import json

from nfl.audit_snapshot import write_audit_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--history-dir", default="history")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = write_audit_snapshot(
        reports_dir=args.reports_dir,
        outputs_dir=args.outputs_dir,
        history_dir=args.history_dir,
        fail_on_error=not args.no_fail,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
