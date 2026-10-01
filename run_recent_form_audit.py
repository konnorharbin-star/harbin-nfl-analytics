from __future__ import annotations

import argparse
import json

from nfl.recent_form_audit import run_recent_form_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL recent-form PBP rolling-origin audit")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start-week", type=int, default=5)
    parser.add_argument("--end-week", type=int, default=18)
    args = parser.parse_args()

    report = run_recent_form_audit(
        start_week=args.start_week,
        end_week=args.end_week,
        refresh=args.refresh,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
