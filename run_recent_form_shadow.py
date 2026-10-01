from __future__ import annotations

import argparse
import json

from nfl.recent_form_shadow import run_recent_form_shadow_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen NFL recent-form totals 2026 shadow audit"
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--shadow-start-week", type=int, default=3)
    parser.add_argument("--minimum-shadow-games", type=int, default=128)
    args = parser.parse_args()

    report = run_recent_form_shadow_audit(
        shadow_start_week=args.shadow_start_week,
        minimum_shadow_games=args.minimum_shadow_games,
        refresh=args.refresh,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
