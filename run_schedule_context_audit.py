from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.schedule_context_audit import run_schedule_context_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage 20 schedule/venue context rolling audit."
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        default="reports/schedule_context_audit.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    audit = run_schedule_context_audit(refresh=args.refresh)
    payload = audit.to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
