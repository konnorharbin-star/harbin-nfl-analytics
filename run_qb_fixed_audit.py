from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.qb_fixed_audit import run_fixed_qb_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-spec rolling-origin QB development audit."
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/qb_fixed_audit.json")
    args = parser.parse_args()

    audit = run_fixed_qb_audit(refresh=args.refresh)
    payload = audit.to_dict()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
