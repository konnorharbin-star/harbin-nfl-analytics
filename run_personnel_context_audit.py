from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.personnel_context_audit import run_personnel_context_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leak-free fixed-spec non-QB personnel context audit."
        )
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        default="reports/personnel_context_audit.json",
    )
    args = parser.parse_args()

    audit = run_personnel_context_audit(refresh=args.refresh)
    payload = audit.to_dict()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
