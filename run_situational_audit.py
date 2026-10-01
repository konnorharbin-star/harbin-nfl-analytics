"""Run the Stage 17 situational PBP rolling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.situational_audit import run_situational_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/situational_audit.json")
    args = parser.parse_args()

    audit = run_situational_audit(refresh=args.refresh)
    payload = audit.to_dict()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
