from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.isotonic_probability_audit import run_isotonic_probability_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/isotonic_probability_audit.json")
    args = parser.parse_args()

    report = run_isotonic_probability_audit(refresh=args.refresh).to_dict()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
