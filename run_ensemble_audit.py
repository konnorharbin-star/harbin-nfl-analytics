from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.ensemble_audit import run_ensemble_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/ensemble_audit.json")
    args = parser.parse_args()

    report = run_ensemble_audit(refresh=args.refresh).to_dict()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
