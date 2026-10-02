"""Run the real-source Stage 20 decoupled-score architecture audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.decoupled_audit import run_decoupled_score_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/decoupled_score_audit.json")
    args = parser.parse_args()

    result = run_decoupled_score_audit(refresh=args.refresh).to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
