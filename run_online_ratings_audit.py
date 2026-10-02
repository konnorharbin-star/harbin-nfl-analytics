"""Run the real-source Stage 21 NCAA-style online ratings audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.online_ratings_audit import run_online_ratings_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/online_ratings_audit.json")
    args = parser.parse_args()

    result = run_online_ratings_audit(refresh=args.refresh).to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
