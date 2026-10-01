"""Run the real-source Stage 20 scoring-decomposition audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.decomposition_audit import run_decomposition_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", default="reports/decomposition_audit.json")
    args = parser.parse_args()

    result = run_decomposition_audit(refresh=args.refresh).to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
