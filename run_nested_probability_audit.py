"""Run the Stage 23 whole-week nested probability audit."""

from __future__ import annotations

import argparse
import json

from nfl.nested_probability_audit import run_nested_probability_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = run_nested_probability_audit(refresh=args.refresh).to_dict()
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
