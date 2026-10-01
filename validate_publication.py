"""Validate that public NFL publication files reconcile with canonical model output."""

from __future__ import annotations

import json

from nfl.publication import validate_publication_files


def main() -> None:
    report = validate_publication_files()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
