from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.regime_correction_audit import run_regime_correction_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Stage 22 targeted regime-correction audit."
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        default="reports/regime_correction_audit.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    audit = run_regime_correction_audit(refresh=args.refresh)
    payload = audit.to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
