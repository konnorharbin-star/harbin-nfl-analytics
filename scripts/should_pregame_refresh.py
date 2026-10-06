"""Emit GitHub Actions outputs when an NFL pregame refresh is due."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl.pregame_refresh import load_refresh_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/current_model.json")
    parser.add_argument("--github-output")
    args = parser.parse_args()

    result = load_refresh_checkpoint(args.report)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.github_output:
        target = Path(args.github_output)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"run={'true' if result.get('run') else 'false'}\n")
            handle.write(f"season={result.get('season') or ''}\n")
            handle.write(f"week={result.get('week') or ''}\n")
            handle.write(
                f"checkpoint={result.get('checkpoint_minutes') or ''}\n"
            )


if __name__ == "__main__":
    main()
