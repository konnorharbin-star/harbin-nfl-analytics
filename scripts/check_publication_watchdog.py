"""Read-only NFL publication heartbeat and file integrity monitor."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl.publication_integrity import validate_publication_manifest
from nfl.publication_watchdog import evaluate_publication_watchdog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/current_model.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--github-summary", default="")
    args = parser.parse_args()
    model_path = Path(args.report)
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(model, dict):
            raise ValueError("canonical model is not a JSON object")
    except (OSError, ValueError) as exc:
        model = {}
        print(f"Publication read failure: {exc}")
    manifest = validate_publication_manifest(
        output_dir=args.output_dir, docs_dir=args.docs_dir
    )
    result = evaluate_publication_watchdog(
        model,
        now=datetime.now(UTC),
        manifest=manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.github_summary:
        details = [
            "## NFL publication watchdog",
            f"- Status: **{result['status']}**",
            f"- Latest model: **{result['model_generated_at']}** UTC",
            f"- Age: **{result['model_age_minutes']}** minutes",
            f"- Nearest kickoff: **{result['nearest_kickoff_minutes']}** minutes",
            f"- Files SHA-verified: **{result['verified_file_count']}**",
            f"- Release: **{result['release_state']}**",
            "- Failures: "
            + (
                "; ".join(result["blocking_reasons"])
                if result["blocking_reasons"] else "none"
            ),
        ]
        with Path(args.github_summary).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(details) + "\n")
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
