"""Fail-closed Phase 4 evidence audit for EA talent experiments.

Reads a JSON manifest of externally verified inputs. Does not access sports books,
download EA data, fit models, or alter betting recommendations.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from ea_lineup_shadow import shadow_metrics

def _dt(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("All evidence timestamps must have timezone offsets")
    return dt

def evaluate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    gates = {}
    for name in ("ratings_archive", "roster_crosswalk", "pregame_lineups",
                 "immutable_forecast_log", "independent_holdout"):
        entry = manifest.get(name, {})
        gates[name] = bool(isinstance(entry, dict) and entry.get("verified") is True
                           and entry.get("path") and Path(entry["path"]).is_file())
    # Extra checks to prevent claiming proof on a mere checkbox.
    if gates["ratings_archive"]:
        entry = manifest["ratings_archive"]
        gates["ratings_archive"] = bool(entry.get("snapshot_at") and entry.get("acquired_at")
                                        and _dt(entry["snapshot_at"]) <= _dt(entry["acquired_at"]))
    report = {"status": "BLOCKED", "gates": gates,
              "missing": [k for k,v in gates.items() if not v],
              "metrics": None, "production_weight": 0.0}
    if not all(gates.values()):
        return report
    # Metrics are descriptive, not a trained or promotion-ready challenger.
    scores = shadow_metrics(manifest["immutable_forecast_log"]["path"])
    if scores["overall"]["games"] < int(manifest.get("minimum_holdout_games", 100)):
        report["missing"].append("holdout_sample_size")
        return report
    report["status"] = "RESEARCH_METRICS_AVAILABLE_NOT_PROMOTED"
    report["metrics"] = scores
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="ea_evidence_report.json")
    args = p.parse_args()
    result = evaluate_manifest(args.manifest)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
