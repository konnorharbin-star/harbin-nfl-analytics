"""Build *review-only* roster crosswalk candidates from independently sourced CSVs.

No name match is accepted automatically. The resulting template must be
human-reviewed before use by ea_import.normalize().
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
from collections import defaultdict


def _name(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _rows(path, required):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required columns")
        return list(reader)


def review_candidates(ea_csv, roster_csv, output_csv):
    ea = _rows(ea_csv, {"ea_player_id", "player_name", "team", "position"})
    roster = _rows(roster_csv, {"model_player_id", "player_name", "team", "position"})
    teams = defaultdict(list)
    for r in roster:
        teams[(r["team"].strip().casefold(), r["position"].strip().casefold())].append(r)
    output = []
    seen = set()
    for r in ea:
        key = r["ea_player_id"].strip()
        if not key or key in seen:
            raise ValueError("Duplicate or blank EA id")
        seen.add(key)
        options = []
        for m in teams[(r["team"].strip().casefold(), r["position"].strip().casefold())]:
            confidence = difflib.SequenceMatcher(
                None, _name(r["player_name"]), _name(m["player_name"])
            ).ratio()
            options.append((confidence, m))
        options.sort(key=lambda x: x[0], reverse=True)
        best = options[0] if options else None
        # Candidate identity is advisory, never becomes a verified crosswalk automatically.
        output.append(
            {
                "ea_player_id": key,
                "ea_player_name": r["player_name"],
                "ea_team": r["team"],
                "position": r["position"],
                "suggested_model_player_id": best[1]["model_player_id"] if best else "",
                "suggested_model_player_name": best[1]["player_name"] if best else "",
                "name_similarity": f"{best[0]:.3f}" if best else "",
                "review_decision": "UNREVIEWED",
                "verified_model_player_id": "",
            }
        )
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(output[0])
            if output
            else [
                "ea_player_id",
                "ea_player_name",
                "ea_team",
                "position",
                "suggested_model_player_id",
                "suggested_model_player_name",
                "name_similarity",
                "review_decision",
                "verified_model_player_id",
            ],
        )
        w.writeheader()
        w.writerows(output)
    return {"ea_players": len(ea), "review_required": len(output), "automatically_approved": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ea_csv")
    p.add_argument("roster_csv")
    p.add_argument("output_csv")
    a = p.parse_args()
    print(review_candidates(a.ea_csv, a.roster_csv, a.output_csv))


if __name__ == "__main__":
    main()
