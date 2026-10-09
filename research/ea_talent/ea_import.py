"""EA ratings normalization and roster crosswalk, research only.

Never infer the snapshot publication time from file modification/download time.
Require provenance metadata and explicit player-ID mappings; ambiguous joins fail.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

REQUIRED_RAW = {"ea_player_id", "player_name", "team", "position", "ovr"}
REQUIRED_MAP = {"ea_player_id", "model_player_id", "team", "effective_from", "effective_to"}


def _time(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("Timezone-aware ISO timestamps required")
    return d


def _read(path, fields):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns: " + ", ".join(sorted(fields - set(reader.fieldnames or []))))
        return list(reader)


def normalize(raw_path, crosswalk_path, metadata_path, output_path, *, as_of, min_coverage=0.0):
    """Normalize licensed/manual EA export to existing adapter contract.

    The mapping is keyed strictly by EA player ID + team + validity interval.
    Mapping rows cannot overlap for an EA player at the requested cutoff.
    Nonmatches are excluded and reported, never silently guessed by name.
    """
    cutoff = _time(as_of)
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    required = {"source_url", "snapshot_at", "obtained_at", "source_sha256", "license_note"}
    if not required.issubset(meta):
        raise ValueError("Incomplete immutable provenance metadata")
    snap, obtained = _time(meta["snapshot_at"]), _time(meta["obtained_at"])
    if snap > cutoff or obtained > cutoff or obtained < snap:
        raise ValueError("Future snapshot or inconsistent acquisition timestamps")
    if not meta["source_url"].startswith("https://") or not meta["license_note"].strip():
        raise ValueError("Unverified source reference or missing usage note")
    actual_hash = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    if actual_hash.lower() != meta["source_sha256"].lower():
        raise ValueError("Ratings CSV digest mismatch")
    raw = _read(raw_path, REQUIRED_RAW)
    maps = _read(crosswalk_path, REQUIRED_MAP)
    by_ea = {}
    for entry in maps:
        start = _time(entry["effective_from"])
        end = _time(entry["effective_to"]) if entry["effective_to"].strip() else None
        if end and end <= start:
            raise ValueError("Invalid crosswalk interval")
        if start <= cutoff and (end is None or cutoff < end):
            by_ea.setdefault(entry["ea_player_id"], []).append(entry)
    output, unmatched, ambiguous, seen_ea, used_ids = [], [], [], set(), set()
    passthrough = ["awr", "spd", "str", "agi", "cod", "inj", "available"]
    for item in raw:
        ea_id = item["ea_player_id"].strip()
        if not ea_id or ea_id in seen_ea:
            raise ValueError("Duplicate or empty EA player ID")
        seen_ea.add(ea_id)
        matches = [m for m in by_ea.get(ea_id, []) if m["team"].strip() == item["team"].strip()]
        if len(matches) > 1:
            ambiguous.append(ea_id)
            continue
        if not matches:
            unmatched.append(ea_id)
            continue
        model_id = matches[0]["model_player_id"].strip()
        if not model_id or model_id in used_ids:
            raise ValueError("Missing or duplicated model player identity")
        used_ids.add(model_id)
        row = {"player_id": model_id, "team": item["team"].strip(),
               "position": item["position"].strip().upper(), "ovr": item["ovr"],
               "snapshot_at": meta["snapshot_at"]}
        row.update({k: item.get(k, "") for k in passthrough if item.get(k, "") != ""})
        output.append(row)
    if ambiguous:
        raise ValueError("Ambiguous active crosswalk IDs: " + ", ".join(ambiguous[:10]))
    coverage = len(output) / len(raw) if raw else 0.0
    if not raw or coverage < min_coverage:
        raise ValueError(f"Crosswalk coverage {coverage:.1%} below required {min_coverage:.1%}")
    # Reuse strict rating and position validation in the sibling adapter.
    from ea_player_talent import load_snapshot
    fields = ["player_id", "team", "position", "ovr", "snapshot_at"] + passthrough
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {out}")
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in output:
                writer.writerow(item)
        # Contract validation and point-in-time guarantee before publication.
        from datetime import timedelta
        load_snapshot(tmp, prediction_at=as_of, kickoff_at=(cutoff + timedelta(seconds=1)).isoformat())
        tmp.replace(out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"raw_players": len(raw), "matched_players": len(output),
            "unmatched_players": len(unmatched), "coverage": coverage,
            "snapshot_at": meta["snapshot_at"], "raw_sha256": actual_hash,
            "unmatched_ids": unmatched, "output": str(out)}
