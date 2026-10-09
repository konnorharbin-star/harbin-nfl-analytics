"""Point-in-time EA player talent features. Research-only; never adjusts picks.

Input CSV: player_id,team,position,ovr,snapshot_at; optional
awr,spd,str,agi,cod,inj,available,depth_rank.
snapshot_at must be an ISO-8601 timestamp with timezone.
This loader uses only explicitly archived, dated records; it never fetches live ratings.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

UNITS = {
    "QB": "QB",
    "RB": "SKILL",
    "FB": "SKILL",
    "WR": "SKILL",
    "TE": "SKILL",
    "LT": "OL",
    "LG": "OL",
    "C": "OL",
    "RG": "OL",
    "RT": "OL",
    "OL": "OL",
    "LE": "FRONT",
    "RE": "FRONT",
    "DT": "FRONT",
    "DE": "FRONT",
    "EDGE": "FRONT",
    "LOLB": "FRONT",
    "MLB": "FRONT",
    "ROLB": "FRONT",
    "LB": "FRONT",
    "CB": "SECONDARY",
    "FS": "SECONDARY",
    "SS": "SECONDARY",
    "S": "SECONDARY",
    "K": "SPECIAL",
    "P": "SPECIAL",
}
CAPS = {"QB": 1, "SKILL": 5, "OL": 5, "FRONT": 7, "SECONDARY": 5, "SPECIAL": 2}
ATTRS = {
    "QB": ("awr", "agi"),
    "SKILL": ("spd", "agi", "cod"),
    "OL": ("str", "awr"),
    "FRONT": ("str", "spd", "awr"),
    "SECONDARY": ("spd", "agi", "cod", "awr"),
    "SPECIAL": ("awr",),
}


def _timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamps must carry a timezone offset")
    return dt


def _score(row: dict, unit: str) -> float:
    ovr = int(row["ovr"])
    if not 0 <= ovr <= 99:
        raise ValueError("OVR outside 0..99")
    available = [float(row[k]) for k in ATTRS[unit] if row.get(k, "").strip()]
    if any(not 0 <= a <= 99 for a in available):
        raise ValueError("Attribute outside 0..99")
    # Small contextual component; rating scale stays 0..99.
    return 0.85 * ovr + 0.15 * (sum(available) / len(available) if available else ovr)


def load_snapshot(path: str | Path, *, prediction_at: str, kickoff_at: str) -> list[dict]:
    """Reject post-prediction snapshots, post-kickoff games and mixed snapshots."""
    prediction, kickoff = _timestamp(prediction_at), _timestamp(kickoff_at)
    if prediction >= kickoff:
        raise ValueError("prediction_at must precede kickoff_at")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        mandatory = {"player_id", "team", "position", "ovr", "snapshot_at"}
        if not mandatory.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing CSV columns: {sorted(mandatory - set(reader.fieldnames or []))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Empty ratings snapshot")
    dates = {_timestamp(r["snapshot_at"]) for r in rows}
    if len(dates) != 1 or next(iter(dates)) > prediction:
        raise ValueError("Mixed, undated, or future ratings snapshot")
    seen = set()
    for r in rows:
        key = r["player_id"]
        if not key or key in seen or not r["team"].strip():
            raise ValueError("Duplicate/missing player id or missing team")
        seen.add(key)
        position = r["position"].strip().upper()
        if position not in UNITS:
            raise ValueError(f"Unsupported position: {position}")
        r["position"] = position
        r["_unit"] = UNITS[position]
        r["_talent"] = _score(r, r["_unit"])
        # EA's INJ/durability attribute is not a medical availability report.
        # Missing third-party lineup status must never be treated as healthy.
        status = (r.get("available") or "").strip().lower()
        if status and status not in {"0", "1", "true", "false", "yes", "no"}:
            raise ValueError("available must be a boolean if supplied")
        r["_availability_known"] = bool(status)
        r["_available"] = status in {"1", "true", "yes"}
    return rows


def team_features(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    """Top-rated active player groups and estimated replacements, not point spreads."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["team"], r["_unit"])].append(r)
    teams = {r["team"] for r in rows}
    output = {}
    for team in sorted(teams):
        feats = {}
        for unit, cap in CAPS.items():
            players = sorted(groups.get((team, unit), []), key=lambda r: r["_talent"], reverse=True)
            active = [r for r in players if r["_available"]]
            full = players[:cap]
            actual = active[:cap]
            # Do not impute a missing player or a missing lineup.
            feats[f"{unit.lower()}_talent"] = (
                round(sum(p["_talent"] for p in actual) / cap, 3) if len(actual) >= cap else None
            )
            feats[f"{unit.lower()}_full_talent"] = (
                round(sum(p["_talent"] for p in full) / cap, 3) if len(full) >= cap else None
            )
            feats[f"{unit.lower()}_availability_gap"] = (
                round(feats[f"{unit.lower()}_full_talent"] - feats[f"{unit.lower()}_talent"], 3)
                if feats[f"{unit.lower()}_talent"] is not None
                and feats[f"{unit.lower()}_full_talent"] is not None
                else None
            )
            feats[f"{unit.lower()}_covered"] = len(actual) >= cap
        output[team] = feats
    return output


def matchup_features(team_features_by_team: dict, home: str, away: str) -> dict:
    """Home-minus-away player talent differentials; no learned points conversion."""
    if home not in team_features_by_team or away not in team_features_by_team:
        raise KeyError("Both team ratings required")
    return {
        f"{unit.lower()}_talent_diff": round(
            team_features_by_team[home][f"{unit.lower()}_talent"]
            - team_features_by_team[away][f"{unit.lower()}_talent"],
            3,
        )
        if team_features_by_team[home][f"{unit.lower()}_talent"] is not None
        and team_features_by_team[away][f"{unit.lower()}_talent"] is not None
        else None
        for unit in CAPS
    }
