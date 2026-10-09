"""Phase 3: independently verified pregame lineups and shadow-score evaluation.

Does NOT learn point adjustments, modify picks or represent verified EA data.
"""

import csv
import math
from collections import defaultdict
from datetime import datetime

from ea_player_talent import load_snapshot


def _dt(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return d


def lineup_units(ratings_csv, lineup_csv, *, prediction_at, kickoff_at):
    """Require verified active starters + reserves; return missing rather than impute."""
    ratings = load_snapshot(ratings_csv, prediction_at=prediction_at, kickoff_at=kickoff_at)
    lookup = {r["player_id"]: r for r in ratings}
    with open(lineup_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"player_id", "team", "role", "observed_at"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Lineup missing required columns")
        lines = list(reader)
    cutoff = _dt(prediction_at)
    seen = set()
    groups = defaultdict(lambda: {"starter": [], "reserve": [], "out": []})
    for r in lines:
        pid = r["player_id"]
        if pid in seen or pid not in lookup:
            raise ValueError("Duplicate or unmapped lineup player ID")
        seen.add(pid)
        if _dt(r["observed_at"]) > cutoff:
            raise ValueError("Future lineup observation")
        if r["team"] != lookup[pid]["team"]:
            raise ValueError("Team mismatch")
        role = r["role"].strip().lower()
        if role not in ("starter", "reserve", "out"):
            raise ValueError("Unverified role")
        groups[(r["team"], lookup[pid]["_unit"])][role].append(lookup[pid]["_talent"])
    result = {}
    for (team, unit), roles in sorted(groups.items()):
        starters = roles["starter"]
        reserves = roles["reserve"]
        out = roles["out"]
        # This is not a sportsbook point adjustment. QB one-for-one replacement
        # is only defined when one active starter and >=1 confirmed reserve.
        replacement_gap = round(max(out) - max(reserves), 3) if out and reserves else None
        result[(team, unit)] = {
            "starter_count": len(starters),
            "reserve_count": len(reserves),
            "out_count": len(out),
            "starter_mean": round(sum(starters) / len(starters), 3) if starters else None,
            "reserve_mean": round(sum(reserves) / len(reserves), 3) if reserves else None,
            "unavailable_vs_reserve_gap": replacement_gap,
            "qb_ready": unit == "QB" and len(starters) == 1 and len(reserves) >= 1,
        }
    return result


def shadow_metrics(csv_path):
    """Compare precommitted base and challenger predicted margins, no price assumptions.

    Candidate forecasts must already be frozen and created before kickoff.
    No fitting is performed here, preventing in-sample evaluation disguised as proof.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "game_id",
            "season",
            "kickoff_at",
            "forecast_at",
            "base_margin",
            "candidate_margin",
            "actual_margin",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Shadow comparison missing fields")
        rows = list(reader)
    if not rows:
        raise ValueError("No shadow forecasts")
    ids = set()
    buckets = defaultdict(list)
    for r in rows:
        game = r["game_id"]
        if not game or game in ids:
            raise ValueError("Duplicate or missing game ID")
        ids.add(game)
        if _dt(r["forecast_at"]) >= _dt(r["kickoff_at"]):
            raise ValueError("Forecast after kickoff")
        nums = [float(r[k]) for k in ("base_margin", "candidate_margin", "actual_margin")]
        if not all(math.isfinite(v) for v in nums):
            raise ValueError("Nonfinite margin")
        buckets[r["season"]].append(nums)

    def scores(records):
        n = len(records)
        return {
            "games": n,
            "base_mae": round(sum(abs(b - y) for b, c, y in records) / n, 5),
            "candidate_mae": round(sum(abs(c - y) for b, c, y in records) / n, 5),
            "base_rmse": round(math.sqrt(sum((b - y) ** 2 for b, c, y in records) / n), 5),
            "candidate_rmse": round(math.sqrt(sum((c - y) ** 2 for b, c, y in records) / n), 5),
        }

    return {
        "overall": scores([v for values in buckets.values() for v in values]),
        "by_season": {year: scores(values) for year, values in sorted(buckets.items())},
        "status": "RESEARCH_ONLY_NOT_PROMOTED",
    }
