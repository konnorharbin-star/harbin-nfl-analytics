"""Point-in-time lineup replacement evidence, without betting-score conversion.

A QB gap is the rating of the pregame depth-chart QB1 *ruled out* minus
the rating of the confirmed replacement *who will start*. Unknown statuses
are never interpreted as healthy. Injury durability attributes are not injury
reports. All lineup evidence must be independently reviewed and source dated.
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
    """Return research-only QB replacement gaps or None when unverified.

    Required lineup CSV:
    player_id,team,role,depth_rank,observed_at,obtained_at,source_url,verified

    The depth ranking is *pregame*; 'starter' denotes an explicitly verified
    expected starting assignment, 'out' a confirmed unavailability designation.
    The reported source is self-attested and must be checked independently.
    """
    ratings = load_snapshot(
        ratings_csv,
        prediction_at=prediction_at,
        kickoff_at=kickoff_at,
    )
    by_player = {r["player_id"]: r for r in ratings}
    with open(lineup_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "player_id",
            "team",
            "role",
            "depth_rank",
            "observed_at",
            "obtained_at",
            "source_url",
            "verified",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Lineup missing verified source/role/depth columns")
        lines = list(reader)

    cutoff = _dt(prediction_at)
    kickoff = _dt(kickoff_at)
    if cutoff >= kickoff:
        raise ValueError("Prediction must precede kickoff")

    by_team_unit = defaultdict(list)
    keys = set()
    ranks = set()
    for row in lines:
        player_id = (row["player_id"] or "").strip()
        if player_id in keys or player_id not in by_player:
            raise ValueError("Duplicate or unmapped lineup player ID")
        keys.add(player_id)
        rated = by_player[player_id]
        team = (row["team"] or "").strip()
        if team != rated["team"]:
            raise ValueError("Team mismatch in lineup versus ratings crosswalk")
        if (row["verified"] or "").strip().lower() != "true":
            raise ValueError("Lineup record is not explicitly source-verified")
        url = (row["source_url"] or "").strip()
        if not url.startswith("https://"):
            raise ValueError("Lineup record lacks HTTPS source provenance")
        observed = _dt((row["observed_at"] or "").strip())
        obtained = _dt((row["obtained_at"] or "").strip())
        if observed > obtained or obtained > cutoff:
            raise ValueError("Future/unobtained lineup information")
        role = (row["role"] or "").strip().lower()
        if role not in {"starter", "reserve", "out", "uncertain"}:
            raise ValueError("Unrecognized lineup availability")
        try:
            rank = int(row["depth_rank"])
        except (TypeError, ValueError) as error:
            raise ValueError("Missing/invalid explicit depth rank") from error
        if rank < 1 or str(rank) != (row["depth_rank"] or "").strip():
            raise ValueError("Depth rank must be a positive whole number")
        rank_key = (team, rated["position"], rank)
        if rank_key in ranks:
            raise ValueError("Ambiguous depth ranking for same position")
        ranks.add(rank_key)
        by_team_unit[(team, rated["_unit"])].append(
            {
                "player": rated,
                "role": role,
                "rank": rank,
                # Do not treat an old starter report as a current confirmation.
                "fresh": (cutoff - observed).total_seconds() <= 72 * 3600,
            }
        )

    result = {}
    for (team, unit), players in sorted(by_team_unit.items()):
        starters = [p for p in players if p["role"] == "starter"]
        reserves = [p for p in players if p["role"] == "reserve"]
        absent = [p for p in players if p["role"] == "out"]
        uncertain = [p for p in players if p["role"] == "uncertain"]
        row_result = {
            "starter_count": len(starters),
            "reserve_count": len(reserves),
            "out_count": len(absent),
            "uncertain_count": len(uncertain),
            "starter_mean": (
                round(sum(p["player"]["_talent"] for p in starters) / len(starters), 3)
                if starters
                else None
            ),
            "reserve_mean": (
                round(sum(p["player"]["_talent"] for p in reserves) / len(reserves), 3)
                if reserves
                else None
            ),
            "unavailable_vs_reserve_gap": None,
            "qb_ready": False,
            "qb_replacement_player_id": None,
            "qb_gap_reason": "NOT_QB_OR_UNVERIFIED",
        }
        if unit == "QB":
            expected = [p for p in players if p["rank"] == 1]
            selected = starters[0] if len(starters) == 1 else None
            if len(expected) != 1 or selected is None:
                row_result["qb_gap_reason"] = "UNVERIFIED_EXPECTED_STARTER"
            elif not expected[0]["fresh"] or not selected["fresh"]:
                row_result["qb_gap_reason"] = "STALE_QB_ASSIGNMENT"
            elif expected[0]["role"] == "starter" and selected["rank"] == 1:
                row_result.update(
                    qb_ready=True,
                    unavailable_vs_reserve_gap=0.0,
                    qb_replacement_player_id=selected["player"]["player_id"],
                    qb_gap_reason="CONFIRMED_QB1_ACTIVE",
                )
            elif expected[0]["role"] == "out" and selected["rank"] > 1:
                row_result.update(
                    qb_ready=True,
                    unavailable_vs_reserve_gap=round(
                        expected[0]["player"]["_talent"]
                        - selected["player"]["_talent"],
                        3,
                    ),
                    qb_replacement_player_id=selected["player"]["player_id"],
                    qb_gap_reason="CONFIRMED_QB1_OUT_REPLACEMENT",
                )
            else:
                row_result["qb_gap_reason"] = "QB1_NOT_CONFIRMED_OUT_OR_ACTIVE"
        result[(team, unit)] = row_result
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
