"""Durable first-seen coupled NFL research, with no betting authority.

Archive original JSON bytes, select the first eligible snapshot per game, and
append source outcome revisions without overwriting forecasts or prior grades.
This new cohort does not replace earlier Project/Drive forecast cohorts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import re

from nfl.coupled_forward_grading import clock, grade_snapshot

SPEC = "NFL_COUPLED_DURABLE_PRIMARY_V1"
METRICS = ("canonical_total_mae", "shadow_total_mae", "canonical_margin_mae",
           "shadow_margin_mae", "total_mae_delta_baseline_minus_shadow",
           "margin_mae_delta_baseline_minus_shadow", "shadow_three_way_brier")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def read_journal(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        original = dict(row)
        expected = original.pop("receipt_sha256", None)
        if digest(encoded(original)) != expected:
            raise ValueError("JOURNAL_RECEIPT_HASH_MISMATCH")
    return rows


def append(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as target:
        for row in rows:
            row = dict(row)
            row["receipt_sha256"] = digest(encoded(row))
            target.write(encoded(row))


def _archive_file(root, data):
    sha = digest(data)
    path = root / "snapshots" / (sha + ".json")
    if path.exists() and path.read_bytes() != data:
        raise ValueError("ARCHIVE_CONTENT_CONFLICT")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return sha


def archive(root, snapshot_path, replay_path, *, at, run_id, code_sha, sidecars=(),
            model_sha="unverified"):
    root = Path(root)
    if at.tzinfo is None:
        raise ValueError("UNZONED_ARCHIVE_CLOCK")
    if not re.fullmatch(r"[0-9]+", str(run_id)):
        raise ValueError("INVALID_SOURCE_RUN_ID")
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("INVALID_CODE_SHA")
    scan_bytes = Path(snapshot_path).read_bytes()
    replay_bytes = Path(replay_path).read_bytes()
    scan, replay = json.loads(scan_bytes), json.loads(replay_bytes)
    if (scan.get("spec") != "NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1"
        or replay.get("spec") != "NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1"
        or scan.get("capture_utc") != replay.get("capture_utc")):
        raise ValueError("WRONG_OR_MISMATCHED_SOURCE_FAMILY")
    captured = clock(scan["capture_utc"])
    if captured > at:
        raise ValueError("FUTURE_CAPTURE")
    year = int(scan["season"])
    if year != captured.year:
        raise ValueError("WRONG_SEASON")
    originals = read_journal(root / "primary.jsonl")
    seen = {}
    for row in originals:
        if row["game_id"] in seen:
            raise ValueError("DUPLICATE_PRIMARY_GAME")
        seen[row["game_id"]] = row
    if originals and {r["model_implementation_sha256"] for r in originals} != {model_sha}:
        raise ValueError("MODEL_CHANGED_NEW_COHORT_REQUIRED")
    replay_map = {}
    for item in replay.get("games", []):
        gid = item.get("game_id")
        if not gid or gid in replay_map:
            raise ValueError("DUPLICATE_REPLAY_GAME")
        replay_map[gid] = item
    new, blocked, repeated = [], [], 0
    scan_seen = set()
    for game in scan.get("games", []):
        gid = game.get("game_id")
        if not isinstance(gid, str) or not re.fullmatch(r"[A-Z0-9_]+", gid) or gid in scan_seen:
            raise ValueError("DUPLICATE_OR_INVALID_SOURCE_GAME")
        scan_seen.add(gid)
        if gid in seen:
            repeated += 1
            continue
        try:
            if game.get("status") != "UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH":
                raise ValueError(game.get("reason", "SOURCE_OR_MODEL_BLOCKED"))
            original = game["frozen_football_score_input"]
            first = clock(original["captured_utc"])
            kickoff = clock(original["kickoff_utc"])
            if not first <= captured <= at < kickoff:
                raise ValueError("POSTKICKOFF_OR_INVALID_CHRONOLOGY")
            if original.get("event_id") != gid or not original.get("content_sha256"):
                raise ValueError("INVALID_ORIGINAL_IDENTITY")
            if replay_map.get(gid, {}).get("forecast") != original:
                raise ValueError("REPLAY_NOT_IDENTICAL_TO_ORIGINAL")
            new.append({"spec": SPEC, "game_id": gid, "season": year,
                        "kickoff_utc": kickoff.isoformat(), "capture_utc": scan["capture_utc"],
                        "archived_utc": at.astimezone(UTC).isoformat(),
                        "snapshot_sha256": digest(scan_bytes),
                        "replay_sha256": digest(replay_bytes),
                        "original_forecast_sha256": original["content_sha256"],
                        "source_run_id": str(run_id), "source_code_sha": code_sha,
                        "model_implementation_sha256": model_sha,
                        "decision": "PASS_RESEARCH_ONLY", "betting_authorized": False})
        except (KeyError, ValueError, TypeError) as error:
            blocked.append({"game_id": gid, "reason": str(error)})
    sidecar_shas = []
    missing_sidecars = [str(path) for path in sidecars if not Path(path).exists()]
    if new:
        _archive_file(root, scan_bytes)
        _archive_file(root, replay_bytes)
        for path in sidecars:
            if Path(path).exists():
                data = Path(path).read_bytes()
                json.loads(data)  # Fail if an alleged JSON sidecar is corrupt.
                sidecar_shas.append(_archive_file(root, data))
        for row in new:
            row["supplemental_sidecar_sha256"] = sidecar_shas
        append(root / "primary.jsonl", new)
    report = {"spec": SPEC, "archived_utc": at.astimezone(UTC).isoformat(),
              "source_run_id": str(run_id), "new_primary_games": len(new),
              "repeated_games_not_recounted": repeated, "blocked": blocked,
              "independent_primary_games": len(originals) + len(new),
              "missing_supplemental_sources": missing_sidecars,
              "betting_authorized": False, "action": "PASS_RESEARCH_ONLY"}
    append(root / "capture_receipts.jsonl", [report])
    return report


def load_verified_snapshot(root, receipt):
    path = Path(root) / "snapshots" / (receipt["snapshot_sha256"] + ".json")
    data = path.read_bytes()
    if digest(data) != receipt["snapshot_sha256"]:
        raise ValueError("ARCHIVE_SNAPSHOT_HASH_MISMATCH")
    snapshot = json.loads(data)
    matches = [g for g in snapshot["games"] if g["game_id"] == receipt["game_id"]]
    if len(matches) != 1:
        raise ValueError("PRIMARY_GAME_NOT_UNIQUE_IN_SOURCE")
    if (snapshot["capture_utc"] != receipt["capture_utc"]
        or matches[0]["frozen_football_score_input"]["content_sha256"]
        != receipt["original_forecast_sha256"]):
        raise ValueError("PRIMARY_RECEIPT_SOURCE_MISMATCH")
    snapshot["games"] = matches
    return snapshot


def summarize(rows):
    n = len(rows)
    summary = {"distinct_primary_games_graded": n,
               "remaining_to_128_game_research_review": max(0, 128-n),
               "metrics": {key: sum(r[key] for r in rows)/n for key in METRICS} if n else None,
               "paired_week_cluster_intervals": None,
               "qualified_player_stat_rows": sum(len(r["qualified_player_stat_rows"]) for r in rows),
               "promotion_ready": False, "profitable_edge_validated": False,
               "official_gamebook_verified": False, "exact_market_prices_verified": False,
               "action": "PASS_RESEARCH_ONLY"}
    if n >= 128:
        groups = defaultdict(list)
        for row in rows:
            # Grade against original event's NFL season/week, never draw count.
            groups[tuple(row["game_id"].split("_")[:2])].append(row)
        weeks = list(groups.values())
        if len(weeks) >= 8:
            rng = random.Random(20261014)
            samples = {key: [] for key in METRICS if "delta" in key}
            for _ in range(2000):
                draw = [r for _ in weeks for r in weeks[rng.randrange(len(weeks))]]
                for key in samples:
                    samples[key].append(sum(r[key] for r in draw)/len(draw))
            # Bonferroni over two registered point-error comparisons; descriptive
            # research only, not a score-distribution or sportsbook profit test.
            summary["paired_week_cluster_intervals"] = {
                key: [sorted(v)[25], sorted(v)[1974]] for key, v in samples.items()}
            summary["week_clusters"] = len(weeks)
            summary["interval_level"] = 0.975
    return summary


def grade_archive(root, finals, players, *, at):
    root = Path(root)
    originals = read_journal(root / "primary.jsonl")
    if len({r["game_id"] for r in originals}) != len(originals):
        raise ValueError("DUPLICATE_PRIMARY_GAME")
    history = read_journal(root / "grades.jsonl")
    identities = {r["game_id"]: r for r in originals}
    latest = {}
    for row in history:
        identity = identities.get(row["game_id"])
        previous = latest.get(row["game_id"])
        if (identity is None or row["snapshot_sha256"] != identity["snapshot_sha256"]
            or row["grade"]["model_original_sha256"] != identity["original_forecast_sha256"]
            or row["previous_grade_receipt_sha256"]
            != (previous["receipt_sha256"] if previous else None)):
            raise ValueError("GRADE_HISTORY_IDENTITY_OR_CHAIN_MISMATCH")
        latest[row["game_id"]] = row
    added, coverage = [], defaultdict(int)
    for receipt in originals:
        scan = load_verified_snapshot(root, receipt)
        result = grade_snapshot(scan, finals, players, at=at)
        for key, count in result["counts"].items():
            coverage[key] += count
        for row in result["games"]:
            comparable = {k: v for k, v in row.items() if k != "graded_asof_utc"}
            grade_sha = digest(encoded(comparable))
            old = latest.get(row["game_id"])
            if old and old["outcome_revision_sha256"] == grade_sha:
                continue
            event = {"spec": SPEC, "game_id": row["game_id"],
                     "snapshot_sha256": receipt["snapshot_sha256"],
                     "outcome_revision_sha256": grade_sha, "grade": row,
                     "previous_grade_receipt_sha256": old.get("receipt_sha256") if old else None,
                     "source_run_id": receipt["source_run_id"], "betting_authorized": False}
            added.append(event)
            latest[row["game_id"]] = event
    append(root / "grades.jsonl", added)
    summary = summarize([r["grade"] for r in latest.values()])
    summary.update({"spec": SPEC, "asof_utc": at.astimezone(UTC).isoformat(),
                    "primary_games_registered": len(originals), "source_coverage": dict(coverage),
                    "new_or_corrected_grade_events": len(added),
                    "missing_player_rows_are_not_zero_or_void": True})
    return summary
