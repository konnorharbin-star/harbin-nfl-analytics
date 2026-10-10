"""Prospective dated personnel evidence, kept outside score fits and release gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.grade_recommendations import team_matches
from scripts.market_first_experiment import immutable
from scripts.recommendation_ledger import canonical, timestamp

ROOT = Path("history/personnel_evidence_v1")
MAX_REPORT_AGE = timedelta(hours=72)
ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/football/{}/injuries"


def injury_records(payload):
    """Retain exact structured source fields; item date is not an official report time."""
    result = []
    if not isinstance(payload, dict):
        raise TypeError("Personnel payload must be an object")
    groups = payload.get("injuries")
    if not isinstance(groups, list):
        raise TypeError("Missing injury groups")
    for group in groups:
        for item in group.get("injuries", []):
            athlete = item.get("athlete") or {}
            team = athlete.get("team") or {}
            ids = set()
            for link in athlete.get("links", []):
                match = re.search(r"/player/(?:[^/]+/)*id/(\d+)", str(link.get("href", "")))
                if match:
                    ids.add(match.group(1))
            direct = athlete.get("id")
            if direct:
                ids.add(str(direct))
            result.append(
                {
                    "source_item_id": str(item.get("id") or ""),
                    "player_id": next(iter(ids)) if len(ids) == 1 else None,
                    "player_name": athlete.get("displayName"),
                    "position": (athlete.get("position") or {}).get("abbreviation"),
                    "source_status": item.get("status"),
                    "source_item_date": item.get("date"),
                    "group_team_id": str(group.get("id") or ""),
                    "team": {
                        k: team.get(k)
                        for k in (
                            "id",
                            "abbreviation",
                            "location",
                            "displayName",
                            "shortDisplayName",
                        )
                    },
                    "source": item.get("source"),
                }
            )
    return result


def evidence_state(record, *, observed, kickoff, source_season, game_season):
    if source_season != game_season:
        return "WRONG_SOURCE_SEASON"
    if str(record["team"].get("id") or "") != record["group_team_id"]:
        return "TEAM_IDENTITY_MISMATCH"
    if not record.get("player_id") or not record.get("source_status"):
        return "MISSING_PLAYER_ID_OR_STATUS"
    try:
        dated = timestamp(record["source_item_date"])
    except (ValueError, TypeError):
        return "MISSING_AWARE_ITEM_DATE"
    if dated > observed or dated >= kickoff:
        return "FUTURE_OR_POSTKICKOFF_ITEM_DATE"
    if observed >= kickoff:
        return "COLLECTED_AFTER_KICKOFF"
    if observed - dated > MAX_REPORT_AGE:
        return "STALE_ITEM_DATE"
    return "FRESH_DATED_SECONDARY_STATUS"


def evaluate(records, games, *, observed, source_season):
    output, counts = [], Counter()
    seen = set()
    for game in games:
        gid = game["game_id"]
        if gid in seen:
            continue
        seen.add(gid)
        kickoff = timestamp(game.get("kickoff") or game.get("date"))
        if kickoff <= observed:
            counts["games_already_started"] += 1
            continue
        for side in ("home", "away"):
            matches = [
                r for r in records if team_matches(game[side + "_team"], r["team"], game["sport"])
            ]
            states = [
                evidence_state(
                    r,
                    observed=observed,
                    kickoff=kickoff,
                    source_season=source_season,
                    game_season=int(game["season"]),
                )
                for r in matches
            ]
            counts.update(states)
            fresh = [
                r
                for r, s in zip(matches, states, strict=True)
                if s == "FRESH_DATED_SECONDARY_STATUS"
            ]
            qb = [r for r in fresh if r["position"] == "QB"]
            output.append(
                {
                    "game_id": gid,
                    "kickoff": kickoff.isoformat(),
                    "side": side,
                    "team": game[side + "_team"],
                    "source_record_count": len(matches),
                    "fresh_source_record_count": len(fresh),
                    "fresh_qb_status_records": qb,
                    "fresh_source_item_ids": [r["source_item_id"] for r in fresh],
                    "record_states": dict(Counter(states)),
                    "status": "DATED_SECONDARY_EVIDENCE"
                    if fresh
                    else "NO_FRESH_PERSONNEL_EVIDENCE",
                    "official_team_report_verified": False,
                    "starter_verified": False,
                    "healthy_roster_inferred": False,
                    "score_adjustment_enabled": False,
                }
            )
    return output, dict(counts)


def fetch(sport):
    league = "nfl" if sport == "nfl" else "college-football"
    url = ENDPOINT.format(league)
    started = datetime.now(UTC)
    with urlopen(
        Request(url, headers={"User-Agent": "HarbinPersonnelResearch/1.0"}), timeout=20
    ) as response:
        body = response.read(12_000_001)
    ended = datetime.now(UTC)
    if len(body) > 12_000_000:
        raise ValueError("Oversized personnel response")
    payload = json.loads(body)
    if payload.get("status") != "success":
        raise ValueError("Personnel source did not report success")
    return payload, {
        "url": url,
        "request_started_at": started.isoformat(),
        "observed_at": ended.isoformat(),
        "raw_response_sha256": hashlib.sha256(body).hexdigest(),
        "response_generated_at": payload.get("timestamp"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=Path("docs/personnel_evidence.json"))
    args = parser.parse_args()
    now = datetime.now(UTC)
    error = None
    records = []
    metadata = {
        "observed_at": now.isoformat(),
        "url": ENDPOINT.format("nfl" if args.sport == "nfl" else "college-football"),
    }
    source_season = None
    try:
        payload, metadata = fetch(args.sport)
        now = timestamp(metadata["observed_at"])
        records = injury_records(payload)
        source_season = payload.get("season", {}).get("year")
    except (OSError, ValueError, TypeError, TimeoutError) as exc:
        error = str(exc)
    with Path("docs/latest.csv").open() as source:
        games = list(csv.DictReader(source))
    for game in games:
        game["sport"] = args.sport
    evaluated, counts = evaluate(records, games, observed=now, source_season=source_season)
    data = {"schema": 1, "sport": args.sport, "records": records}
    digest = hashlib.sha256(canonical(data).encode()).hexdigest()
    immutable(args.root / "payloads" / (digest + ".json"), data)
    capture = {
        "schema": 1,
        "sport": args.sport,
        "source": metadata,
        "source_season": source_season,
        "payload_sha256": digest,
        "error": error,
        "game_evidence": evaluated,
        "record_states": counts,
        "model_score_change_enabled": False,
        "betting_authorized": False,
    }
    rid = hashlib.sha256(canonical(capture).encode()).hexdigest()[:24]
    immutable(args.root / "captures" / (rid + ".json"), capture)
    report = {
        "sport": args.sport,
        "observed_at": now.isoformat(),
        "capture_id": rid,
        "status": "SOURCE_ERROR"
        if error
        else (
            "DATED_SECONDARY_RESEARCH"
            if any(r["fresh_source_record_count"] for r in evaluated)
            else "NO_FRESH_PERSONNEL_EVIDENCE"
        ),
        "error": error,
        "source_url": metadata["url"],
        "source_season": source_season,
        "response_generated_at": metadata.get("response_generated_at"),
        "source_records": len(records),
        "source_record_date_states": dict(
            Counter(
                evidence_state(
                    r,
                    observed=now,
                    kickoff=now + timedelta(days=366),
                    source_season=source_season,
                    game_season=source_season,
                )
                for r in records
            )
        ),
        "upcoming_team_games": len(evaluated),
        "team_games_with_fresh_records": sum(
            bool(r["fresh_source_record_count"]) for r in evaluated
        ),
        "team_games_with_fresh_qb_status": sum(
            bool(r["fresh_qb_status_records"]) for r in evaluated
        ),
        "record_states": counts,
        "starter_verified": False,
        "official_report_verified": False,
        "betting_authorized": False,
        "model_score_change_enabled": False,
        "limitations": [
            "Source response timestamp does not establish item freshness",
            "No injury record does not mean a healthy roster",
            "Dated secondary statuses do not verify starters",
            "Current-only evidence is never backfilled into historical models",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
