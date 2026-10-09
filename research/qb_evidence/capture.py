"""Archive current QB evidence with explicit unverified/confirmed states.

Research only: no ratings or betting-point conversion. GitHub run artifacts
provide a first-observed capture time; model outputs alone never prove starter.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path


def stamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamp must have explicit UTC offset")
    return result.astimezone(UTC)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No records: {path}")
    return rows


def choose_input(pattern: str) -> Path:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(pattern)
    if len(paths) == 1:
        return paths[0]

    def version(path: Path) -> tuple[int, int]:
        match = re.search(r"cfb_model_(\d{4})_week(\d+)\.csv$", path.name)
        if not match:
            raise ValueError("Ambiguous prediction glob: " + pattern)
        return int(match.group(1)), int(match.group(2))

    return sorted(paths, key=version)[-1]


def _optional_source(path: Path | None, *, capture_at: datetime) -> dict:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "game_id", "team", "model_player_id", "role", "depth_rank",
            "observed_at", "obtained_at", "source_url", "verified",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Independent starter source lacks required columns")
        rows = list(reader)
    roster = defaultdict(list)
    used = set()
    for row in rows:
        game, team, pid = (
            str(row[field] or "").strip()
            for field in ("game_id", "team", "model_player_id")
        )
        role = str(row["role"] or "").strip().lower()
        if not game or not team or not pid:
            raise ValueError("Blank independent evidence identity")
        if role not in {"starter", "reserve", "out", "uncertain"}:
            raise ValueError("Unsupported source role")
        if str(row["verified"] or "").strip().lower() != "true":
            raise ValueError("Unreviewed record cannot be starter evidence")
        if not str(row["source_url"] or "").startswith("https://"):
            raise ValueError("Independent record requires original HTTPS source")
        observed, obtained = stamp(row["observed_at"]), stamp(row["obtained_at"])
        if observed > obtained or obtained > capture_at:
            raise ValueError("Source observation/acquisition after cutoff")
        try:
            depth = int(row["depth_rank"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Missing source QB depth rank") from exc
        if depth < 1 or str(depth) != str(row["depth_rank"]).strip():
            raise ValueError("Invalid positive depth rank")
        key = (game, team, pid)
        if key in used:
            raise ValueError("Duplicate independent player identity")
        used.add(key)
        roster[(game, team)].append({
            "model_player_id": pid,
            "role": role,
            "depth_rank": depth,
            "observed_at": observed,
            "obtained_at": obtained,
            "source_url": row["source_url"],
        })
    return roster


def _verification(records: list[dict], expected_id: str, capture_at: datetime) -> dict:
    result = {
        "independent_starter_id": "",
        "independent_backup_id": "",
        "independent_qb1_out_id": "",
        "independent_starter_source": "",
        "independent_evidence_status": "NO_INDEPENDENT_LINEUP",
        "independent_source_fresh": False,
        "ea_player_gap_rating_units": None,
        "eligible_for_talent_backtest": False,
    }
    if not records:
        return result
    if any(
        capture_at - row["observed_at"] > timedelta(hours=72)
        for row in records
    ):
        result["independent_evidence_status"] = "STALE_LINEUP"
        return result
    ranks = [row["depth_rank"] for row in records]
    if len(ranks) != len(set(ranks)):
        result["independent_evidence_status"] = "AMBIGUOUS_DEPTH"
        return result
    starters = [row for row in records if row["role"] == "starter"]
    if len(starters) != 1:
        result["independent_evidence_status"] = "NO_UNIQUE_CONFIRMED_STARTER"
        return result
    started = starters[0]
    result["independent_starter_id"] = started["model_player_id"]
    result["independent_starter_source"] = started["source_url"]
    result["independent_source_fresh"] = True
    backup = next(
        (row for row in records if row["depth_rank"] == 2 and row["role"] == "reserve"),
        None,
    )
    if backup:
        result["independent_backup_id"] = backup["model_player_id"]
    qb_one_out = next(
        (row for row in records if row["depth_rank"] == 1 and row["role"] == "out"),
        None,
    )
    if qb_one_out:
        result["independent_qb1_out_id"] = qb_one_out["model_player_id"]
    if not expected_id:
        result["independent_evidence_status"] = "NO_MODEL_QB_ID"
    elif started["model_player_id"] != expected_id:
        result["independent_evidence_status"] = "STARTER_CONFLICT_WITH_MODEL"
    elif qb_one_out and started["depth_rank"] > 1:
        result["independent_evidence_status"] = "QB1_OUT_NAMED_REPLACEMENT"
    elif started["depth_rank"] == 1 and backup:
        result["independent_evidence_status"] = "STARTER_AND_BACKUP_CONFIRMED"
    else:
        result["independent_evidence_status"] = "STARTER_CONFIRMED_BACKUP_UNKNOWN"
    # External EA player mapping/ratings remain absent; never compute a gap.
    return result


def build(
    model_csv: Path,
    *,
    sport: str,
    capture_at: str,
    verified_lineups: Path | None = None,
    max_days_ahead: int = 14,
) -> tuple[list[dict], dict]:
    capture = stamp(capture_at)
    if max_days_ahead < 1:
        raise ValueError("Invalid forward horizon")
    data = _read(model_csv)
    required = {"game_id", "home_team", "away_team", "season"}
    required.add("kickoff" if sport == "nfl" else "date")
    if not required.issubset(data[0]):
        raise ValueError("Model output lacks required game identity fields")
    source = _optional_source(verified_lineups, capture_at=capture)
    evidence = []
    excluded = Counter()
    # Model boards may repeat a game for spread/total/moneyline markets.
    # A repeated game is safe to collapse only if its QB identities agree.
    games = {}
    identity_fields = ["game_id", "season", "home_team", "away_team"]
    identity_fields += (
        ["kickoff", "home_expected_qb_id", "away_expected_qb_id"]
        if sport == "nfl" else ["date"]
    )
    for game in data:
        game_id = str(game["game_id"]).strip()
        if not game_id:
            raise ValueError("Blank model game ID")
        previous = games.get(game_id)
        if previous is not None:
            if any(previous.get(k, "") != game.get(k, "") for k in identity_fields):
                raise ValueError("Conflicting duplicate game identity/QB")
            excluded["duplicate_market_rows"] += 1
            continue
        games[game_id] = game

    for game in games.values():
        game_id = str(game["game_id"]).strip()
        kickoff = stamp(game["kickoff" if sport == "nfl" else "date"])
        if kickoff <= capture:
            excluded["already_kicked_off"] += 1
            continue
        if kickoff > capture + timedelta(days=max_days_ahead):
            excluded["beyond_forward_horizon"] += 1
            continue
        if int(game["season"]) != kickoff.year:
            raise ValueError("Season/kickoff mismatch")
        for side in ("home", "away"):
            team = str(game[f"{side}_team"]).strip()
            expected = (
                str(game.get(f"{side}_expected_qb_id") or "").strip()
                if sport == "nfl" else ""
            )
            details = _verification(
                source.get((game_id, team), []), expected, capture
            )
            item = {
                "capture_at": capture.isoformat(),
                "game_id": game_id,
                "season": game["season"],
                "kickoff": kickoff.isoformat(),
                "team": team,
                "side": side,
                "model_expected_qb_id": expected,
                "model_expected_qb_name": (
                    game.get(f"{side}_expected_qb_name", "") if sport == "nfl" else ""
                ),
                "model_qb_source": (
                    game.get(f"{side}_expected_qb_source", "") if sport == "nfl" else ""
                ),
                "model_qb_decision_ready": (
                    game.get(f"{side}_expected_qb_decision_ready", "")
                    if sport == "nfl" else ""
                ),
                "model_injury_freshness": game.get(
                    f"{side}_injury_freshness_status",
                    game.get(f"{side}_qb_injury_evidence_status", ""),
                ),
                "model_depth_freshness": game.get(
                    f"{side}_depth_freshness_status", ""
                ),
                "model_starter_verified": (
                    str(game.get(f"{side}_qb_starter_verified", "")).lower() == "true"
                    if sport == "cfb" else False
                ),
                **details,
            }
            evidence.append(item)
    source_hash = hashlib.sha256(model_csv.read_bytes()).hexdigest()
    summary = {
        "spec": "pregame_qb_evidence_capture_v1",
        "status": "RESEARCH_ONLY_NO_TALENT_BACKTEST",
        "sport": sport,
        "captured_at": capture.isoformat(),
        "model_source_path": str(model_csv),
        "model_source_sha256": source_hash,
        "model_game_rows": len(data),
        "unique_game_rows": len(games),
        "upcoming_games": len(evidence) // 2,
        "team_evidence_rows": len(evidence),
        "source_records_provided": bool(source),
        "evidence_states": dict(Counter(
            row["independent_evidence_status"] for row in evidence
        )),
        "excluded": dict(excluded),
        "validated_replacement_ratings_count": 0,
        "live_model_score_adjustment": 0,
        "production_wager_adjustment": 0,
        "limitations": [
            "Model expected-QB output is not independent starter proof.",
            "CFB output lacks named expected starter IDs.",
            "Independent source records are optional and currently not provided.",
            "No stable crosswalked EA individual player dataset is available.",
            "Model output may predate capture; original source time is unverified.",
            "Artifacts expire; retain hashes and exports for permanent research archiving.",
        ],
    }
    return evidence, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", required=True, choices=["nfl", "cfb"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--capture-at", required=True)
    parser.add_argument("--verified-lineups", type=Path)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    file = choose_input(args.predictions)
    evidence, summary = build(
        file, sport=args.sport, capture_at=args.capture_at,
        verified_lineups=args.verified_lineups,
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as fh:
        if evidence:
            writer = csv.DictWriter(fh, fieldnames=list(evidence[0]))
            writer.writeheader()
            writer.writerows(evidence)
        else:
            fh.write("game_id,team,independent_evidence_status\n")
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
