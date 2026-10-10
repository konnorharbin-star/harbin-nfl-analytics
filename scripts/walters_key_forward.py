"""First-published Walters key-number shadow; frozen independently of odds.

This module NEVER changes production scores, betting outputs or portfolio logic.
2025 development observations were previously inspected. 2026 is the first
prospective test. The same fixed alpha=1 shadow is retained even if it loses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.forward_price_validation import first_commit_time, utc
from scripts.grade_recommendations import fetch as fetch_scoreboard
from scripts.grade_recommendations import final_result
from scripts.walters_key_number_distribution import (
    LINES,
    bootstrap,
    distribution,
    fit,
    load,
    metrics,
    outcomes,
)

VERSION = "walters_key_number_forward_v1"
FROZEN_ALPHA = 1.0
SOURCE_YEARS = {"nfl": (2022, 2023, 2024, 2025),
                "cfb": (2023, 2024, 2025)}
GAME_ID = re.compile(r"^[A-Za-z0-9_-]{1,90}$")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raw_forecasts(path, sport):
    required = {"game_id", "season", "week", "home_team", "away_team",
                "model_margin_home", "kickoff" if sport == "nfl" else "date"}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing canonical pregame fair-score fields")
        rows, seen = [], {}
        for row in reader:
            gid = str(row["game_id"]).strip()
            if not GAME_ID.fullmatch(gid):
                raise ValueError("Unsafe game identity")
            if not str(row["home_team"]).strip() or not str(row["away_team"]).strip():
                raise ValueError("Missing team")
            if row["home_team"] == row["away_team"]:
                raise ValueError("Same home/away team")
            try:
                season = int(row["season"])
                week = int(row["week"])
                margin = float(row["model_margin_home"])
                kickoff = utc(row["kickoff"] if sport == "nfl" else row["date"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("Invalid timestamp or model score") from exc
            if not math.isfinite(margin) or week < 1 or week > 30:
                raise ValueError("Invalid forecast margin or week")
            observation = {
                "game_id": gid, "sport": sport, "season": season, "week": week,
                "home_team": row["home_team"], "away_team": row["away_team"],
                "kickoff": kickoff, "projected_margin": margin,
            }
            if gid in seen:
                if seen[gid] != observation:
                    raise ValueError("Conflicting duplicate game forecasts")
                # NFL dashboard publishes one game three times, by market.
                # Identical football forecasts count as one frozen game.
                continue
            seen[gid] = observation
            rows.append(observation)
    return rows


def make_snapshots(board, historical, sport, *, now=None, horizon_days=7):
    """Immutable record payloads, but no writes until caller decides to capture."""
    if sport not in SOURCE_YEARS:
        raise ValueError("Unsupported sport")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Capture clock must be aware")
    now = now.astimezone(UTC)
    # Frozen 2024 tuning selected alpha 1 for BOTH independent football fits.
    past = load(historical, sport)
    years = set(SOURCE_YEARS[sport])
    train = [r for r in past if r["season"] in years]
    observed_years = {r["season"] for r in train}
    if observed_years != years:
        raise ValueError("Full prior season training archive required")
    if any(r["season"] >= 2026 for r in past):
        raise ValueError("Forward season cannot leak into coefficients")
    model = fit(train)
    raw = _raw_forecasts(board, sport)
    model_sha = _digest(historical)
    board_sha = _digest(board)
    result = []
    for row in raw:
        if row["season"] != 2026:
            continue
        if not now + timedelta(minutes=5) < row["kickoff"] <= now + timedelta(
            days=horizon_days
        ):
            continue
        baseline = distribution(model, row["projected_margin"], 0.0)
        challenger = distribution(model, row["projected_margin"], FROZEN_ALPHA)
        line_data = []
        for line in LINES:
            base = outcomes(baseline, home_spread=line)
            candidate = outcomes(challenger, home_spread=line)
            line_data.append({
                "home_spread": line,
                "baseline": list(base),
                "key_number": list(candidate),
            })
        record = {
            "spec": VERSION, "release_state": "SHADOW",
            "game_id": row["game_id"], "sport": sport, "season": 2026,
            "week": row["week"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "kickoff": row["kickoff"].isoformat(),
            "captured_at": now.isoformat(),
            "independent_home_margin": row["projected_margin"],
            "training_seasons": list(SOURCE_YEARS[sport]),
            "historical_source_sha256": model_sha,
            "live_board_sha256": board_sha,
            "model_mean": model["mean"],
            "model_sigma": model["sigma"],
            "key_multipliers": {str(k): v for k, v in model["multipliers"].items()},
            "fixed_alpha": FROZEN_ALPHA,
            "diagnostic_spreads": line_data,
            "sportsbook_odds_used": False,
            "actual_sportsbook_entry_verified": False,
            "wager_authorized": False, "production_model_changed": False,
            "source_limitation": (
                "Board availability is observed at capture time; original"
                " fair-score production time requires separate evidence."
            ),
        }
        result.append(record)
    return result


def capture(records, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fresh, exists = 0, 0
    for row in records:
        target = root / f"{row['game_id']}.json"
        if target.exists():
            # This protects first-snapshot immutability; never silently overwrite.
            old = json.loads(target.read_text(encoding="utf-8"))
            if old["game_id"] != row["game_id"] or old["spec"] != VERSION:
                raise ValueError("Immutable receipt identity conflict")
            exists += 1
            continue
        with target.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n")
        fresh += 1
    return {"newly_frozen": fresh, "already_frozen": exists}


def _threeway_score(actual_margin, record):
    metric = {}
    for name in ("baseline", "key_number"):
        cross_entropy = brier = 0.0
        for item in record["diagnostic_spreads"]:
            win, push, loss = item[name]
            probabilities = (win, push, loss)
            verdict = actual_margin + item["home_spread"]
            idx = 0 if verdict > 0 else 1 if verdict == 0 else 2
            cross_entropy -= math.log(max(1e-12, probabilities[idx]))
            brier += sum((p - (idx == k)) ** 2 for k, p in enumerate(probabilities))
        n = len(record["diagnostic_spreads"])
        metric[name] = {"log_loss": cross_entropy / n, "brier": brier / n}
    return metric


def grade(root, *, now=None, commit_lookup=first_commit_time,
          scoreboard_fetch=fetch_scoreboard):
    root = Path(root)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Grade clock must be aware")
    now = now.astimezone(UTC)
    rows, cache, exclusions = [], {}, {}
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("spec") != VERSION or record.get("release_state") != "SHADOW":
            exclusions[path.name] = "UNRECOGNIZED_OR_UNFROZEN_SPEC"
            continue
        if str(record.get("game_id")) != path.stem:
            exclusions[path.name] = "WRONG_RECORD_ID"
            continue
        try:
            kickoff = utc(record["kickoff"])
            captured = utc(record["captured_at"])
            published = commit_lookup(path)
        except (KeyError, TypeError, ValueError):
            exclusions[path.name] = "INVALID_TIMESTAMP"
            continue
        if published is None or not captured <= published < kickoff:
            exclusions[path.name] = "NO_VERIFIED_PREGAME_PUBLICATION"
            continue
        if record.get("sportsbook_odds_used") is not False:
            exclusions[path.name] = "FOOTBALL_SCORE_CONTAMINATED_BY_MARKET"
            continue
        if kickoff >= now:
            exclusions[path.name] = "PENDING_KICKOFF"
            continue
        date = kickoff.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        key = record["sport"], date
        if key not in cache:
            if len(cache) >= 80:
                exclusions[path.name] = "SCOREBOARD_QUERY_CAP"
                continue
            try:
                cache[key] = scoreboard_fetch(*key)
            except (ValueError, TimeoutError, OSError) as exc:
                cache[key] = None
                exclusions[str(key)] = type(exc).__name__
        if cache[key] is None:
            exclusions[path.name] = "SCOREBOARD_UNAVAILABLE"
            continue
        payload, source_url = cache[key]
        final = final_result(record, payload)
        if final is None:
            exclusions[path.name] = "NOT_CONFIRMED_FINAL"
            continue
        score = _threeway_score(
            final["home_score"] - final["away_score"], record
        )
        rows.append({
            "game_id": record["game_id"], "season": 2026,
            "week": int(record["week"]), "home_team": record["home_team"],
            "away_team": record["away_team"], "kickoff": record["kickoff"],
            "captured_at": record["captured_at"],
            "published_at": published.isoformat(),
            "final_score": final,
            "final_source_url": source_url,
            "final_scoreboard_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "baseline": score["baseline"], "key_number": score["key_number"],
        })
    report = {
        "spec": VERSION, "evaluated_at": now.isoformat(),
        "status": "FORWARD_SHADOW_RESEARCH_ONLY",
        "graded_games": len(rows), "ungraded": exclusions,
        "betting_authorized": False, "production_promotion_authorized": False,
        "book_quote_samples": 0, "roi": None, "clv": None,
        "minimum_graded_games": 128, "minimum_distinct_weeks": 8,
        "scores": {}, "paired": None,
        "limitation": (
            "Football probability scoring on synthetic fixed spreads only."
            " No independently verified executable book prices or real CLV."
        ),
        "graded_rows": rows,
    }
    if rows:
        base = [
            {"game_id": r["game_id"], "season": 2026, "week": r["week"],
             **r["baseline"]}
            for r in rows
        ]
        cand = [
            {"game_id": r["game_id"], "season": 2026, "week": r["week"],
             **r["key_number"]}
            for r in rows
        ]
        weeks = len({r["week"] for r in rows})
        report["scores"] = {"baseline": metrics(base), "key_number": metrics(cand)}
        if len(rows) >= 128 and weeks >= 8:
            report["paired"] = bootstrap(base, cand)
            report["status"] = "FORWARD_STATISTICAL_RESEARCH"
        else:
            report["status"] = "FORWARD_INSUFFICIENT_SAMPLE"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=tuple(SOURCE_YEARS), required=True)
    parser.add_argument("--board", type=Path, default=Path("docs/latest.csv"))
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path("history/walters_key_forward_v1/forecasts")
    )
    parser.add_argument("--out", type=Path, default=Path("reports/walters_key_forward.json"))
    args = parser.parse_args()
    now = datetime.now(UTC)
    snapshots = make_snapshots(args.board, args.historical, args.sport, now=now)
    new = capture(snapshots, args.root)
    report = grade(args.root, now=now)
    report["capture_summary"] = new
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "sport": args.sport, **new,
        "graded_games": report["graded_games"], "status": report["status"],
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
