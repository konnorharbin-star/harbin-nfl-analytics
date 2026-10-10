"""Grade frozen paper recommendations from free public final scoreboard data.

Source is independent of the prediction ledger but not an official league feed.
Missing, conflicting, rescheduled or ambiguous games remain ungraded. No wagers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from scripts.recommendation_ledger import canonical, publication_is_pregame, timestamp

ENDPOINT = "https://site.api.espn.com/apis/site/v2/sports/football/{}/scoreboard?dates={}&limit=500"
ALIASES = {"JAC": "JAX", "LA": "LAR", "WSH": "WAS"}


def team_matches(label, team, sport):
    if sport == "nfl":
        left = str(label).upper()
        right = str(team.get("abbreviation", "")).upper()
        return ALIASES.get(left, left) == ALIASES.get(right, right)

    def normalize(x):
        return re.sub(r"[^a-z0-9]", "", str(x).casefold())

    return bool(label) and normalize(label) in {
        normalize(team.get(k, "")) for k in ("location", "displayName", "shortDisplayName")
    }


def final_result(row, payload):
    matches = []
    for event in payload.get("events", []):
        if row["sport"] == "cfb" and str(event.get("id")) != str(row["game_id"]):
            continue
        status = event.get("status", {}).get("type", {})
        if status.get("completed") is not True or not str(status.get("name", "")).startswith(
            "STATUS_FINAL"
        ):
            continue
        competitions = event.get("competitions", [])
        if len(competitions) != 1:
            continue
        competitors = competitions[0].get("competitors", [])
        if len(competitors) != 2:
            continue
        teams = {x.get("homeAway"): x for x in competitors}
        if set(teams) != {"home", "away"}:
            continue
        if not all(
            team_matches(row[f"{side}_team"], teams[side].get("team", {}), row["sport"])
            for side in ("home", "away")
        ):
            continue
        try:
            if abs((timestamp(event["date"]) - timestamp(row["kickoff"])).total_seconds()) > 300:
                continue
            scores = [float(teams[side]["score"]) for side in ("home", "away")]
            if not all(math.isfinite(n) and n.is_integer() and 0 <= n <= 120 for n in scores):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        matches.append(
            {
                "home_score": int(scores[0]),
                "away_score": int(scores[1]),
                "event_id": str(event["id"]),
            }
        )
    return matches[0] if len(matches) == 1 else None


def grade(receipt, final):
    row = receipt["original_board_row"]
    if receipt["decision"] != "BET" or timestamp(receipt["recommended_at"]) >= timestamp(
        row["kickoff"]
    ):
        raise ValueError("Not a prospective paper recommendation")
    home, away = final["home_score"], final["away_score"]
    side, market = row["side"], row["market"]
    if market == "total":
        if side not in {"over", "under"}:
            return None
        difference = (home + away - float(row["line"])) * (1 if side == "over" else -1)
    else:
        if side in {"home", row["home_team"]}:
            margin = home - away
        elif side in {"away", row["away_team"]}:
            margin = away - home
        else:
            return None
        if market == "moneyline" and margin == 0:
            return None  # Settlement rules vary: manual review required.
        if market not in {"moneyline", "spread"}:
            return None
        difference = margin + (float(row["line"]) if market == "spread" else 0)
    odds = float(row["american_odds"])
    payout = odds / 100 if odds > 0 else 100 / -odds
    result = "WIN" if difference > 0 else "LOSS" if difference < 0 else "PUSH"
    return {
        "bet_id": receipt["bet_id"],
        "result": result,
        "profit_units": payout if result == "WIN" else -1 if result == "LOSS" else 0,
        "final_score": final,
        "stake_units": 1,
        "closing_price": None,
        "clv": None,
        "clv_status": "MISSING_VERIFIED_CLOSING_QUOTE",
        "actual_wager": False,
    }


def fetch(sport, date):
    league = "nfl" if sport == "nfl" else "college-football"
    url = ENDPOINT.format(league, date) + ("&groups=80" if sport == "cfb" else "")
    with urlopen(
        Request(url, headers={"User-Agent": "HarbinPaperResearch/1.0"}), timeout=15
    ) as response:
        data = response.read(3_000_001)
    if len(data) > 3_000_000:
        raise ValueError("Oversized scoreboard")
    return json.loads(data), url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("history/recommendations_v1"))
    parser.add_argument("--out", type=Path, default=Path("docs/recommendation_performance.json"))
    args = parser.parse_args()
    receipts = [json.loads(p.read_text()) for p in sorted((args.root / "bets").glob("*.json"))]
    grades_dir = args.root / "grades"
    grades_dir.mkdir(parents=True, exist_ok=True)
    cache, errors = {}, {}
    now = datetime.now(UTC)
    for receipt in receipts:
        target = grades_dir / (receipt["bet_id"] + ".json")
        if target.exists():
            continue
        row = receipt["original_board_row"]
        kickoff = timestamp(row["kickoff"])
        if kickoff >= now:
            continue
        receipt_path = args.root / "bets" / (receipt["bet_id"] + ".json")
        publication = (
            subprocess.run(
                ["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(receipt_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
        if len(publication) != 1:
            errors[receipt["bet_id"]] = "Recommendation publication commit unavailable"
            continue
        commit_sha, committed_at = publication[0].split(" ", 1)
        if not publication_is_pregame(receipt["recommended_at"], committed_at, row["kickoff"]):
            errors[receipt["bet_id"]] = "Recommendation not committed before kickoff"
            continue
        key = row["sport"], kickoff.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        if key not in cache:
            if len(cache) >= 30:
                errors[str(key)] = "Daily request cap; retry next scheduled run"
                continue
            try:
                cache[key] = fetch(*key)
            except (OSError, ValueError, TimeoutError) as exc:
                cache[key] = None
                errors[str(key)] = type(exc).__name__
        if cache[key] is None:
            continue
        payload, url = cache[key]
        final = final_result(row, payload)
        if final is None:
            continue
        record = grade(receipt, final)
        if record is None:
            continue
        record.update(
            {
                "graded_at": now.isoformat(),
                "source_url": url,
                "recommendation_commit": commit_sha,
                "source_sha256": hashlib.sha256(canonical(payload).encode()).hexdigest(),
                "score_source": "ESPN public final scoreboard, undocumented API",
            }
        )
        # Grade files are immutable; corrections require a separately reviewed event.
        with tempfile.NamedTemporaryFile(mode="w", dir=grades_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical(record))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    grades = {p.stem: json.loads(p.read_text()) for p in grades_dir.glob("*.json")}
    ordered = sorted(receipts, key=lambda r: (r["original_board_row"]["kickoff"], r["bet_id"]))
    settled = [grades[r["bet_id"]] for r in ordered if r["bet_id"] in grades]
    units, peak, drawdown = 0, 0, 0
    for record in settled:
        units += record["profit_units"]
        peak = max(peak, units)
        drawdown = max(drawdown, peak - units)
    report = {
        "recommended_paper_bets": len(receipts),
        "graded_bets": len(settled),
        "wins": sum(r["result"] == "WIN" for r in settled),
        "losses": sum(r["result"] == "LOSS" for r in settled),
        "pushes": sum(r["result"] == "PUSH" for r in settled),
        "units": units,
        "roi": units / len(settled) if settled else None,
        "max_drawdown": drawdown,
        "clv_samples": 0,
        "average_clv": None,
        "no_bet_receipts": len(list((args.root / "no_bets").glob("*.json"))),
        "source_errors": errors,
        "actual_wagers": 0,
        "limitations": [
            "CLV integration pending",
            "No profitability claim",
            "Research ledger excluded from recommendation counts",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
