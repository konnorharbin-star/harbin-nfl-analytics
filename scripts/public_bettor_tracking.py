"""Frozen public analyst-pick tracker. Research only; no sportsbook execution.

Reads independently captured, timestamped editorial selections and grades only
settled NFL main markets. Props remain unresolved without audited player boxes.
Free source: nflverse-derived 2026 NFL schedules through a public GitHub mirror.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from urllib.request import Request, urlopen

REQUIRED = (
    "record_id", "league", "season", "week", "event_id", "kickoff_utc",
    "analyst", "source_url", "source_updated_utc", "observed_utc",
    "source_type", "market", "selection", "participant", "line",
    "american_odds", "bookmaker", "entry_status",
)
SCHEDULE = ("https://raw.githubusercontent.com/antonwilms/"
            "sleeper-dashboard-data/main/nflverse/schedule/{year}.json")


def as_utc(value):
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("all timestamps must include a UTC offset")
    return moment


def read_picks(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    ids = set()
    for row in rows:
        missing = [x for x in REQUIRED if not row.get(x)]
        if missing:
            raise ValueError(f"missing fields {missing} for {row.get('record_id')}")
        if row["record_id"] in ids:
            raise ValueError(f"duplicate tracked pick {row['record_id']}")
        ids.add(row["record_id"])
        if row["league"] != "NFL":
            raise ValueError("NFL tracker rejects unknown sport")
        published = as_utc(row["source_updated_utc"])
        captured = as_utc(row["observed_utc"])
        kickoff = as_utc(row["kickoff_utc"])
        if not published <= captured < kickoff:
            raise ValueError("pick lacks verified pre-kickoff temporal ordering")
        if row["source_type"] != "EDITORIAL_PUBLISHED":
            raise ValueError("only frozen published editorial picks accepted")
        if row["market"] not in {"spread", "total", "moneyline", "receiving_yards",
                                 "receptions", "rushing_yards", "anytime_td"}:
            raise ValueError("unsupported or ambiguous market")
        selection = row["selection"].upper()
        if row["market"] in {"spread", "moneyline"}:
            if not selection.isalpha() or not 2 <= len(selection) <= 3:
                raise ValueError("a team market requires an NFL team abbreviation")
        elif row["selection"].lower() not in {"over", "under"}:
            raise ValueError("prop and total markets require over or under")
        if row["market"] in {"spread", "total", "receiving_yards", "receptions",
                              "rushing_yards"}:
            float(row["line"])
        if int(row["american_odds"]) == 0:
            raise ValueError("invalid American odds")
        if row["entry_status"] != "OBSERVED_EDITORIAL_NOT_EXECUTABLE":
            raise ValueError("cannot treat article prices as executable")
    return rows


def grade(pick, game):
    """Return (W/L/P/PENDING/NEEDS_PLAYER_BOX, not account-validated)."""
    if not game or game.get("homeScore") is None or game.get("awayScore") is None:
        return "PENDING"
    if pick["market"] not in {"spread", "total", "moneyline"}:
        return "NEEDS_PLAYER_BOX"
    home = float(game["homeScore"])
    away = float(game["awayScore"])
    home_team = game["homeTeam"].upper()
    away_team = game["awayTeam"].upper()
    selection = pick["selection"].upper()
    if pick["market"] == "total":
        diff = home + away - float(pick["line"])
        if pick["selection"].lower() == "under":
            diff = -diff
    elif pick["market"] == "spread":
        if selection not in {home_team, away_team}:
            raise ValueError("spread pick does not name a scheduled team")
        diff = ((home - away) if selection == home_team else (away - home)) + float(pick["line"])
    else:
        if selection not in {home_team, away_team}:
            raise ValueError("moneyline pick does not name a scheduled team")
        diff = (home - away) if selection == home_team else (away - home)
    return "W" if diff > 0 else "L" if diff < 0 else "P"


def theoretical_units(outcome, odds):
    """Article price hypothetical only. Never represent as accepted stake."""
    if outcome == "W":
        value = int(odds)
        return value / 100 if value > 0 else 100 / abs(value)
    if outcome == "L":
        return -1.0
    return 0.0 if outcome == "P" else None


def review(picks, games):
    by_id = {g["gameId"]: g for g in games}
    rows = []
    for pick in picks:
        result = grade(pick, by_id.get(pick["event_id"]))
        units = theoretical_units(result, pick["american_odds"])
        rows.append({
            "record_id": pick["record_id"],
            "analyst": pick["analyst"],
            "market": pick["market"],
            "selection": pick["selection"],
            "line": pick["line"],
            "article_american_odds": int(pick["american_odds"]),
            "source_url": pick["source_url"],
            "published_utc": pick["source_updated_utc"],
            "observed_utc": pick["observed_utc"],
            "grade": result,
            "article_price_hypothetical_units": units,
            "account_quote_verified": False,
            "executed_bet": False,
        })
    per_author = {}
    for r in rows:
        entry = per_author.setdefault(r["analyst"], {"wins": 0, "losses": 0,
                                                      "pushes": 0, "pending": 0,
                                                      "needs_player_box": 0,
                                                      "hypothetical_units": 0.0})
        out = r["grade"]
        if out == "W":
            entry["wins"] += 1
        elif out == "L":
            entry["losses"] += 1
        elif out == "P":
            entry["pushes"] += 1
        elif out == "PENDING":
            entry["pending"] += 1
        else:
            entry["needs_player_box"] += 1
        if r["article_price_hypothetical_units"] is not None:
            entry["hypothetical_units"] += r["article_price_hypothetical_units"]
    return {
        "status": "RESEARCH_ONLY",
        "pick_count": len(rows),
        "scope": "Frozen published opinions; not bettor account execution records.",
        "price_provenance": "EDITORIAL_ONLY_NOT_EXECUTABLE",
        "uses_model_as_forecast": False,
        "decision": "PASS",
        "author_summary": per_author,
        "picks": rows,
    }


def load_schedule(year, input_path=None):
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    req = Request(SCHEDULE.format(year=year),
                  headers={"User-Agent": "BetMGMEgdeLabPublicEditorialAudit/1.0"})
    with urlopen(req, timeout=25) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks", type=Path, default=Path("research/public_expert_picks_2026.csv"))
    p.add_argument("--schedule-file", type=Path)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("reports/public_bettor_tracking.json"))
    args = p.parse_args()
    picks = read_picks(args.picks)
    raw = load_schedule(args.season, args.schedule_file)
    result = review(picks, raw["games"])
    result["season"] = args.season
    result["schedule_snapshot_generated_at"] = raw.get("generatedAt")
    result["schedule_source"] = str(args.schedule_file or SCHEDULE.format(year=args.season))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "picks": len(picks),
                      "grade_counts": {k: sum(1 for r in result["picks"] if r["grade"] == k)
                                       for k in ("W", "L", "P", "PENDING", "NEEDS_PLAYER_BOX")}},
                     indent=2))


if __name__ == "__main__":
    main()
