"""EA 27 official *team* ratings: read-only pregame residual audit.

Not player ratings. Not a score adjustment, fit, backtest of a challenger,
or verified immutable forward experiment. Evaluates source-dated 2026 team
prior only against 2026 archived predictions and settled game outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

NFL_ABBR = {
    "Rams": "LA", "Eagles": "PHI", "Ravens": "BAL", "Broncos": "DEN",
    "Patriots": "NE", "Lions": "DET", "Seahawks": "SEA", "49ers": "SF",
    "Bills": "BUF", "Chiefs": "KC", "Cowboys": "DAL", "Bears": "CHI",
    "Bengals": "CIN", "Steelers": "PIT", "Chargers": "LAC", "Colts": "IND",
    "Texans": "HOU", "Buccaneers": "TB", "Falcons": "ATL", "Packers": "GB",
    "Panthers": "CAR", "Vikings": "MIN", "Commanders": "WAS", "Giants": "NYG",
    "Jaguars": "JAX", "Browns": "CLE", "Saints": "NO", "Jets": "NYJ",
    "Raiders": "LV", "Cardinals": "ARI", "Titans": "TEN", "Dolphins": "MIA",
}
CFB_ALIASES = {
    "cal": "california", "ole miss": "ole miss", "mississippi": "ole miss",
    "mississippi state": "mississippi st", "san diego state": "san diego st",
    "washington state": "washington st", "western michigan": "w michigan",
    "eastern michigan": "e michigan", "central michigan": "c michigan",
    "western kentucky": "w kentucky", "new mexico state": "new mexico st",
    "appalachian state": "app st", "georgia southern": "ga southern",
    "jacksonville state": "jax state", "kennesaw state": "kennesaw st",
    "coastal carolina": "c carolina", "middle tennessee": "middle tenn",
    "northern illinois": "niu", "louisiana monroe": "ul monroe",
    "ul monroe": "ul monroe", "florida atlantic": "fla atlantic",
    "florida international": "fiu", "south florida": "usf",
    "connecticut": "uconn", "massachusetts": "umass",
    "north dakota state": "ndsu", "sacramento state": "sac state",
    "miami ohio": "miami oh", "miami (oh)": "miami oh",
    "miami fl": "miami", "hawaii": "hawaii",
    "texas san antonio": "utsa", "texas el paso": "utep",
}


def canonical(name: str, sport: str) -> str:
    if sport == "nfl":
        return NFL_ABBR.get(name, name).strip().upper()
    key = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    return CFB_ALIASES.get(key, key)


def aware(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Forecast timestamps must be timezone-aware")
    return stamp


def read_ratings(data_path: Path, meta_path: Path, sport: str) -> tuple[dict, dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    published = date.fromisoformat(meta["published_date"])
    if meta.get("season") != 2026 or meta.get("ratings_level") != "TEAM_NOT_PLAYER":
        raise ValueError("Only source-dated 2026 team ratings are supported")
    if not meta.get("source_url", "").startswith("https://www.ea.com/"):
        raise ValueError("Official source attribution required")
    with data_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not {"team", "offense", "defense", "overall"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError("Ratings schema missing required fields")
        rows = list(reader)
    if not rows:
        raise ValueError("Ratings file is empty")
    ratings = {}
    for row in rows:
        team = canonical(row["team"], sport)
        if not team or team in ratings:
            raise ValueError(f"Duplicate/blank team identity: {team!r}")
        try:
            values = tuple(int(row[k]) for k in ("offense", "defense", "overall"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EA ratings for {team}") from exc
        if any(not 0 <= v <= 99 for v in values):
            raise ValueError(f"Out-of-range EA ratings for {team}")
        ratings[team] = values
    return ratings, {
        "source_url": meta["source_url"],
        "published_date": published.isoformat(),
        "ratings_count": len(ratings),
        "source_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "article_revision_not_independently_archived": True,
    }


def read_games(path: Path, sport: str) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        need = (
            {"season", "game_id", "home_team", "away_team", "kickoff",
             "captured_at", "baseline_home_margin", "baseline_total",
             "home_score", "away_score"}
            if sport == "nfl" else
            {"season", "game_id", "home_team", "away_team", "kickoff",
             "first_snapshot", "projected_margin_home", "projected_total",
             "actual_margin_home", "actual_total"}
        )
        if not need.issubset(reader.fieldnames or []):
            raise ValueError("Archived predictions missing required columns")
        data = list(reader)
    if not data:
        raise ValueError("No prediction rows")
    output, seen = [], set()
    excluded = defaultdict(int)
    for r in data:
        if int(r["season"]) != 2026:
            excluded["not_2026"] += 1
            continue
        game_id = str(r["game_id"]).strip()
        if not game_id or game_id in seen:
            raise ValueError("Duplicate or blank game id")
        seen.add(game_id)
        kickoff = aware(r["kickoff"])
        observed = aware(r["captured_at" if sport == "nfl" else "first_snapshot"])
        if observed >= kickoff:
            excluded["late_or_nonpregame_snapshot"] += 1
            continue
        fields = (
            ("baseline_home_margin", "baseline_total")
            if sport == "nfl" else
            ("projected_margin_home", "projected_total")
        )
        try:
            pm, pt = (float(r[k]) for k in fields)
            if sport == "nfl":
                hs, as_ = float(r["home_score"]), float(r["away_score"])
                am, at = hs - as_, hs + as_
            else:
                am, at = float(r["actual_margin_home"]), float(r["actual_total"])
        except (TypeError, ValueError):
            excluded["ungraded_game"] += 1
            continue
        if not all(math.isfinite(x) for x in (pm, pt, am, at)):
            excluded["nonfinite_score"] += 1
            continue
        if pt < 0 or at < 0 or abs(am) > at:
            raise ValueError(f"Impossible score for {game_id}")
        output.append({
            "game_id": game_id, "kickoff": kickoff, "observed": observed,
            "home": canonical(r["home_team"], sport),
            "away": canonical(r["away_team"], sport),
            "projected_margin": pm, "projected_total": pt,
            "actual_margin": am, "actual_total": at,
        })
    return output, dict(excluded)


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    vx = sum((a - xm) ** 2 for a in x)
    vy = sum((b - ym) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True)) / (
        math.sqrt(vx) * math.sqrt(vy)
    )


def audit(
    ratings: dict, games: list[dict], published_date: str, sport: str
) -> tuple[dict, list[dict]]:
    publication = date.fromisoformat(published_date)
    matched = []
    excluded = defaultdict(int)
    missing_teams = defaultdict(int)
    for g in games:
        if g["kickoff"].date() <= publication:
            excluded["kickoff_before_or_on_publication_date"] += 1
            continue
        if g["observed"].date() <= publication:
            excluded["forecast_before_or_on_publication_date"] += 1
            continue
        if g["home"] not in ratings or g["away"] not in ratings:
            excluded["missing_ea_team_rating"] += 1
            for team in (g["home"], g["away"]):
                if team not in ratings:
                    missing_teams[team] += 1
            continue
        home, away = ratings[g["home"]], ratings[g["away"]]
        # Opponent-aware offensive/defensive EA units, *not scoreboard points*.
        home_matchup = home[0] - away[1]
        away_matchup = away[0] - home[1]
        diff = home_matchup - away_matchup
        total_pressure = home_matchup + away_matchup
        matched.append({
            "game_id": g["game_id"],
            "home_team": g["home"], "away_team": g["away"],
            "source_published_date": publication.isoformat(),
            "forecast_captured_at": g["observed"].isoformat(),
            "kickoff": g["kickoff"].isoformat(),
            "home_offense_ea": home[0], "away_offense_ea": away[0],
            "home_defense_ea": home[1], "away_defense_ea": away[1],
            "talent_matchup_diff_units": diff,
            "talent_total_pressure_units": total_pressure,
            "base_margin_error": g["projected_margin"] - g["actual_margin"],
            "base_total_error": g["projected_total"] - g["actual_total"],
            "projected_margin": g["projected_margin"],
            "actual_margin": g["actual_margin"],
            "projected_total": g["projected_total"],
            "actual_total": g["actual_total"],
        })
    def average(items: list[float]) -> float | None:
        return sum(items) / len(items) if items else None
    by_group = {}
    for name, selector in {
        "EA_HOME_ADVANTAGE": lambda r: r["talent_matchup_diff_units"] >= 3,
        "EA_EVEN": lambda r: abs(r["talent_matchup_diff_units"]) < 3,
        "EA_AWAY_ADVANTAGE": lambda r: r["talent_matchup_diff_units"] <= -3,
    }.items():
        subset = [row for row in matched if selector(row)]
        by_group[name] = {
            "games": len(subset),
            "baseline_margin_mae": average(
                [abs(row["base_margin_error"]) for row in subset]
            ),
            "baseline_total_mae": average(
                [abs(row["base_total_error"]) for row in subset]
            ),
        }
    result = {
        "status": "DESCRIPTIVE_ONLY_NOT_VALIDATED_PREDICTIVE_GAIN",
        "sport": sport, "year": 2026, "published_date": published_date,
        "eligible_pregame_graded_games": len(games),
        "matched_games": len(matched),
        "coverage": len(matched) / len(games) if games else 0,
        "exclusions": dict(excluded),
        "missing_team_ids": dict(sorted(missing_teams.items())),
        "base_margin_mae": average(
            [abs(row["base_margin_error"]) for row in matched]
        ),
        "base_total_mae": average(
            [abs(row["base_total_error"]) for row in matched]
        ),
        "talent_residual_margin_corr": _corr(
            [float(row["talent_matchup_diff_units"]) for row in matched],
            [row["base_margin_error"] for row in matched],
        ),
        "talent_residual_total_corr": _corr(
            [float(row["talent_total_pressure_units"]) for row in matched],
            [row["base_total_error"] for row in matched],
        ),
        "by_pregame_ea_advantage": by_group,
        "candidate_margin_mae": None,
        "candidate_total_mae": None,
        "approved_prediction_adjustment": 0,
        "approved_wager_adjustment": 0,
        "caveats": [
            "Official team ratings are not player ratings or availability.",
            "Dated EA news article values have no independent immutable revision proof.",
            "Archived forecast timestamps alone do not prove independent precommitment.",
            "No challenger forecast or scoring conversion has been fitted or validated.",
            "Correlations are descriptive and cannot establish causality or betting edge.",
        ],
    }
    return result, matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-output", required=True, type=Path)
    args = parser.parse_args()
    ratings, source = read_ratings(args.ratings, args.metadata, args.sport)
    games, exclusions = read_games(args.games, args.sport)
    results, rows = audit(
        ratings, games, source["published_date"], args.sport
    )
    results["source_provenance"] = source
    results["archive_exclusions"] = exclusions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with args.rows_output.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("game_id\n")
    print(json.dumps({
        "status": results["status"], "games": results["matched_games"],
        "coverage": results["coverage"],
        "base_margin_mae": results["base_margin_mae"],
        "base_total_mae": results["base_total_mae"],
        "talent_residual_margin_corr": results["talent_residual_margin_corr"],
        "talent_residual_total_corr": results["talent_residual_total_corr"],
        "missing_team_ids": results["missing_team_ids"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
