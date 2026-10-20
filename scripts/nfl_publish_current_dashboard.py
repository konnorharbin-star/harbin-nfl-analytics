"""Publish the current independent NFL research bundle as GitHub Pages files.

Never replace frozen first-seen forecasts, alter calibrated betting policies, or
claim existing historical market pages are current. Fail closed on stale inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nfl.public_input_quality import publication_quality

EXPECTED_SPEC = "NFL_INDEPENDENT_FORECAST_PUBLICATION_V1"
PAGE_FILES = {
    "independent_forecasts.json": "independent_latest.json",
    "independent_forecasts.html": "independent_latest.html",
    "independent_forecasts.png": "independent_latest.png",
    "independent_game_predictions.csv": "independent_game_predictions.csv",
    "independent_player_predictions.csv": "independent_player_predictions.csv",
}


def clock(value):
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("UNZONED_PUBLICATION_TIMESTAMP")
    return d.astimezone(UTC)


def validate(bundle, *, at, previous=None):
    """Require a recent original and at least one not-started game; no postgame refresh."""
    if at.tzinfo is None:
        raise ValueError("UNZONED_NOW")
    at = at.astimezone(UTC)
    if bundle.get("spec") != EXPECTED_SPEC or bundle.get("season") != 2026:
        raise ValueError("UNKNOWN_SEASON_OR_FORECAST_FAMILY")
    if (
        bundle.get("sportsbook_prices_used") is not False
        or bundle.get("automatic_betting_enabled") is not False
        or bundle.get("qualified_bets") != 0
        or bundle.get("accuracy_vs_market") != "UNPROVEN"
    ):
        raise ValueError("NOT_RESEARCH_ONLY_INDEPENDENT_FORECAST")
    captured, generated = clock(bundle["model_capture_utc"]), clock(bundle["generated_utc"])
    if not captured <= generated <= at or at - generated > timedelta(hours=3):
        raise ValueError("STALE_OR_FUTURE_RESEARCH_CAPTURE")
    if previous and generated <= clock(previous["generated_utc"]):
        raise ValueError("WOULD_REPLACE_NEWER_PUBLICATION")
    if bundle.get("games_requested") != len(bundle.get("games", [])):
        raise ValueError("INCOMPLETE_GAME_ROSTER")
    futures = []
    for game in bundle["games"]:
        kickoff = clock(game["kickoff_utc"]) if "kickoff_utc" in game else None
        if game.get("predicted_score") is not None:
            if not kickoff or captured >= kickoff:
                raise ValueError("POSTKICKOFF_OR_UNTIMED_FORECAST")
            points = game["predicted_score"]
            if abs(points["home"] + points["away"] - game["predicted_total"]) > 1e-6:
                raise ValueError("INCONSISTENT_SCORE_TOTAL")
            if abs(points["home"] - points["away"] + game["fair_home_spread"]) > 1e-6:
                raise ValueError("INCONSISTENT_SCORE_SPREAD")
            if at < kickoff:
                futures.append(game)
    if not futures:
        raise ValueError("NO_UPCOMING_GAMES_TO_PUBLISH")
    return futures


def dashboard(bundle, *, at, input_quality=None):
    games = bundle["games"]
    future = {g["game_id"] for g in validate(bundle, at=at)}
    rows = []
    for game in games:
        if game.get("predicted_score") is None:
            cells = ("Unavailable", "—", "—", "—", "—")
        else:
            points = game["predicted_score"]
            cells = (
                f"{points['away']:.1f} – {points['home']:.1f}",
                f"{game['fair_home_spread']:+.1f}",
                f"{game['predicted_total']:.1f}",
                html.escape(str(game["kickoff_utc"])),
                "Upcoming" if game["game_id"] in future else "Started – frozen forecast",
            )
        matchup = html.escape(
            str(game.get("away_team", game["game_id"])) + " @ " + str(game.get("home_team", ""))
        )
        rows.append(
            "<tr><td>" + matchup + "</td>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    cap = html.escape(bundle["model_capture_utc"])
    gen = html.escape(bundle["generated_utc"])
    quality = input_quality or {
        "status": "INPUT_CHECKS_UNAVAILABLE",
        "games": [],
        "reason": "NO_CURRENT_INPUT_REPORT",
    }
    quality_label = (
        "Current input checks"
        if quality["status"] == "SUPPLEMENTAL_INPUT_CHECKS_AVAILABLE"
        else "Current input checks unavailable"
    )
    quality_html = "<h2>Input checks</h2><p>" + quality_label + "</p>"
    if quality.get("reason"):
        quality_html += (
            "<p>No matching current input report is available. Forecasts remain conditional.</p>"
        )
    if quality.get("games"):
        quality_html += (
            "<p>These checks were collected after the forecast and did not change its values. "
            "Reported injury statuses do not confirm a starting lineup. Outdoor weather "
            "forecasts do not confirm roof position.</p>"
            "<p><a href='independent_input_quality.json'>"
            "Dated input evidence and source hashes</a></p>"
            "<table><thead><tr><th>Matchup</th><th>Game binding</th><th>Dated injury statuses</th>"
            "<th>Outdoor weather</th><th>Missing or blocked</th></tr></thead><tbody>"
        )
        for item in quality["games"]:
            contract = item["contract"]
            missing = contract["blocking_reasons"] + contract["missing_required_inputs"]
            labels = {
                "confirmed_starting_qbs": "Confirmed starting quarterbacks",
                "final_game_specific_player_availability": "Final player availability",
                "game_specific_player_availability": "Player availability",
                "game_day_roof_position": "Roof position",
                "game_time_weather_and_roof_status": "Weather and roof status",
                "venue_bound_kickoff_weather_forecast": "Weather forecast at kickoff",
            }
            missing = [labels.get(value, "Game or source verification") for value in missing]
            cells = [
                item["away_team"] + " @ " + item["home_team"],
                "Passed" if contract["event_binding_passed"] else "Blocked",
                str(contract["reported_injury_statuses_verified"]),
                "Verified forecast" if contract["outdoor_weather_verified"] else "Unavailable",
                ", ".join(missing),
            ]
            quality_html += (
                "<tr>" + "".join("<td>" + html.escape(c) + "</td>" for c in cells) + "</tr>"
            )
        quality_html += "</tbody></table>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>NFL Independent Forecasts – Week {int(bundle['week'])}</title>"
        "<style>body{background:#101216;color:#f0f1f6;font:15px Arial,sans-serif;"
        "max-width:1250px;margin:26px auto;padding:0 16px}a{color:#9bc8ff}"
        "table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;"
        "border-bottom:1px solid #383d47;text-align:left}th{color:#9ca7b9}"
        "tr:nth-child(even){background:#1c2028}.warn{color:#e5bb6e}"
        ".links a{margin-right:17px}.score{max-width:100%;height:auto}"
        "</style></head><body>"
        f"<h1>Our NFL predictions · Week {int(bundle['week'])}</h1>"
        "<p class='warn'><strong>RESEARCH ONLY · NO APPROVED BETS.</strong> "
        "Scores, spreads and totals are independent conditional forecasts. "
        "Lineups, injuries, overtime, calibration and market superiority are unverified.</p>"
        f"<p>Model captured (UTC): {cap} · Bundle created (UTC): {gen}</p>"
        f"<p>Upcoming matchups at publication: {len(future)}. "
        "Started games are historical forecasts, not live betting signals.</p>"
        "<p class='links'><a href='independent_latest.html'>All player projections</a>"
        "<a href='independent_player_predictions.csv'>Player CSV</a>"
        "<a href='independent_game_predictions.csv'>Game CSV</a>"
        "<a href='independent_latest.json'>Full forecast JSON</a>"
        "<a href='independent_latest.png'>Score dashboard PNG</a>"
        "<a href='legacy_market_week5.html'>Archived Week 5 market board</a></p>"
        "<table><thead><tr><th>Matchup</th><th>Our projected away–home</th>"
        "<th>Our home spread</th><th>Our total</th><th>Kickoff UTC</th>"
        "<th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + quality_html
        + "<img class='score' src='independent_latest.png' "
        "alt='Our independent NFL score predictions'>"
        "<p>Home spread is negative of forecast home margin, not a calibrated cover probability. "
        "No sportsbook price, positive EV or fill is asserted.</p></body></html>\n"
    )


def publish(source, destination, *, at):
    source, destination = Path(source), Path(destination)
    bundle = json.loads((source / "independent_forecasts.json").read_text())
    forecast_sha = hashlib.sha256((source / "independent_forecasts.json").read_bytes()).hexdigest()
    prev_file = destination / "independent_latest.json"
    previous = json.loads(prev_file.read_text()) if prev_file.exists() else None
    futures = validate(bundle, at=at, previous=previous)
    # All validation and existence checks precede any write.
    for src in PAGE_FILES:
        if not (source / src).is_file() or (source / src).stat().st_size == 0:
            raise ValueError("MISSING_BUNDLE_ASSET:" + src)
    quality_path = source / "independent_input_quality.json"
    quality_raw, quality = None, None
    if quality_path.exists():
        try:
            quality_raw = quality_path.read_bytes()
            quality = json.loads(quality_raw)
        except (OSError, ValueError):
            quality = {}  # Present-but-invalid is visibly unavailable.
    quality_view = publication_quality(quality, bundle, forecast_sha256=forecast_sha, at=at)
    rendered = dashboard(bundle, at=at, input_quality=quality_view)
    destination.mkdir(parents=True, exist_ok=True)
    previous_board = destination / "index.html"
    archive = destination / "legacy_market_week5.html"
    if (
        previous_board.is_file()
        and not archive.exists()
        and "NFL Model Week 5" in previous_board.read_text()[:500]
    ):
        shutil.copyfile(previous_board, archive)
    for src, dst in PAGE_FILES.items():
        shutil.copyfile(source / src, destination / dst)
    (destination / "index.html").write_text(rendered, encoding="utf-8")
    # Always overwrite the quality artifact: an older successful check must not
    # survive a missing, stale or mismatched supplemental report.
    published_quality = dict(quality_view)
    if quality_view["status"] == "SUPPLEMENTAL_INPUT_CHECKS_AVAILABLE":
        published_quality["source_evidence"] = quality
    (destination / "independent_input_quality.json").write_text(
        json.dumps(published_quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "spec": "NFL_PAGES_INDEPENDENT_DASHBOARD_V1",
        "model_capture_utc": bundle["model_capture_utc"],
        "generated_utc": bundle["generated_utc"],
        "dashboard_published_utc": at.astimezone(UTC).isoformat(),
        "season": bundle["season"],
        "week": bundle["week"],
        "upcoming_games": len(futures),
        "games_forecast": bundle["games_forecast"],
        "model_lineage": bundle.get("predictive_family"),
        "qualified_bets": 0,
        "market_superiority": "UNPROVEN",
        "source_json_sha256": forecast_sha,
        "input_quality_status": quality_view["status"],
        "input_quality_sha256": hashlib.sha256(quality_raw).hexdigest() if quality_raw else None,
        "legacy_market_board_preserved": archive.exists(),
    }
    (destination / "independent_latest_meta.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    a = parser.parse_args()
    print(json.dumps(publish(a.source, a.docs, at=datetime.now(UTC)), indent=2))


if __name__ == "__main__":
    main()
