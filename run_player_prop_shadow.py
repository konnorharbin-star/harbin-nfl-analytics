"""Run free research-only player-stat challenger and freeze first pregame forecasts."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl.data import NFLDataClient
from nfl.prop_challenger import diagnostic, forecast_week, source_frames
from nfl.prop_probability import LEDGER_PATH as PROBABILITY_LEDGER
from nfl.prop_probability import OUT_PATH as PROBABILITY_CSV
from nfl.prop_probability import (
    append_first_seen_curves,
    build_probability_experiment,
    write_probability_csv,
)

OUTPUT_JSON = Path("docs/player_prop_shadow.json")
OUTPUT_CSV = Path("docs/player_prop_shadow.csv")
LEDGER = Path("history/player_prop_forward.csv")
VERSION = "prop_forecast_research_v1"

FORWARD_FIELDS = (
    "version", "first_observed_at_utc", "game_id", "season", "week",
    "kickoff_utc", "team", "player_id", "player_name", "position", "market",
    "baseline_three_game_mean", "challenger_mean",
    "window", "shrinkage", "player_active_at_kickoff_verified",
    "calibrated_prop_probability", "book", "line",
)


def _write_csv(path: Path, rows: list[dict[str, Any]],
               fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_first_seen(rows: list[dict[str, Any]], *,
                      timestamp: datetime, ledger: Path = LEDGER) -> dict[str, int]:
    if timestamp.tzinfo is None:
        raise ValueError("prospective observation must be timezone-aware")
    existing: list[dict[str, Any]] = []
    if ledger.exists():
        with ledger.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (str(row["game_id"]), str(row["player_id"]), str(row["market"]))

    seen = {key(x) for x in existing}
    added = 0
    for row in rows:
        start = datetime.fromisoformat(str(row["kickoff_utc"]))
        if start <= timestamp or key(row) in seen:
            continue
        record = {k:row.get(k) for k in FORWARD_FIELDS}
        record["version"] = VERSION
        record["first_observed_at_utc"] = timestamp.astimezone(UTC).isoformat()
        existing.append(record)
        seen.add(key(row))
        added+=1
    _write_csv(ledger,existing,FORWARD_FIELDS)
    return {"previous_entries": len(existing)-added, "new_entries": added,
            "total_entries": len(existing)}


def _week_from_latest(path: Path) -> tuple[int,int]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    weeks = []
    for row in rows:
        try:
            weeks.append((int(row["season"]),int(row["week"])))
        except (ValueError, KeyError, TypeError):
            continue
    if not weeks:
        raise ValueError("docs/latest.csv has no season/week for a current slate")
    return max(weeks)


def build_shadow_report(
    *, client: NFLDataClient,
    season: int, week: int, as_of: datetime,
    out_json: Path=OUTPUT_JSON, out_csv: Path=OUTPUT_CSV,
    ledger: Path=LEDGER,
) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("as_of requires timezone")
    years = sorted({2024, 2025, season})
    games = client.load_schedules(years)
    players = client.load_player_stats(years)
    schedule, lines, source = source_frames(games, players)
    evidence = diagnostic(schedule, lines)
    options = {
        market: tuple(value) for market, value in
        evidence["development_choices"].items()
    }
    forecasts = forecast_week(
        schedule, lines, season=season, week=week, options=options,
        future_only_at=as_of,
    )
    ledger_counts = append_first_seen(
        forecasts,timestamp=as_of,ledger=ledger
    )
    _write_csv(
        out_csv, forecasts,
        (
            "game_id","season","week","kickoff_utc","team",
            "player_id","player_name","position","market",
            "prior_season_games","most_recent_seen_week",
            "baseline_three_game_mean","challenger_mean",
            "window","shrinkage","line","book","true_market_ev",
            "recommendation",
        ),
    )
    report = {
        "schema_version": 1, "generated_at_utc":as_of.astimezone(UTC).isoformat(),
        "season":season, "week":week,
        "status":"RESEARCH_ONLY_UNPRICED_PLAYER_FORECASTS",
        "automatic_betting_enabled":False, "paid_sources_used":False,
        "actual_sportsbook_edge_proven":False,
        "2025_is_pristine_holdout":False,
        "input_diagnostics":source,
        "historical_diagnostic":evidence,
        "forward_ledger":ledger_counts,
        "pregame_player_market_forecasts":len(forecasts),
        "unique_players":len({x["player_id"] for x in forecasts}),
        "unpriced_markets":sorted({x["market"] for x in forecasts}),
        "blocked_price_reason":"NO_VERIFIED_CURRENT_PLAYER_PROP_BOOK_ODDS",
        "limitations":[
            "Estimates are point forecasts only; no calibrated distribution or prop probabilities.",
            ("A player not listed at target kickoff may not participate; "
             "starting/active status is unknown."),
            ("2024 chooses hyperparameters. 2025 has been inspected in earlier "
             "football modeling: diagnostic, not untouched holdout."),
            ("Historical player source may receive revisions; prospective pregame "
             "forecasts are preserved first-seen."),
            ("Missing postgame player rows are graded zero, including inactive "
             "players; detailed DNP status is not verified."),
            ("No fair EV/price, market-beating result, CLV, wagering permission "
             "or tradeable edge can be inferred."),
        ],
    }
    # Statistical player-prop probability experiment is isolated from the
    # main model and from unpriced mean forecasts. An optional calibration
    # failure must be explicit rather than reusing stale saved probabilities.
    probability_json = out_json.parent / "player_prop_probability_shadow.json"
    probability_csv = out_csv.parent / "player_prop_probability_shadow.csv"
    try:
        probability_report, curves = build_probability_experiment(
            schedule, lines, options=options, season=season, week=week,
            current_forecasts=forecasts,
        )
        write_probability_csv(curves, probability_csv)
        probability_report["frozen_forward"] = append_first_seen_curves(
            curves, observed_at=as_of, destination=PROBABILITY_LEDGER
            if out_json == OUTPUT_JSON else out_json.parent / "probability_forward.csv",
        )
        probability_report["status"] = "SYNTHETIC_LINES_DIAGNOSTIC_RESEARCH"
    except Exception as exc:
        probability_report = {
            "status": "BLOCKED_PROBABILITY_EXPERIMENT",
            "generated_at_utc": as_of.astimezone(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "automatic_betting_enabled": False,
            "calibrated_sportsbook_edge_proven": False,
            "2026_forward_probability_rows": 0,
        }
        write_probability_csv([], probability_csv)
    probability_json.write_text(
        json.dumps(probability_report, indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    )
    report["player_prop_probability_experiment"] = {
        "status": probability_report["status"],
        "forward_probability_rows": probability_report.get(
            "2026_forward_probability_rows",0
        ),
        "calibrated_sportsbook_edge_proven": False,
    }
    out_json.parent.mkdir(parents=True,exist_ok=True)
    out_json.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",type=int)
    parser.add_argument("--week",type=int)
    a=parser.parse_args()
    season,week= (
        (a.season,a.week) if a.season and a.week
        else _week_from_latest(Path("docs/latest.csv"))
    )
    try:
        report=build_shadow_report(
            client=NFLDataClient(), season=season,week=week,
            as_of=datetime.now(UTC),
        )
    except Exception as exc:
        probability_json = OUTPUT_JSON.parent / "player_prop_probability_shadow.json"
        probability_json.write_text(json.dumps({
            "status": "BLOCKED_PLAYER_STAT_SOURCE_OR_SCHEMA",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "automatic_betting_enabled": False,
            "calibrated_sportsbook_edge_proven": False,
        }, sort_keys=True, indent=2) + "\n")
        write_probability_csv([], PROBABILITY_CSV)
        # Do not break the canonical NFL model when an optional, free
        # player-stat source is unavailable; replace any stale report with
        # an explicit failure, never silently re-use old forecasts.
        failure = {
            "status":"BLOCKED_PLAYER_STAT_SOURCE_OR_SCHEMA",
            "generated_at_utc":datetime.now(UTC).isoformat(),
            "error":f"{type(exc).__name__}: {exc}",
            "automatic_betting_enabled":False,
            "paid_sources_used":False,
            "actual_sportsbook_edge_proven":False,
            "pregame_player_market_forecasts":0,
        }
        OUTPUT_JSON.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(failure,indent=2,sort_keys=True)+"\n")
        _write_csv(OUTPUT_CSV,[],(
            "game_id","player_id","market","challenger_mean","recommendation"
        ))
        print(f"::warning::Player prop shadow blocked: {failure['error']}")
        return 0
    print(json.dumps({
        "status":report["status"],
        "forecast_rows":report["pregame_player_market_forecasts"],
        "ledger":report["forward_ledger"],
        "historical":report["historical_diagnostic"]["by_market"],
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
