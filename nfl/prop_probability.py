"""NFL player-stat threshold probabilities from pregame-only empirical errors.

Development: 2024 pregame predictions and outcomes fit conditional error CDFs.
Diagnostic: 2025 fixed market-independent synthetic half-point thresholds.
Forward: first-seen 2026 probabilities at those same fixed thresholds.

Not a calibrated *sportsbook* model: no contemporaneous prop odds, active/inactive
verification or fresh out-of-sample profitability record. No wagers.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .prop_challenger import METRICS, forecast_week

PROP_THRESHOLD_GRID = {
    "receptions": (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5),
    "receiving_yards": (24.5, 39.5, 49.5, 64.5, 79.5, 99.5, 124.5),
    "rushing_yards": (19.5, 34.5, 49.5, 64.5, 79.5, 99.5, 124.5),
    "passing_yards": (149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5),
}
SCALE_FLOORS = {
    "receptions": 1.5,
    "receiving_yards": 18.0,
    "rushing_yards": 18.0,
    "passing_yards": 45.0,
}
MIN_POSITION_RESIDUALS = 80
MIN_MARKET_RESIDUALS = 160
MIN_VALID_DIAGNOSTIC = 120
PROBABILITY_FLOOR = 0.01
VERSION = "prop_empirical_error_cdf_v1"
FORWARD_FIELDS = (
    "version", "first_observed_at_utc", "game_id", "season", "week",
    "kickoff_utc", "team", "player_id", "player_name", "position",
    "market", "synthetic_line", "probability_over",
    "probability_under", "residual_sample_count",
    "probability_basis", "book_price", "book",
    "quoted_at", "model_betting_ev", "is_bet_recommendation",
)
OUT_PATH = Path("docs/player_prop_probability_shadow.csv")
LEDGER_PATH = Path("history/player_prop_probability_forward.csv")


def _scale(market: str, prediction: float) -> float:
    floor = SCALE_FLOORS[market]
    return max(floor, floor * .25 + .45 * abs(prediction))


def _actual(outcomes: dict[tuple[str, str], dict[str, Any]],
            row: dict[str, Any]) -> float | None:
    player = outcomes.get((str(row["game_id"]),str(row["player_id"])))
    if player is None:
        # The prediction was frozen before the game; a missing player is DNP.
        # Whether the player was ruled out is NOT independently known.
        return 0.0
    observed = player.get(row["market"])
    if observed is None or not math.isfinite(float(observed)):
        return None
    return float(observed)


def _predictions_by_season(
    games: list[dict[str, Any]], stats: list[dict[str, Any]],
    *, season: int, options: dict[str, tuple[int, float]]
) -> list[tuple[dict[str, Any], float]]:
    outcomes = {(str(r["game_id"]),str(r["player_id"])):r for r in stats}
    completed = {
        str(g["game_id"]) for g in games
        if g["season"] == season and g["completed"]
    }
    weeks = sorted({
        g["week"] for g in games
        if g["season"] == season and g["completed"] and g["week"] >= 4
    })
    entries: list[tuple[dict[str, Any],float]] = []
    for week in weeks:
        for prediction in forecast_week(
            games, stats, season=season, week=week, options=options
        ):
            if prediction["game_id"] not in completed:
                continue
            realized = _actual(outcomes,prediction)
            if realized is not None:
                entries.append((prediction,realized))
    return entries


def fit_empirical_errors(
    samples: list[tuple[dict[str, Any],float]],
) -> dict[tuple[str, str, str], list[float]]:
    """Residuals normalized to stat-scale, with fixed position fallback."""
    grouped: dict[tuple[str,str,str],list[float]] = defaultdict(list)
    for row, actual in samples:
        market = row["market"]
        position = row["position"]
        for method, field in (
            ("challenger","challenger_mean"),
            ("baseline","baseline_three_game_mean"),
        ):
            prediction = float(row[field])
            normalized = (actual-prediction)/_scale(market,prediction)
            grouped[(method,market,position)].append(normalized)
            grouped[(method,market,"ALL")].append(normalized)
    for residuals in grouped.values():
        residuals.sort()
    return dict(grouped)


def _errors_for(
    grouped: dict[tuple[str,str,str],list[float]], *,
    method: str, market: str, position: str
) -> tuple[list[float],str] | None:
    player = grouped.get((method,market,position),[])
    if len(player) >= MIN_POSITION_RESIDUALS:
        return player, "POSITION_2024_EMPIRICAL"
    overall = grouped.get((method,market,"ALL"),[])
    if len(overall) >= MIN_MARKET_RESIDUALS:
        return overall, "MARKET_2024_EMPIRICAL"
    return None


def over_probability(
    forecast: float, line: float, market: str, errors: list[float]
) -> float:
    """Monotone empirical tail with Beta(1,1) smoothing and tail clipping.

    Fixed half-point lines have no push. Returned probabilities are research
    forecasts, never bookmaker-calibrated execution probabilities.
    """
    if market not in PROP_THRESHOLD_GRID or not math.isfinite(forecast):
        raise ValueError("unknown stat or nonfinite forecast")
    if not math.isfinite(line) or line < 0 or line % 1 != .5 or not errors:
        raise ValueError("synthetic threshold must be finite nonnegative half point")
    normalized = (line-forecast)/_scale(market,forecast)
    above = sum(error > normalized for error in errors)
    p = (above + 1) / (len(errors) + 2)
    return max(PROBABILITY_FLOOR,min(1-PROBABILITY_FLOOR,p))


def probability_rows(
    predictions: list[dict[str, Any]], *,
    residuals: dict[tuple[str,str,str],list[float]],
    method: str="challenger",
) -> list[dict[str, Any]]:
    curves: list[dict[str,Any]] = []
    for row in predictions:
        market = row["market"]
        found = _errors_for(
            residuals,method=method,market=market,position=row["position"]
        )
        if found is None:
            continue
        errors,basis = found
        previous = 1.0
        for line in PROP_THRESHOLD_GRID[market]:
            p = over_probability(float(row[
                "challenger_mean" if method=="challenger"
                else "baseline_three_game_mean"
            ]),line,market,errors)
            if p > previous + 1e-12:
                raise RuntimeError("over probabilities must be monotone by line")
            previous = p
            curves.append({
                "game_id":row["game_id"],"season":row["season"],
                "week":row["week"],"kickoff_utc":row["kickoff_utc"],
                "team":row["team"],"player_id":row["player_id"],
                "player_name":row["player_name"],"position":row["position"],
                "market":market,"synthetic_line":line,
                "probability_over":round(p,6),
                "probability_under":round(1-p,6),
                "residual_sample_count":len(errors),
                "probability_basis":basis,
                "book_price":None,"book":None,"quoted_at":None,
                "model_betting_ev":None,"is_bet_recommendation":False,
            })
    return curves


def _scores(items: list[tuple[int, float, float]]) -> dict[str, Any]:
    if not items:
        return {"n":0,"brier":None,"log_loss":None,"ece_10":None}
    n=len(items)
    brier=sum((p-y)**2 for y,p,_ in items)/n
    loss=sum(-(y*math.log(p)+(1-y)*math.log1p(-p)) for y,p,_ in items)/n
    bins = defaultdict(list)
    for y,p,_ in items:
        bins[min(9,int(p*10))].append((y,p))
    ece=sum(
        len(v)/n * abs(
            sum(y for y,_ in v)/len(v) - sum(p for _,p in v)/len(v)
        )
        for v in bins.values()
    )
    return {"n":n,"brier":round(brier,7),
            "log_loss":round(loss,7),"ece_10":round(ece,7)}


def _weekly_interval(
    weekly: dict[int,list[float]]
) -> tuple[float,float] | None:
    """Deterministic, week-clustered 95% descriptive bootstrap interval."""
    weeks=sorted(weekly)
    if len(weeks)<8:
        return None
    rng=random.Random(20261008)
    per_week=[sum(weekly[w])/len(weekly[w]) for w in weeks]
    boot=[]
    for _ in range(400):
        chosen=[per_week[rng.randrange(len(weeks))] for _ in weeks]
        boot.append(sum(chosen)/len(chosen))
    boot.sort()
    return round(boot[9],7),round(boot[390],7)


def evaluate_diagnostic(
    entries: list[tuple[dict[str,Any],float]],
    residuals: dict[tuple[str,str,str],list[float]],
) -> dict[str,Any]:
    by_market={}
    for market in METRICS:
        selected=[(r,y) for r,y in entries if r["market"]==market]
        challenger=[]
        baseline=[]
        paired_weeks: dict[int,list[float]]=defaultdict(list)
        distinct_games=set()
        for r,y in selected:
            ca=_errors_for(
                residuals,method="challenger",market=market,position=r["position"]
            )
            ba=_errors_for(
                residuals,method="baseline",market=market,position=r["position"]
            )
            if ca is None or ba is None:
                continue
            distinct_games.add(r["game_id"])
            for line in PROP_THRESHOLD_GRID[market]:
                outcome=int(y>line)
                pc=over_probability(float(r["challenger_mean"]),line,market,ca[0])
                pb=over_probability(
                    float(r["baseline_three_game_mean"]),line,market,ba[0]
                )
                challenger.append((outcome,pc,line))
                baseline.append((outcome,pb,line))
                paired_weeks[int(r["week"])].append(
                    (pc-outcome)**2-(pb-outcome)**2
                )
        cs,bs=_scores(challenger),_scores(baseline)
        if cs["n"]<MIN_VALID_DIAGNOSTIC:
            status="INSUFFICIENT_2025_SYNTHETIC_THRESHOLDS"
        else:
            status="HISTORICAL_SYNTHETIC_LINES_DIAGNOSTIC_ONLY"
        by_market[market]={
            "status":status,"player_game_rows":len(selected),
            "evaluated_synthetic_thresholds":cs["n"],
            "unique_games":len(distinct_games),
            "distinct_weeks":len(paired_weeks),
            "fixed_threshold_grid":list(PROP_THRESHOLD_GRID[market]),
            "challenger":cs,"rolling_three_baseline":bs,
            "brier_delta_challenger_minus_baseline":(
                round(cs["brier"]-bs["brier"],7)
                if cs["brier"] is not None else None
            ),
            "week_clustered_brier_delta_ci95":_weekly_interval(paired_weeks),
            "tested_vs_actual_bookmaker_odds":False,
            "profitable_market_edge_proven":False,
        }
    return by_market


def build_probability_experiment(
    games: list[dict[str,Any]], stats: list[dict[str,Any]], *,
    options: dict[str,tuple[int,float]],
    season: int, week: int,
    current_forecasts: list[dict[str,Any]],
) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    training=_predictions_by_season(games,stats,season=2024,options=options)
    if len(training)<MIN_MARKET_RESIDUALS:
        raise ValueError("Not enough 2024 strictly pregame player-stat residuals")
    residuals=fit_empirical_errors(training)
    diagnostic=_predictions_by_season(games,stats,season=2025,options=options)
    curves=probability_rows(current_forecasts,residuals=residuals)
    return {
        "schema_version":1,"spec_version":VERSION,
        "development_season":2024,"diagnostic_season":2025,
        "diagnostic_is_pristine_holdout":False,
        "prospective_season":season,"prospective_week":week,
        "2024_development_player_stat_rows":len(training),
        "2025_diagnostic_player_stat_rows":len(diagnostic),
        "2026_forward_probability_rows":len(curves),
        "2026_forward_unique_player_markets":len({
            (r["game_id"],r["player_id"],r["market"]) for r in curves
        }),
        "2025_synthetic_threshold_diagnostic":evaluate_diagnostic(diagnostic,residuals),
        "market_odds_used":False,
        "synthetic_lines_are_offered_sportsbook_lines":False,
        "probabilities_calibrated_against_bookmakers":False,
        "calibrated_sportsbook_edge_proven":False,
        "automatic_betting_enabled":False,
        "paid_sources_used":False,
        "limitations":[
            "Fixed hypothetical half-point thresholds, not bookmaker quotes.",
            "2024 hyperparameters and residuals share development data; "
            "2025 was previously inspected and is not a fresh untouched holdout.",
            "Pre-game expected active-player status unknown; zero for missing stat rows.",
            "Synthetic thresholds within one game/player are correlated, "
            "so threshold counts are not independent observations.",
            "Empirical residuals can be miscalibrated by opponent, player usage, "
            "injury, roster role and changing team context.",
            "Clipping/Beta smoothing do not prove prediction quality or profit.",
        ],
    },curves


def write_probability_csv(
    rows: list[dict[str,Any]], path: Path=OUT_PATH,
    *, fields: tuple[str,...]=FORWARD_FIELDS
) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_first_seen_curves(
    rows: list[dict[str,Any]], *,
    observed_at: datetime, destination: Path=LEDGER_PATH,
) -> dict[str,int]:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    prior=[]
    if destination.exists():
        with destination.open(newline="",encoding="utf-8") as f:
            prior=list(csv.DictReader(f))
    def key(row: dict[str,Any]) -> tuple[str,str,str,str]:
        return (
            str(row["game_id"]),str(row["player_id"]),
            str(row["market"]),str(row["synthetic_line"]),
        )
    seen={key(x) for x in prior}
    appended=0
    for item in rows:
        kickoff=datetime.fromisoformat(str(item["kickoff_utc"]))
        if kickoff.tzinfo is None or kickoff<=observed_at or key(item) in seen:
            continue
        row={field:item.get(field) for field in FORWARD_FIELDS}
        row["version"]=VERSION
        row["first_observed_at_utc"]=observed_at.astimezone(UTC).isoformat()
        prior.append(row)
        seen.add(key(item))
        appended+=1
    write_probability_csv(prior,destination)
    return {"new_entries":appended,"existing_entries":len(prior)-appended,
            "total_entries":len(prior)}
