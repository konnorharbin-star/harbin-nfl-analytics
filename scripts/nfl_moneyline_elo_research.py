"""NFL two-way moneyline research: outcome-only Elo, no wager or quote assertions.

All games in one NFL week are predicted BEFORE scores in that same week update
team ratings. NFL historical archive moneylines are retrospectively observed
comparators only: neither origin-timestamped nor accepted U.S. BetMGM offers.
2024-25 diagnostics were previously inspected; NOT clean holdout evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import random

from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc

VERSION = "NFL_TWO_WAY_MONEYLINE_ELO_RESEARCH_V1"
BASE = 1500.
K = 20.
HOME_ELO = 65.
OFFSEASON_RETENTION = 2. / 3.
WARMUP_START = 2014
TIE_PRIOR_LAST_YEAR = 2023
DIAGNOSTIC_YEARS = (2024, 2025)
ALIASES = {"OAK": "LV", "STL": "LA", "LAR": "LA", "SD": "LAC"}
NO_BET = "PASS"


def _team(code):
    return ALIASES.get(str(code), str(code))


def _utc(raw):
    if isinstance(raw, datetime):
        at = raw
    elif isinstance(raw, str):
        at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp is required")
    if at.tzinfo is None or at.utcoffset() != timedelta(0):
        raise ValueError("UTC-aware timestamp required")
    return at.astimezone(timezone.utc)


def american_implied(odds):
    if isinstance(odds, bool) or odds is None:
        return None
    x = float(odds)
    if not math.isfinite(x) or -100 < x < 100:
        return None
    return 100 / (x + 100) if x > 0 else -x / (100 - x)


def market_home_no_vig(home, away):
    a, b = american_implied(home), american_implied(away)
    if a is None or b is None or a + b < .9 or a + b > 1.35:
        return None
    return a / (a + b)


def elo_home_probability(delta):
    return 1. / (1. + 10 ** (-float(delta) / 400.))


def prior_tie_rate(rows):
    dev = [r for r in rows
           if int(r["season"]) <= TIE_PRIOR_LAST_YEAR
           and r.get("home_score") is not None and r.get("away_score") is not None
           and r.get("game_type") == "REG"]
    if len(dev) < 300:
        raise ValueError("need >=300 prior completed regular season games to estimate tie prior")
    ties = sum(float(g["home_score"]) == float(g["away_score"]) for g in dev)
    # Fixed half-count smoothing, 2014-23 only; no 2024+ score leakage.
    return (ties + .5) / (len(dev) + 1.), len(dev), ties


def pregame_elo(games, *, tie_rate):
    """Forecast every game in the whole group, then learn from the group."""
    if not 0 <= tie_rate < .10:
        raise ValueError("invalid training-era tie prior")
    groups = defaultdict(list)
    seen = set()
    for raw in games:
        if raw.get("game_type") != "REG" or int(raw["season"]) < WARMUP_START:
            continue
        gid = str(raw["game_id"])
        if not gid or gid in seen:
            raise ValueError("missing/duplicate schedule game")
        seen.add(gid)
        groups[(int(raw["season"]), int(raw["week"]))].append(raw)
    ratings = {}
    last_season = None
    forecast = []
    for (season, week), batch in sorted(groups.items()):
        if season != last_season:
            if last_season is not None:
                ratings = {team: BASE + (rating - BASE) * OFFSEASON_RETENTION
                           for team, rating in ratings.items()}
            last_season = season
        staging = []
        teams_this_week = set()
        for g in sorted(batch, key=lambda x: str(x["game_id"])):
            home, away = _team(g["home_team"]), _team(g["away_team"])
            if home == away or home in teams_this_week or away in teams_this_week:
                raise ValueError("duplicate team in season/week or invalid matchup")
            teams_this_week.update((home, away))
            neutral = str(g.get("location", "Home")).lower() == "neutral"
            delta = ratings.get(home, BASE) - ratings.get(away, BASE) + (
                0 if neutral else HOME_ELO)
            cond = elo_home_probability(delta)
            hp, ap = g.get("home_score"), g.get("away_score")
            kickoff = schedule_kickoff_utc(g)
            row = {
                "game_id": str(g["game_id"]), "season": season, "week": week,
                "home_team": home, "away_team": away, "neutral": neutral,
                "kickoff_utc": kickoff.isoformat() if kickoff else None,
                "p_home_win": (1-tie_rate) * cond,
                "p_away_win": (1-tie_rate) * (1-cond),
                "p_tie_push": tie_rate,
                "conditional_p_home_win_given_no_tie": cond,
                "elo_home_delta": round(delta, 6),
                "home_score": hp, "away_score": ap,
                "archive_home_moneyline": g.get("home_moneyline"),
                "archive_away_moneyline": g.get("away_moneyline"),
                "forecast_time_provenance": "REPLAY_WEEK_START_ORIGIN_UNAVAILABLE",
            }
            forecast.append(row)
            staging.append((home, away, hp, ap, cond))
        # A partly incomplete week must not update ratings for any of its
        # later games. This also means a Thursday final is not used to
        # secretly revise a Sunday prediction in the same week.
        if not all(h is not None and a is not None for _,_,h,a,_ in staging):
            continue
        for home, away, hp, ap, prior in staging:
            result = 1. if float(hp) > float(ap) else .0 if float(hp) < float(ap) else .5
            change = K * (result - prior)
            ratings[home] = ratings.get(home, BASE) + change
            ratings[away] = ratings.get(away, BASE) - change
    return forecast


def _score(rows):
    non_ties = [r for r in rows if r["home_score"] != r["away_score"]]
    def metric(subset, key):
        if not subset:
            return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
        y = [float(r["home_score"] > r["away_score"]) for r in subset]
        p = [min(1-1e-9, max(1e-9, float(r[key]))) for r in subset]
        return {"n":len(subset),
                "brier":round(sum((a-b)**2 for a,b in zip(y,p))/len(y),7),
                "log_loss":round(-sum(a*math.log(b)+(1-a)*math.log1p(-b)
                                   for a,b in zip(y,p))/len(y),7),
                "accuracy":round(sum((b>=.5)==bool(a)
                                      for a,b in zip(y,p))/len(y),7)}
    all_games = [r for r in rows if r["home_score"] is not None
                 and r["away_score"] is not None]
    benchmark = [dict(r, market_conditional_probability=market_home_no_vig(
        r["archive_home_moneyline"],r["archive_away_moneyline"]))
                 for r in non_ties]
    matched = [r for r in benchmark if r["market_conditional_probability"] is not None]
    diag = {"total_games": len(all_games),
            "ties": len(all_games) - len(non_ties),
            "model_conditional_non_tie":metric(non_ties,"conditional_p_home_win_given_no_tie"),
            "archive_novig_matched": metric(matched,"market_conditional_probability"),
            "model_same_archive_cohort":metric(matched,"conditional_p_home_win_given_no_tie"),
            "market_stage":"NFLVERSE_HISTORICAL_ARCHIVED_ODDS_ORIGIN_TIME_UNKNOWN",
            "not_an_account_quote":True,"not_pristine_holdout":True,
            "profitable_edge_proven":False,"decision":"PASS"}
    return diag


def freeze_future(rows, *, observed_utc, path, horizon_days=10):
    now = _utc(observed_utc)
    if horizon_days < 1 or horizon_days > 21:
        raise ValueError("invalid horizon")
    file = Path(path)
    current = []
    if file.exists():
        with file.open(encoding="utf8") as f:
            current = [json.loads(line) for line in f if line.strip()]
    existing = {x["game_id"] for x in current}
    if len(existing) != len(current):
        raise ValueError("original first-seen history has duplicate game")
    created = []
    for row in rows:
        if row["game_id"] in existing or row["season"] != 2026:
            continue
        if row["home_score"] is not None or row["away_score"] is not None:
            continue
        kickoff_text = row["kickoff_utc"]
        if not kickoff_text:
            continue
        kickoff = _utc(kickoff_text)
        if not now < kickoff <= now+timedelta(days=horizon_days):
            continue
        values = {k: row[k] for k in (
            "game_id","season","week","home_team","away_team","neutral",
            "kickoff_utc","p_home_win","p_away_win","p_tie_push",
            "conditional_p_home_win_given_no_tie","elo_home_delta",
        )}
        created.append({
            "model": VERSION, "first_observed_at_utc": now.isoformat(),
            "source": "NFLVERSE_SCHEDULES_OUTCOME_ONLY_ELO_PRIOR",
            **values,
            "price_source":None, "original_book_quote_time":None,
            "bookmaker_account_acceptance":False,
            "payout_estimated_ev":None, "decision":"PASS", "wager":False,
        })
    if created:
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("a",encoding="utf8") as f:
            for r in created:
                f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    return {"existing":len(current),"added":len(created),
            "total":len(current)+len(created),
            "new_games":[r["game_id"] for r in created]}


def research_report(rows, *, tie_training, tie_count, tie_rate):
    forecast = pregame_elo(rows,tie_rate=tie_rate)
    diagnostic = {}
    for season in DIAGNOSTIC_YEARS:
        diag = [r for r in forecast if r["season"] == season
                and r["home_score"] is not None and r["away_score"] is not None]
        diagnostic[str(season)] = _score(diag)
    return forecast,{
        "model":VERSION,"model_kind":"OUTCOME_ONLY_PUBLIC_ELO_NOT_BETTING_MODEL",
        "tie_prior_training_last_year":TIE_PRIOR_LAST_YEAR,
        "tie_prior_source_games":tie_training,
        "tie_prior_actual_ties":tie_count,
        "tie_prior":round(tie_rate,8),
        "elo_fixed_prior":{"K":K,"home_elo":HOME_ELO,"offseason_retention":OFFSEASON_RETENTION},
        "diagnostic_seasons_previous_research_inspected":list(DIAGNOSTIC_YEARS),
        "diagnostics":diagnostic,"replay_snapshots_have_origin_times":False,
        "archive_moneylines_are_user_executable_quotes":False,
        "independently_profitable_edge":False,"approved_bets":0,"decision":"PASS",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--season",type=int,default=2026)
    cli.add_argument("--horizon-days",type=int,default=10)
    cli.add_argument("--journal",type=Path,default=Path("history/moneyline_elo_v1_2026.jsonl"))
    cli.add_argument("--report",type=Path,default=Path("reports/moneyline_elo_research.json"))
    cli.add_argument("--refresh",action="store_true")
    args=cli.parse_args()
    if args.season != 2026:
        cli.error("original forward study is registered for 2026 only")
    frame = NFLDataClient().load_schedules(list(range(WARMUP_START,args.season+1)),refresh=args.refresh)
    games=frame.to_dicts()
    tie, n, count=prior_tie_rate(games)
    forecast,report=research_report(games,tie_training=n,tie_count=count,tie_rate=tie)
    report["forward"]=freeze_future(forecast,
        observed_utc=datetime.now(timezone.utc),path=args.journal,
        horizon_days=args.horizon_days)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"decision":report["decision"],"model":report["model"],
        "2024":report["diagnostics"]["2024"],"2025":report["diagnostics"]["2025"],
        "forward":report["forward"]},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
