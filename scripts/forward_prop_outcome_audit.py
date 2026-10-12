"""Score genuinely frozen NFL prop probabilities against *later* official stat rows.

Research-only: no account bet or financial ROI. Does not label a missing
player-stat row as zero; player may be inactive/void. Grade only games with
final scores known after kickoff and official stats. A player/game with many
synthetic thresholds contributes ONE cluster to cohort uncertainty.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.professional_method_audit import _utc
from nfl.pregame import schedule_kickoff_utc

VERSION="NFL_FROZEN_PROP_FORWARD_SCORING_V1"
MARKET_FIELDS={
    "receiving_yards":"receiving_yards",
    "rushing_yards":"rushing_yards",
    "receptions":"receptions",
    "passing_yards":"passing_yards",
}
REQUIRED_FORECAST={"game_id","season","week","kickoff_utc","team","player_id",
                   "market","synthetic_line","probability_over",
                   "probability_under","first_observed_at_utc"}
REQUIRED_STAT={"season","week","player_id"}


def _number(value,name):
    if isinstance(value,bool):
        raise ValueError("boolean is not "+name)
    try: n=float(value)
    except (TypeError,ValueError):raise ValueError("invalid "+name) from None
    if not math.isfinite(n):raise ValueError("nonfinite "+name)
    return n


def _asgame(rows):
    games={}
    for row in rows:
        event=str(row.get("game_id",""))
        if not event or event in games:
            raise ValueError("duplicate/blank official game identifier")
        games[event]=row
    return games


def _stats(rows):
    stats={}
    for row in rows:
        if any(k not in row for k in REQUIRED_STAT):
            raise ValueError("official player stats missing temporal keys")
        # nflverse's current weekly stats use 'team'; older compatible
        # extracts sometimes use 'recent_team'. Accept either only when
        # unambiguous, and confirm event game_id downstream if supplied.
        teams=[str(row[k]) for k in ("team","recent_team")
               if row.get(k) not in (None,"")]
        if not teams:
            raise ValueError("official player stats missing team identity")
        if len(set(teams))!=1:
            raise ValueError("conflicting official team identities")
        key=(int(row["season"]),int(row["week"]),teams[0],
             str(row["player_id"]))
        if key in stats: raise ValueError("duplicate player game stats "+str(key))
        stats[key]=row
    return stats


def grade(forecasts, games, stats, *, captured_utc, previous=None):
    """Return newly graded first-seen threshold records and audit metadata."""
    observed=_utc(captured_utc)
    games=_asgame(games)
    player_stats=_stats(stats)
    already={r["forecast_id"] for r in (previous or [])}
    first_seen=set()
    scored=[]
    pending=defaultdict(int)
    for f in forecasts:
        if any(k not in f or f[k] is None or f[k]=="" for k in REQUIRED_FORECAST):
            raise ValueError("incomplete frozen forecast")
        event=f["game_id"]
        if f["market"] not in MARKET_FIELDS:
            pending["UNSUPPORTED_MARKET"]+=1
            continue
        line=_number(f["synthetic_line"],"synthetic threshold")
        if not math.isclose(line%1,0.5,abs_tol=1e-9):
            raise ValueError("zero-push stored forecast unexpectedly has integer threshold")
        p=_number(f["probability_over"],"probability")
        under=_number(f["probability_under"],"under probability")
        if not (0<=p<=1 and 0<=under<=1 and
                math.isclose(p+under,1,abs_tol=1e-6)):
            raise ValueError("bad frozen probability")
        kickoff=_utc(f["kickoff_utc"])
        if _utc(f["first_observed_at_utc"])>=kickoff:
            raise ValueError("forecast not actually frozen before kickoff")
        # Float formatting (49.50 = 49.5) should not create duplicate rows.
        unique=(event,str(f["player_id"]),str(f["market"]),line)
        if unique in first_seen:
            raise ValueError("duplicate original player threshold forecast")
        first_seen.add(unique)
        fid=hashlib.sha256(
            json.dumps(unique,separators=(",",":")).encode()).hexdigest()[:24]
        if fid in already:
            pending["ALREADY_GRADED"]+=1
            continue
        game=games.get(event)
        if game is None:
            pending["GAME_ABSENT_IN_SCHEDULE"]+=1
            continue
        if str(game.get("game_type"))!="REG":
            pending["NON_REGULAR_EVENT"]+=1
            continue
        if (str(game.get("season"))!=str(f["season"]) or
            str(game.get("week"))!=str(f["week"])):
            raise ValueError("forecast/game season week mismatch")
        official_kick=schedule_kickoff_utc(game)
        if official_kick is None or official_kick!=kickoff:
            pending["KICKOFF_CHANGED_OR_UNVERIFIED"]+=1
            continue
        if str(f["team"]) not in (str(game.get("home_team")),str(game.get("away_team"))):
            raise ValueError("forecast player team does not play in specified event")
        if kickoff+timedelta(hours=3)>observed:
            pending["INSUFFICIENT_TIME_AFTER_KICKOFF"]+=1
            continue
        if game.get("home_score") is None or game.get("away_score") is None:
            pending["GAME_NOT_FINAL"]+=1
            continue
        key=(int(f["season"]),int(f["week"]),str(f["team"]),str(f["player_id"]))
        player=player_stats.get(key)
        if player is None:
            # Missing is NOT equal to zero: DNP/no-snap and sportsbook void
            # treatment are contract-specific.
            pending["MISSING_PLAYER_STATS_OR_DNP"]+=1
            continue
        if player.get("game_id") not in (None,"",event):
            pending["PLAYER_GAME_ID_MISMATCH"]+=1
            continue
        if player.get("season_type") not in (None,"","REG"):
            pending["PLAYER_NOT_REGULAR_SEASON"]+=1
            continue
        val=player.get(MARKET_FIELDS[f["market"]])
        if val is None:
            pending["STAT_FIELD_UNAVAILABLE"]+=1
            continue
        actual=_number(val,"official "+f["market"])
        if actual!=int(actual):
            raise ValueError("non-integer NFL official stat outcome")
        outcome=int(actual>line)
        row={
            "version":VERSION,"record_type":"FIRST_SETTLED_FORECAST_SCORE",
            "forecast_id":fid,"event_id":event,
            "season":int(f["season"]),"week":int(f["week"]),
            "team":str(f["team"]),"player_id":str(f["player_id"]),
            "market":f["market"],"line":line,
            "first_forecast_utc":_utc(f["first_observed_at_utc"]).isoformat(),
            "kickoff_utc":kickoff.isoformat(),
            "first_scored_observed_utc":observed.isoformat(),
            "p_over":p,"actual_stat":int(actual),"actual_over":outcome,
            "brier_over":round((p-outcome)**2,10),
            "actual_bet_placed":False,
            "verified_betmgm_quote":False,
            "betting_roi":None,
        }
        scored.append(row)
    groups=defaultdict(list)
    for row in scored:
        groups[(row["event_id"],row["player_id"],row["market"])].append(
            row["brier_over"]
        )
    market_groups=defaultdict(list)
    for (event,player,market),briers in groups.items():
        market_groups[market].append(sum(briers)/len(briers))
    summary={
        name:{
            "independent_player_game_clusters":len(values),
            "graded_correlated_threshold_rows":sum(
                1 for s in scored if s["market"]==name),
            "equal_weight_player_game_brier":round(sum(values)/len(values),8),
            "market_odds_baseline":None,
            "valid_2026_prospective_roi_estimate":False,
        } for name,values in sorted(market_groups.items())
    }
    return scored,{
        "version":VERSION,"decision":"PASS",
        "graded_new_threshold_rows":len(scored),
        "new_independent_player_game_clusters":len(groups),
        "ungraded_reason_counts":dict(sorted(pending.items())),
        "score_metrics_by_market":summary,
        "independently_validated_betmgm_positive_ev":False,
        "any_account_wager_verified":False,
        "status":"SCORED_RESEARCH" if scored else "NO_FINAL_RESULT_ELIGIBLE",
        "note":"Thresholds for same player/game correlated; use player-game clusters, no bookmaker ROI without timestamped odds.",
    }


def read_csv(path):
    with Path(path).open(newline="",encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def append_results(path,entries):
    """Persist graded future outcomes, never overwrite original first score."""
    path=Path(path)
    existing=[]
    if path.exists():
        existing=[json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    seen={r["forecast_id"] for r in existing}
    if len(seen)!=len(existing): raise ValueError("duplicate prior scored forecast")
    if any(x["forecast_id"] in seen for x in entries):
        raise ValueError("cannot overwrite an earlier scored forecast")
    if len({x["forecast_id"] for x in entries})!=len(entries):
        raise ValueError("duplicated new result identifier")
    if entries:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf8") as stream:
            for row in entries:
                stream.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    return len(existing)+len(entries)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--source",type=Path,default=Path("history/player_prop_probability_forward.csv"))
    p.add_argument("--history",type=Path,default=Path("history/player_prop_outcomes_2026.jsonl"))
    p.add_argument("--report",type=Path,default=Path("reports/forward_prop_outcome_audit.json"))
    a=p.parse_args()
    from nfl.data import NFLDataClient
    client=NFLDataClient()
    captured=datetime.now(timezone.utc).isoformat()
    source=read_csv(a.source)
    try:
        schedule=client.load_schedules([a.season],refresh=True).to_dicts()
        stats=client.load_player_stats([a.season],refresh=True).to_dicts()
        failures=[]
    except Exception as error:
        schedule=[];stats=[]
        failures=[str(type(error).__name__)+": "+str(error)[:250]]
    old=[]
    if a.history.exists():
        old=[json.loads(x) for x in a.history.read_text(encoding="utf8").splitlines() if x.strip()]
    new,report=grade(source,schedule,stats,captured_utc=captured,previous=old)
    report["source_errors"]=failures
    report["captured_utc"]=captured
    report["total_first_seen_rows_in_source"]=len(source)
    report["total_previously_graded"]=len(old)
    report["newly_persisted"]=len(new)
    if new:append_results(a.history,new)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
