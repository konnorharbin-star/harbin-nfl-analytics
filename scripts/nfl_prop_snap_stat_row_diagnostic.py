"""NFL snap-history-to-stat-row observation diagnostic.

The binary outcome is whether an eligible pre-week player appears in the
official nflverse game-stat table, NOT whether an offered sportsbook prop is
settled or void. MISSING stat row is not proven DNP; no real BetMGM odds.
PFR source copied retrospectively lacks original individual upload timestamps.

Fit 2024 only. Score 2025 (previously inspected) without refitting. Freeze a
separate 2026 first-seen source-row-proxy model (NOT a trading probability).
Never change existing v1/v2 2026 stats conditional forecast journals.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import nflreadpy as nfl

from nfl.data import NFLDataClient
from nfl.prop_challenger import forecast_week, source_frames
from scripts.participation_conditional_prop_v2 import DEFAULT_OPTIONS
from scripts.nfl_pfr_prior_snap_coverage import unique_crosswalk
from scripts.nfl_moneyline_elo_research import _team

VERSION="NFL_PFR_STAT_ROW_PRESENCE_RESEARCH_V1"
FEATURES=("intercept","prior_snap_mean_60","prior_snap_last_60",
          "prior_snap_observed_count_4","prior_snap_trend_60")
RIDGE=50.
DEVELOPMENT_YEAR=2024
DIAGNOSTIC_YEAR=2025
MIN_DEV=400
MIN_DIAG=200
BOOTSTRAPS=400
HORIZON_DAYS=10

def _utc(s):
    d=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if d.tzinfo is None or d.utcoffset()!=timedelta(0):
        raise ValueError("UTC timestamp required")
    return d.astimezone(timezone.utc)

def sigmoid(x):
    return 1./(1.+math.exp(-max(-25.,min(25.,x))))

def loss(y,p):
    p=max(1e-9,min(1-1e-9,float(p)))
    return -(y*math.log(p)+(1-y)*math.log1p(-p))

def _schedule_index(games):
    x=defaultdict(list)
    for g in games:
        if g["game_type"]!="REG":continue
        season=int(g["season"])
        for tm in (_team(g["home_team"]),_team(g["away_team"])):
            x[(season,tm)].append(g)
    for v in x.values():v.sort(key=lambda g:(int(g["week"]),str(g["game_id"])))
    return x

def make_cohorts(games,stats,raw_schedule,raw_snaps,crosswalk,*,season,
                 frozen_only_after_utc=None):
    # Reuse original eligibility rule, generated before reading this week's outcomes.
    completed={g["game_id"] for g in games
        if g["season"]==season and g["completed"]}
    weeks=sorted({g["week"] for g in games if g["season"]==season and g["week"]>=4})
    source_game=_schedule_index(raw_schedule)
    snaps={}
    source_coverage=defaultdict(int)
    for s in raw_snaps:
        if s.get("game_type")!="REG":continue
        gid=str(s.get("game_id") or "")
        t=_team(s.get("team") or "")
        pid=str(s.get("pfr_player_id") or "")
        if not gid or not t or not pid:continue
        k=(gid,t,pid)
        if k in snaps:raise ValueError("duplicated PFR snap entry")
        v=s.get("offense_snaps")
        z=float(v) if v is not None else None
        if z is not None and (not math.isfinite(z) or z<0 or z>150):
            raise ValueError("invalid source snap count")
        snaps[k]=z
        source_coverage[(gid,t)]+=1
    outcomes={(s["game_id"],s["player_id"]):s for s in stats}
    out=[]
    seen=set()
    count=defaultdict(int)
    now=_utc(frozen_only_after_utc) if frozen_only_after_utc else None
    for week in weeks:
        if now is not None:
            forecasts=forecast_week(games,stats,season=season,week=week,
                                   options=DEFAULT_OPTIONS,future_only_at=now)
        else:
            forecasts=forecast_week(games,stats,season=season,week=week,
                                   options=DEFAULT_OPTIONS)
        for p in forecasts:
            if now is None and p["game_id"] not in completed:continue
            if now is not None:
                ko=_utc(p["kickoff_utc"])
                if not now<ko<=now+timedelta(days=HORIZON_DAYS):continue
            unique=(p["game_id"],p["player_id"])
            if unique in seen:continue  # avoid multiple prop markets per player/game
            seen.add(unique)
            team=_team(p["team"])
            pid=crosswalk.get(str(p["player_id"]))
            if not pid:
                count["UNIQUE_GSIS_PFR_ID_UNAVAILABLE"]+=1
                continue
            history=[g for g in source_game[(season,team)]
                     if int(g["week"])<week
                     and g.get("home_score") is not None
                     and g.get("away_score") is not None][-4:]
            prior=[g for g in history if source_coverage.get((str(g["game_id"]),team),0)>0]
            if len(prior)<2:
                count["PRIOR_TEAM_SNAP_COVERAGE_TOO_SPARSE"]+=1
                continue
            # IMPORTANT: Missing PFR row is unknown, never zero snaps.
            seq=[snaps[(str(g["game_id"]),team,pid)]
                 for g in prior
                 if snaps.get((str(g["game_id"]),team,pid)) is not None]
            if len(seq)<2:
                count["PRIOR_PLAYER_SNAP_OBSERVATIONS_TOO_SPARSE"]+=1
                continue
            first,last=seq[0],seq[-1]
            mean=sum(seq)/len(seq)
            features=(1.,mean/60.,last/60.,len(seq)/4.,(last-first)/60.)
            if not all(math.isfinite(x) for x in features):
                raise ValueError("nonfinite as-of snap features")
            present=(unique in outcomes) if now is None else None
            out.append({
                "game_id":p["game_id"],"player_id":p["player_id"],
                "season":season,"week":week,"kickoff_utc":p["kickoff_utc"],
                "team":team,"player_name":p["player_name"],
                "observed_stats_row_presence":int(present) if present is not None else None,
                "features":features,"last_4_observed_snap_rows":len(seq),
                "feature_vintage":"PRIOR_COMPLETED_SEASON_WEEK_ONLY",
                "GSIS_to_PFR_unique":True,
                "eligibility_from_target_game_stats":False,
                "DNP_or_void_inferred":False,
            })
    out.sort(key=lambda r:(r["game_id"],r["player_id"]))
    return out,dict(count)

def fit(rows,*,ridge=RIDGE,min_dev=MIN_DEV):
    if len(rows)<min_dev:raise ValueError("insufficient prior-game 2024 development cohort")
    if any(r["season"]!=DEVELOPMENT_YEAR for r in rows):
        raise ValueError("fit season must be 2024")
    if any(r["observed_stats_row_presence"] not in (0,1) for r in rows):
        raise ValueError("missing/invalid development label")
    x=np.asarray([r["features"] for r in rows],dtype=float)
    y=np.asarray([r["observed_stats_row_presence"] for r in rows],dtype=float)
    if x.shape[1]!=len(FEATURES):raise ValueError("wrong feature width")
    prior=(sum(y)+1)/(len(y)+2)  # Beta(1,1) smoothed event rate baseline
    beta=np.array([math.log(prior/(1-prior))]+[0.]*(x.shape[1]-1))
    penalty=np.diag([0.]+[float(ridge)]*(x.shape[1]-1))
    for iteration in range(100):
        eta=np.clip(x@beta,-25,25)
        p=1/(1+np.exp(-eta))
        weight=np.maximum(p*(1-p),1e-8)
        step=np.linalg.solve(x.T@(x*weight[:,None])+penalty,
                             x.T@(p-y)+penalty@beta)
        beta-=step
        if np.max(np.abs(step))<1e-9:break
    if not np.all(np.isfinite(beta)):
        raise ValueError("nonfinite coefficients")
    return beta.tolist(),float(prior)

def predict(features,beta):
    if len(features)!=len(beta) or not all(math.isfinite(float(z)) for z in features):
        raise ValueError("invalid pregame feature vector")
    return sigmoid(sum(a*b for a,b in zip(features,beta)))

def _metrics(rows,beta,baseline):
    if not rows:return {"n":0}
    scores=[(r["game_id"],int(r["observed_stats_row_presence"]),
              predict(r["features"],beta)) for r in rows]
    n=len(scores)
    return {
        "n_player_games":n,"n_games":len({g for g,_,_ in scores}),
        "stats_row_observation_rate":round(sum(y for _,y,_ in scores)/n,7),
        "baseline_brier":round(sum((baseline-y)**2 for _,y,_ in scores)/n,8),
        "snap_brier":round(sum((p-y)**2 for _,y,p in scores)/n,8),
        "baseline_log_loss":round(sum(loss(y,baseline) for _,y,_ in scores)/n,8),
        "snap_log_loss":round(sum(loss(y,p) for _,y,p in scores)/n,8),
        "paired_brier_delta_snap_minus_baseline":round(
            sum((p-y)**2-(baseline-y)**2 for _,y,p in scores)/n,8),
        "snap_model_is_actual_DNP_or_void_probability":False,
        "account_BetMGM_price_verified":False,
    }

def cluster_interval(rows,beta,baseline,*,draws=BOOTSTRAPS,seed=2026):
    by_game=defaultdict(list)
    for r in rows:
        y=int(r["observed_stats_row_presence"])
        p=predict(r["features"],beta)
        by_game[r["game_id"]].append((p-y)**2-(baseline-y)**2)
    keys=sorted(by_game)
    if len(keys)<20:return None
    rng=random.Random(seed)
    sample=[]
    for _ in range(draws):
        chosen=[keys[rng.randrange(len(keys))] for _ in keys]
        diffs=[v for k in chosen for v in by_game[k]]
        sample.append(sum(diffs)/len(diffs))
    sample.sort()
    return [round(sample[int((draws-1)*.025)],8),
            round(sample[int((draws-1)*.975)],8)]

def freeze(rows,coefficients,*,observed_utc,path):
    at=_utc(observed_utc)
    path=Path(path)
    old=[json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    keys={(r["game_id"],r["player_id"]) for r in old}
    if len(keys)!=len(old):raise ValueError("duplicate original frozen stat-row-proxy")
    sha=hashlib.sha256(json.dumps(coefficients,sort_keys=True).encode()).hexdigest()
    fresh=[];known=set()
    for r in rows:
        key=(r["game_id"],r["player_id"])
        if key in keys:continue
        if key in known:raise ValueError("duplicate future candidate player-game")
        known.add(key)
        kickoff=_utc(r["kickoff_utc"])
        if not at<kickoff<=at+timedelta(days=HORIZON_DAYS):continue
        if r["observed_stats_row_presence"] is not None:
            raise ValueError("original future forecast accidentally contains target outcome")
        fresh.append({
            "version":VERSION,"first_observed_at_utc":at.isoformat(),
            "game_id":r["game_id"],"player_id":r["player_id"],
            "team":r["team"],"player_name":r["player_name"],
            "season":r["season"],"week":r["week"],
            "kickoff_utc":r["kickoff_utc"],
            "frozen_feature_vector":r["features"],
            "fit_beta_sha256":sha,
            "p_official_stat_row_present_NOT_DNP":predict(r["features"],coefficients),
            "p_player_active":None,
            "p_prop_void":None,
            "betmgm_account_price":None,"estimated_true_EV":None,
            "status":"RESEARCH_ONLY","wager":False,
        })
    if fresh:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a") as f:
            for r in fresh:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    return {"existing":len(old),"new":len(fresh),"total":len(old)+len(fresh)}

def run():
    nflclient=NFLDataClient()
    schedule=nflclient.load_schedules([2024,2025,2026],refresh=True)
    stats=nflclient.load_player_stats([2024,2025,2026],refresh=True)
    raw_snap=nfl.load_snap_counts([2024,2025,2026]).to_dicts()
    pfr_players=nfl.load_players().to_dicts()
    gsis_pfr=unique_crosswalk(pfr_players)
    games,player_rows,_=source_frames(schedule,stats)
    raw_schedules=schedule.to_dicts()
    dev,rejected_dev=make_cohorts(games,player_rows,raw_schedules,raw_snap,gsis_pfr,
                                   season=2024)
    beta,mean=fit(dev)
    holdout,rejected_holdout=make_cohorts(games,player_rows,raw_schedules,raw_snap,gsis_pfr,
                                         season=2025)
    if len(holdout)<MIN_DIAG:
        raise ValueError("insufficient 2025 diagnostic player-games")
    now=datetime.now(timezone.utc).isoformat()
    forward,rejected_forward=make_cohorts(games,player_rows,raw_schedules,raw_snap,gsis_pfr,
                           season=2026,frozen_only_after_utc=now)
    original=freeze(forward,beta,observed_utc=now,
        path=Path("history/player_prop_stat_row_proxy_v1_2026.jsonl"))
    diag=_metrics(holdout,beta,mean)
    ci=cluster_interval(holdout,beta,mean)
    report={
        "version":VERSION,"decision":"PASS","model_type":"OFFICIAL_STAT_ROW_PRESENCE_PROXY",
        "2024_train_player_games":len(dev),"2024_train_mean_rate":mean,
        "2024_rejected_feature_coverage":rejected_dev,
        "fitted_features":FEATURES,"fitted_beta":beta,"ridge":RIDGE,
        "2025_retrospective_diagnostic":diag,
        "2025_cluster_brier_delta_ci95":ci,
        "2025_previously_inspected_not_clean_profitability_holdout":True,
        "2025_rejected_feature_coverage":rejected_holdout,
        "2026_future_first_observed_utc":now,
        "2026_future_eligible_candidates":len(forward),
        "2026_rejected_feature_coverage":rejected_forward,
        "2026_forward_freeze":original,
        "source_historical_pfr_original_row_publication_times_known":False,
        "label_does_NOT_prove_played_or_void":True,
        "betmgm_bookmaker_offers_verified":0,
        "approved_profit_model":False,"approved_wagers":0,"true_EV":None,
        "original_full_prop_or_v2_journals_modified":False,
    }
    p=Path("reports/player_prop_stat_row_proxy_research.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return report

if __name__=="__main__":run()
