"""NFL moneyline challenger: prior-week point form, last-observed QB EPA, rest.

Two branches, both RESEARCH ONLY:
  1. Truly football-only logistic probability independent of sportsbook.
  2. Retrospective market-logit OFFSET challenger, comparing same archived
     moneyline cohort to market-only. The archive has UNKNOWN source time,
     therefore market-offset model is NOT deployable at live BetMGM prices.

2019-22 coefficients fixed in development, 2023+ chronologically evaluated.
Past 2023-25 results were inspected in this project: NO pristine edge test.
Every target NFL week is predicted BEFORE any of that week's scores or QB
stats are added; a partially unfinished week changes no team state.
Any last-observed QB is a PROXY, not a confirmed starting QB. NO bets.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc
from scripts.nfl_moneyline_elo_research import market_home_no_vig,prior_tie_rate

VERSION="NFL_PRIOR_WEEK_QB_EPA_MONEYLINE_CHALLENGER_V2_10_DAY_WINDOW"
DEVELOPMENT_YEARS=(2019,2020,2021,2022)
HISTORICAL_DIAGNOSTIC_YEARS=(2023,2024,2025)
FEATURES=("intercept","prior_margin_diff","last_qb_epa_diff",
          "qb_proxy_continuity_diff","rest_advantage","home_field")
RIDGE_ALPHA=75.0
MIN_TRAIN_GAMES=700
TEAM_RECENT=6
QB_RECENT=5
P_FLOOR=1e-5
ALIASES={"OAK":"LV","STL":"LA","LAR":"LA","SD":"LAC"}

def team(name):
    return ALIASES.get(str(name),str(name))

def utc(value):
    if isinstance(value,datetime):r=value
    elif isinstance(value,str):r=datetime.fromisoformat(value.replace("Z","+00:00"))
    else:raise ValueError("UTC timestamp required")
    if r.tzinfo is None or r.utcoffset()!=timedelta(0):
        raise ValueError("UTC timezone required")
    return r.astimezone(timezone.utc)

def logit(x):
    x=max(P_FLOOR,min(1-P_FLOOR,float(x)))
    return math.log(x/(1-x))

def logistic(x):
    return 1/(1+math.exp(-max(-30,min(30,float(x)))))

def num(v,default=None):
    if v is None:return default
    try:
        z=float(v)
        return z if math.isfinite(z) else default
    except (ValueError,TypeError,OverflowError):
        return default

def qb_week_candidates(stats):
    """Only source-available completed weekly NFL GSIS player rows (no roster inference)."""
    ans=defaultdict(list)
    seen=set()
    for r in stats:
        if r.get("season_type")!="REG" or r.get("position")!="QB":
            continue
        key=(int(r["season"]),int(r["week"]),str(r.get("game_id")),team(r["team"]))
        pid=str(r.get("player_id") or "")
        if not pid or (key,pid) in seen:
            continue
        seen.add((key,pid))
        att=num(r.get("attempts"),0.)
        sacks=num(r.get("sacks_suffered"),0.)
        epa=num(r.get("passing_epa"))
        if att<0 or sacks<0 or (att+sacks)<1 or epa is None:
            continue
        ans[key].append((att+sacks,att,pid,epa))
    return {k:max(v,key=lambda a:(a[0],a[1],a[2])) for k,v in ans.items()}

def prior_features(schedules,stats):
    """Never use a target week's final scores/QB records to predict its games."""
    qbs=qb_week_candidates(stats)
    groups=defaultdict(list)
    seen=set()
    for g in schedules:
        if g.get("game_type")!="REG":
            continue
        yr=int(g["season"])
        if yr<2014 or yr>2026:continue
        gid=str(g["game_id"])
        if not gid or gid in seen:raise ValueError("duplicate/blank game ID")
        seen.add(gid)
        groups[(yr,int(g["week"]))].append(g)
    margin=defaultdict(list)
    qb_history=defaultdict(list)
    records=[]
    previous_year=None
    for (year,week),games in sorted(groups.items()):
        if year!=previous_year:
            # Nothing from a new season's results is available before Week 1.
            # Last-season QB availability is not authenticated, so fail closed
            # rather than transplant it to a potentially new starter.
            margin=defaultdict(list)
            qb_history=defaultdict(list)
            previous_year=year
        staged=[]
        appeared=set()
        for g in sorted(games,key=lambda x:str(x["game_id"])):
            h,a=team(g["home_team"]),team(g["away_team"])
            if h==a or h in appeared or a in appeared:
                raise ValueError("team appears twice in a season/week")
            appeared.update((h,a))
            def margin_prior(t):
                return sum(margin[t][-TEAM_RECENT:])/max(1,len(margin[t][-TEAM_RECENT:]))
            def qb_stats_prior(t):
                q=qb_history[t][-QB_RECENT:]
                if not q:return 0.,0.,False
                denom=sum(z[1] for z in q)
                recent_epa=(sum(z[2] for z in q)/denom) if denom else 0.
                continuity=float(len(q)>=2 and q[-1][0]==q[-2][0])
                return recent_epa,continuity,True
            hep,hcont,hhas=qb_stats_prior(h)
            aep,acont,ahas=qb_stats_prior(a)
            rest_h=num(g.get("home_rest"))
            rest_a=num(g.get("away_rest"))
            venue=0. if str(g.get("location","Home")).lower()=="neutral" else 1.
            vec=[
                1.,
                max(-3.,min(3.,(margin_prior(h)-margin_prior(a))/14.)),
                max(-3.,min(3.,(hep-aep)/.35)),
                hcont-acont,
                max(-2.,min(2.,(rest_h-rest_a)/7.)) if rest_h is not None and rest_a is not None else 0.,
                venue,
            ]
            if len(vec)!=len(FEATURES) or not all(math.isfinite(x) for x in vec):
                raise ValueError("invalid original pre-week football features")
            kickoff=schedule_kickoff_utc(g)
            row={
                "game_id":str(g["game_id"]),"season":year,"week":week,
                "home":h,"away":a,"kickoff_utc":kickoff.isoformat() if kickoff else None,
                "features":vec,"home_qb_proxy_known":hhas,
                "away_qb_proxy_known":ahas,
                "player_starter_status_verified":False,
                "home_score":g.get("home_score"),"away_score":g.get("away_score"),
                "archive_home_moneyline":g.get("home_moneyline"),
                "archive_away_moneyline":g.get("away_moneyline"),
                "feature_rule":"PRE_WEEK_ONLY_NO_INTRA_WEEK_UPDATES",
            }
            records.append(row)
            staged.append((g,h,a))
        # Critical: all week outcomes and QB stats hidden until that week
        # has finished (avoid Thurs/Sun same-week historical leakage).
        if any(g.get("home_score") is None or g.get("away_score") is None for g,_,_ in staged):
            continue
        for g,h,a in staged:
            hs=float(g["home_score"]);aws=float(g["away_score"])
            margin[h].append(hs-aws);margin[a].append(aws-hs)
            for t in (h,a):
                key=(year,week,str(g["game_id"]),t)
                candidate=qbs.get(key)
                if candidate is not None:
                    dropbacks,_,pid,total_epa=candidate
                    qb_history[t].append((pid,dropbacks,total_epa))
    return records

def fit(rows,*,use_market_offset,alpha=RIDGE_ALPHA):
    training=[r for r in rows if r["season"] in DEVELOPMENT_YEARS
              and r["home_score"] is not None and r["away_score"] is not None
              and float(r["home_score"])!=float(r["away_score"])
              and (not use_market_offset or market_home_no_vig(
                  r["archive_home_moneyline"],r["archive_away_moneyline"]) is not None)]
    if len(training)<MIN_TRAIN_GAMES:
        raise ValueError("insufficient original pre-2023 development games")
    x=np.asarray([r["features"] for r in training],dtype=float)
    y=np.asarray([float(float(r["home_score"])>float(r["away_score"])) for r in training])
    if x.shape[1]!=len(FEATURES):raise ValueError("wrong feature vector width")
    offsets=np.asarray([logit(market_home_no_vig(
        r["archive_home_moneyline"],r["archive_away_moneyline"]))
        if use_market_offset else 0. for r in training])
    beta=np.zeros(x.shape[1],dtype=float)
    # Shared, PREDECLARED ridge shrinkage; preserve the free intercept.
    penalty=np.diag([0.]+[float(alpha)]*(x.shape[1]-1))
    for iteration in range(80):
        eta=np.clip(offsets+x@beta,-30.,30.)
        p=1/(1+np.exp(-eta))
        weight=np.maximum(p*(1-p),1e-8)
        gradient=x.T@(p-y)+penalty@beta
        hessian=x.T@(x*weight[:,None])+penalty
        step=np.linalg.solve(hessian,gradient)
        beta-=step
        if float(np.max(np.abs(step)))<1e-9:break
    if not np.all(np.isfinite(beta)):
        raise ValueError("nonfinite calibrated feature coefficients")
    return beta.tolist(),len(training)

def evaluate(rows,beta,*,market_offset):
    records=[]
    for r in rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        h=float(r["home_score"]);a=float(r["away_score"])
        if h==a:continue
        market=market_home_no_vig(r["archive_home_moneyline"],r["archive_away_moneyline"])
        if market is None:
            continue
        offset=logit(market) if market_offset else 0.
        pred=logistic(offset+sum(b*x for b,x in zip(beta,r["features"])))
        records.append((int(h>a),pred,market))
    if not records:return {"n":0}
    n=len(records)
    def summarize(j):
        brier=sum((r[j]-r[0])**2 for r in records)/n
        ll=sum(-(r[0]*math.log(r[j])+(1-r[0])*math.log1p(-r[j])) for r in records)/n
        return {"brier":round(brier,8),"log_loss":round(ll,8)}
    m=summarize(1);b=summarize(2)
    return {"n":n,"candidate":m,"archived_market_reference":b,
        "candidate_minus_market_brier":round(m["brier"]-b["brier"],8),
        "candidate_minus_market_log_loss":round(m["log_loss"]-b["log_loss"],8),
        "market_prices_account_accepted":0,"market_quote_origin_verified":0,
        "approved_profit_edge":False}

def run(*,schedules,players):
    rows=prior_features(schedules,players)
    independent,n=fit(rows,use_market_offset=False)
    market_resid,n2=fit(rows,use_market_offset=True)
    report={
        "version":VERSION,"decision":"PASS",
        "features":FEATURES,"alpha":RIDGE_ALPHA,
        "development_years":DEVELOPMENT_YEARS,
        "n_independent_development":n,"n_market_offset_development":n2,
        "independent_coefficients":independent,
        "market_offset_coefficients":market_resid,
        "diagnostic_years_already_inspected":HISTORICAL_DIAGNOSTIC_YEARS,
        "diagnostics":{},
        "market_offset_only_archival":True,
        "2023_2025_is_not_pristine_holdout":True,
        "no_contemporaneous_BetMGM_quote":True,
        "QB_last_observed_proxy_NOT_verified_starter":True,
        "betting_EV":None,"recommended_wagers":[],"approved_profitable_model":False,
    }
    for yr in HISTORICAL_DIAGNOSTIC_YEARS:
        group=[r for r in rows if r["season"]==yr]
        report["diagnostics"][str(yr)]={
            "market_free_challenger":evaluate(group,independent,market_offset=False),
            "archived_market_plus_QB_challenger":evaluate(group,market_resid,market_offset=True)}
    return rows,report

def freeze_first_seen(rows,*,coefficients,tie_prior,observed_utc,path,horizon_days=10):
    if not 0<horizon_days<=21:raise ValueError("future horizon must be between 1 and 21 days")
    at=utc(observed_utc)
    path=Path(path)
    existing=[]
    if path.exists():
        existing=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    old={x["game_id"] for x in existing}
    if len(old)!=len(existing):raise ValueError("duplicate immutable candidate game")
    new=[]
    coeff_hash=hashlib.sha256(json.dumps(coefficients,separators=(",",":")).encode()).hexdigest()
    for r in rows:
        ko=utc(r["kickoff_utc"]) if r.get("kickoff_utc") else None
        if r["season"]!=2026 or r["game_id"] in old or ko is None or not at<ko<=at+timedelta(days=horizon_days):
            continue
        if r["home_score"] is not None or r["away_score"] is not None:
            continue
        z=sum(c*f for c,f in zip(coefficients,r["features"]))
        cond=logistic(z)
        frozen={
            "model":VERSION,"first_observed_at_utc":at.isoformat(),
            "game_id":r["game_id"],"season":r["season"],"week":r["week"],
            "kickoff_utc":r["kickoff_utc"],"home":r["home"],"away":r["away"],
            "frozen_asof_football_feature_vector":r["features"],
            "football_feature_names":FEATURES,
            "coefficient_sha256":coeff_hash,
            "home_win_no_tie_conditional":cond,
            "p_home_win":cond*(1-tie_prior),
            "p_away_win":(1-cond)*(1-tie_prior),
            "p_tie_push":tie_prior,
            "QB_proxy_status":"LAST_OBSERVED_NOT_VERIFIED_STARTER",
            "market_odds":None,"BetMGM_US_offer_original_time":None,
            "bet_true_EV":None,"wager":False,"decision":"PASS",
        }
        new.append(frozen)
    if new:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf8") as f:
            for row in new:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    return {"original_existing":len(existing),"frozen_new":len(new),
            "total":len(existing)+len(new),"first_seen_game_ids":[r["game_id"] for r in new]}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--journal",type=Path,default=Path("history/moneyline_weekly_qb_challenger_v2_2026.jsonl"))
    ap.add_argument("--report",type=Path,default=Path("reports/moneyline_weekly_qb_challenger.json"))
    a=ap.parse_args()
    if a.season!=2026:ap.error("registered prospective study requires 2026")
    data=NFLDataClient()
    seasons=list(range(2014,2027))
    schedules=data.load_schedules(seasons,refresh=True).to_dicts()
    players=data.load_player_stats(list(range(2019,2027)),refresh=True).to_dicts()
    rows,report=run(schedules=schedules,players=players)
    tie,tn,tt=prior_tie_rate(schedules)
    report["forward_window_days"]=10
    report["v1_full_schedule_194_originals_retained_as_long_horizon_not_next_week_forecasts"]=True
    report["tie_prior_2014_2023"]=tie
    report["tie_prior_dev_games"]=tn
    report["tie_prior_dev_ties"]=tt
    report["forward"]=freeze_first_seen(rows,
        coefficients=report["independent_coefficients"],tie_prior=tie,
        observed_utc=datetime.now(timezone.utc).isoformat(),path=a.journal,horizon_days=10)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"model":VERSION,"decision":report["decision"],
         "development_games":report["n_independent_development"],
         "market_offset_development":report["n_market_offset_development"],
         "diagnostics":report["diagnostics"],"forward":report["forward"]},
         indent=2,sort_keys=True))
if __name__=="__main__":main()
