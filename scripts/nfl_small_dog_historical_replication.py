"""Free nflverse historical falsification for published NFL small-spread dogs.

This is historical line/odds archive analysis, not historical account-executable
sportsbook offers and not prospective first-seen forecasts. Never use archive
odds to prove a real executable expected value.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import UTC,datetime
import argparse,json,random
from pathlib import Path

SOURCE="https://github.com/nflverse/nfldata/blob/master/data/games.csv"
PAPER="https://doi.org/10.5750/jpm.v19i2.2325"
SEASONS=(2022,2023,2024,2025)
SEED=20261013

def parse(games,years=SEASONS):
    out=[]
    for x in games:
        if (x.get("game_type")!="REG" or x.get("season") not in years
            or x.get("home_score") is None or x.get("away_score") is None):
            continue
        sp=x.get("spread_line")
        h=x.get("home_moneyline");a=x.get("away_moneyline")
        if any(v is None for v in (sp,h,a)):continue
        try:
            sp,h,a=float(sp),float(h),float(a)
            home,away=int(x["home_score"]),int(x["away_score"])
        except (TypeError,ValueError):continue
        if home==away or sp<=0:continue
        if h>0 and a<0:
            side="home";odds=h
        elif a>0 and h<0:
            side="away";odds=a
        else:continue
        if abs(odds-round(odds))>1e-9:continue
        won=(home>away)==(side=="home")
        out.append({
            "game_id":x["game_id"],"season":int(x["season"]),
            "week":int(x["week"]),"point_spread":sp,
            "underdog_side":side,"american_price":int(odds),
            "outcome":"win" if won else "loss",
            "hypothetical_net_units":odds/100 if won else -1,
        })
    unique={(r["season"],r["game_id"]) for r in out}
    if len(unique)!=len(out):
        raise ValueError("DUPLICATE_ARCHIVED_NFL_GAME")
    return out

def report(rows,*,seed=SEED,replications=5000):
    n=len(rows);wins=sum(r["outcome"]=="win" for r in rows)
    if not n:return {"games":0,"roi":None,"interval95":None}
    net=sum(r["hypothetical_net_units"] for r in rows)
    groups=defaultdict(list)
    for r in rows:groups[(r["season"],r["week"])].append(r["hypothetical_net_units"])
    groups=list(groups.values())
    if len(groups)<8:return {"games":n,"wins":wins,"units":net,"roi":net/n,
                             "interval95":None,"week_clusters":len(groups)}
    rng=random.Random(seed)
    vals=[]
    for _ in range(replications):
        total=0.;count=0
        for ignored in groups:
            sample=groups[rng.randrange(len(groups))]
            total+=sum(sample);count+=len(sample)
        vals.append(total/count)
    vals.sort()
    return {"games":n,"wins":wins,"units":net,"roi":net/n,
            "week_clusters":len(groups),
            "week_cluster_bootstrap95":[vals[int(.025*replications)],
                                        vals[int(.975*replications)-1]],
            "independent_executable_bookmaker_quotes_confirmed":False,
            "betting_authorized":False}

def analyze(games):
    observations=parse(games)
    cohorts={"spread_le_3":[r for r in observations if 0<r["point_spread"]<=3],
             "spread_gt_3_le_7":[r for r in observations if 3<r["point_spread"]<=7]}
    years={str(yr):{group:report([r for r in selected if r["season"]==yr])
                    for group,selected in cohorts.items()} for yr in SEASONS}
    return {
      "spec":"PUBLISHED_NFL_SMALL_SPREAD_UNDERDOG_FALSIFICATION_V1",
      "published_research":PAPER,
      "source":SOURCE,
      "historical_source_price_offer_time_authenticated":False,
      "bookmaker_and_account_executable_offer_verified":False,
      "season_window":list(SEASONS),
      "rules":{"main":"0 < spread <=3","comparison":"3 < spread <=7",
               "underdog_must_have_positive_moneyline":True,
               "fixed_unit_stake":1},
      "total":{key:report(vals) for key,vals in cohorts.items()},
      "by_season":years,
      "later_years_known_to_researchers_not_untouched_forward_confirmation":True,
      "economic_betting_edge_validated":False,
      "betting_authorized":False,
      "action":"PASS_RESEARCH_ONLY",
    }

def main():
    import nflreadpy as nfl
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("reports/nfl_small_dog_historical_replication.json"))
    args=p.parse_args()
    result=analyze(nfl.load_schedules(list(SEASONS)).to_dicts())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"reproduction":result["spec"],"roi":result["total"],
        "economic_betting_edge_validated":False},sort_keys=True))
if __name__=="__main__":main()
