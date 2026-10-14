"""Free nflverse historical falsification for published NFL small-spread dogs.

This is historical line/odds archive analysis, not historical account-executable
sportsbook offers and not prospective first-seen forecasts. Never use archive
odds to prove a real executable expected value.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import UTC,datetime
import argparse,json,random,math,hashlib,csv,io
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
            home,away=float(x["home_score"]),float(x["away_score"])
        except (TypeError,ValueError):continue
        if (any(not math.isfinite(v) for v in (sp,h,a,home,away))
            or min(home,away)<0 or not home.is_integer() or not away.is_integer()
            or sp==0 or not 1<=int(x["week"])<=18):continue
        # nflverse positive spread means HOME favorite, not absolute spread.
        # Include home underdogs as well as away underdogs using magnitude.
        side="away" if sp>0 else "home"
        odds=a if side=="away" else h
        favorite=h if side=="away" else a
        if odds<100 or favorite>-100 or odds<=favorite:continue
        if abs(odds-round(odds))>1e-9:continue
        push=home==away
        won=not push and ((home>away)==(side=="home"))
        out.append({
            "game_id":x["game_id"],"season":int(x["season"]),
            "week":int(x["week"]),"point_spread":abs(sp),
            "underdog_side":side,"american_price":int(odds),
            "outcome":"push" if push else "win" if won else "loss",
            "hypothetical_net_units":0 if push else odds/100 if won else -1,
        })
    unique={(r["season"],r["game_id"]) for r in out}
    if len(unique)!=len(out):
        raise ValueError("DUPLICATE_ARCHIVED_NFL_GAME")
    return out

def report(rows,*,seed=SEED,replications=5000):
    n=len(rows);wins=sum(r["outcome"]=="win" for r in rows)
    pushes=sum(r["outcome"]=="push" for r in rows)
    if not n:return {"games":0,"roi":None,"interval95":None}
    net=sum(r["hypothetical_net_units"] for r in rows)
    groups=defaultdict(list)
    for r in rows:groups[(r["season"],r["week"])].append(r["hypothetical_net_units"])
    groups=list(groups.values())
    if len(groups)<8:return {"games":n,"wins":wins,"pushes":pushes,"units":net,"roi":net/n,
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
    return {"games":n,"wins":wins,"pushes":pushes,"units":net,"roi":net/n,
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
               "spread_is_absolute_magnitude":True,
               "home_and_away_underdogs_included":True,
               "tie_return_assumed_stake_refund_not_verified_book_rule":True,
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
    p.add_argument("--source-csv",type=Path,help="Replay the preserved research source subset without network")
    args=p.parse_args()
    fields=("game_id","season","week","game_type","home_team","away_team",
            "spread_line","home_moneyline","away_moneyline","home_score","away_score")
    if args.source_csv:
        with args.source_csv.open(newline="") as handle:
            source_rows=list(csv.DictReader(handle))
        for row in source_rows:
            for key in ("season","week"):
                row[key]=int(row[key])
            for key in fields[6:]:
                row[key]=float(row[key]) if row[key] else None
    else:
        source_rows=nfl.load_schedules(list(SEASONS)).to_dicts()
    result=analyze(source_rows)
    handle=io.StringIO(newline="")
    writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    writer.writerows(source_rows)
    serialized=args.source_csv.read_text() if args.source_csv else handle.getvalue()
    result["source_capture_utc"]=datetime.now(UTC).isoformat()
    result["source_schedule_rows"]=len(source_rows)
    result["source_input_snapshot_sha256"]=hashlib.sha256(serialized.encode()).hexdigest()
    result["source_input_snapshot_path"]="research/nfl_small_dog_source_2022_2025.csv"
    source_path=Path(result["source_input_snapshot_path"])
    source_path.parent.mkdir(parents=True,exist_ok=True)
    if not args.source_csv:
        source_path.write_text(serialized,encoding="utf-8")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"reproduction":result["spec"],"roi":result["total"],
        "economic_betting_edge_validated":False},sort_keys=True))
if __name__=="__main__":main()
