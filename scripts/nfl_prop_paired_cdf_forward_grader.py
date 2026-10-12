"""2026 prospective *conditional* NFL player-stat OVER probability study.

Freeze exact matched synthetic half-point lines for ORIGINAL 2024 player CDF
and 2024 prior-snap bucket challenger BEFORE each 2026 kickoff. After >=72h
and independently checked complete game/team source coverage, grade only
players with an explicit official matched game stat row. Missing row is NOT
zero, under, inactive, DNP, void or a profitable bettor's settlement.

Both CDFs were built before 2026 outcomes; no model refresh or hindsight
reconstruction is allowed after the first paired forecast. Prices are NONE:
all rows RESEARCH_ONLY / PASS and no sportsbook wagering.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import random

from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc
from nfl.prop_probability import PROP_THRESHOLD_GRID,_scale,over_probability
from scripts.nfl_player_prop_full_distribution import (
    VERSION as FULL_MODEL,hash_object,
)
from scripts.nfl_prop_snap_conditional_cdf import (
    VERSION as SNAP_MODEL,bucket
)
from scripts.nfl_moneyline_elo_research import _team

VERSION="NFL_2026_ORIGINAL_VS_SNAP_PROP_CONDITIONAL_CDF_FORWARD_V1"
MATURE_HOURS=72
MIN_PER_TEAM=4
MIN_COMBINED=12
BOOTSTRAPS=400
FAMILYWISE_MARKETS=4
HORIZON_DAYS=10
FROZEN_PATH=Path("history/player_prop_paired_cdf_lines_v1_2026.jsonl")
GRADE_PATH=Path("history/player_prop_paired_cdf_line_grades_v1_2026.jsonl")
REPORT_PATH=Path("reports/player_prop_paired_cdf_forward_2026.json")


def utc(s):
    d=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if d.tzinfo is None or d.utcoffset()!=timedelta(0):
        raise ValueError("UTC timestamp required")
    return d.astimezone(timezone.utc)


def read_jsonl(path):
    p=Path(path)
    if not p.exists():return []
    return [json.loads(line) for line in p.read_text(encoding="utf8").splitlines()
            if line.strip()]


def sha_payload(value):
    return hash_object(value)


def validate_catalogs(base,snap):
    for name,cat in (("original",base),("snap",snap)):
        if not isinstance(cat,dict) or cat.get("sha256")!=hash_object({
            k:v for k,v in cat.items() if k!="sha256"}):
            raise ValueError(f"{name} 2024 calibration SHA mismatch")
        if cat.get("development_year",cat.get("training_year"))!=2024:
            raise ValueError(f"{name} calibration used other outcome year")
    if snap.get("base_full_distribution_catalog_sha256")!=base["sha256"]:
        raise ValueError("snap CDF did not fit against original frozen catalog")


def forecast_key(r):
    return (str(r["game_id"]),str(r["player_id"]),str(r["market"]))


def line_key(r):
    return (str(r["game_id"]),str(r["player_id"]),
            str(r["market"]),float(r["synthetic_line"]))


def freeze_pairs(full,snap_sidecars,base_catalog,snap_catalog,*,now,path):
    validate_catalogs(base_catalog,snap_catalog)
    now=utc(now)
    side={}
    for r in snap_sidecars:
        k=forecast_key(r)
        if k in side:raise ValueError("duplicate 2026 snap CDF group")
        side[k]=r
    original={}
    for f in full:
        k=forecast_key(f)
        if k in original:raise ValueError("duplicate original full-CDF group")
        original[k]=f
    prior=read_jsonl(path)
    seen=set()
    for r in prior:
        k=line_key(r)
        if k in seen:raise ValueError("duplicate existing originally frozen threshold")
        seen.add(k)
        if r.get("model")!=VERSION or r.get("wager") is not False:
            raise ValueError("existing paired research identity corrupted")
        saved_sha=r.get("frozen_row_sha256")
        if saved_sha!=sha_payload({key:val for key,val in r.items()
                                   if key!="frozen_row_sha256"}):
            raise ValueError("existing paired original changed on disk")
    additions=[]
    reasons=defaultdict(int)
    for k,sidecar in sorted(side.items()):
        f=original.get(k)
        if f is None:
            reasons["ORIGINAL_FULL_CDF_NOT_FOUND"]+=1
            continue
        if f.get("model")!=FULL_MODEL or sidecar.get("model")!=SNAP_MODEL:
            raise ValueError("incompatible source model lineage")
        if f.get("residual_catalog_sha256")!=base_catalog["sha256"]:
            raise ValueError("original full CDF SHA inconsistent")
        if (sidecar.get("base_2024_CDF_sha256")!=base_catalog["sha256"] or
            sidecar.get("snap_2024_CDF_sha256")!=snap_catalog["sha256"]):
            raise ValueError("original frozen CDF sidecar catalog mismatch")
        if (f["kickoff_utc"]!=sidecar["kickoff_utc"] or
            f["team"]!=sidecar.get("team",f["team"]) or
            f["position"]!=sidecar["position"] or
            not math.isclose(float(f["original_point_mean"]),
                             float(sidecar["original_forecast_mean"]),
                             rel_tol=0,abs_tol=1e-9) or
            not math.isclose(float(f["original_scale"]),
                             float(sidecar["original_forecast_scale"]),
                             rel_tol=0,abs_tol=1e-9)):
            raise ValueError("full original and snap sidecar are not identical base contracts")
        ko=utc(f["kickoff_utc"])
        first=utc(f["first_observed_at_utc"])
        side_at=utc(sidecar["first_observed_at_utc"])
        source_snaps_at=utc(sidecar["original_snap_context_first_seen_utc"])
        if not first<=source_snaps_at<=side_at<=now<ko:
            reasons["ORIGINAL_SOURCE_AFTER_FREEZE_OR_KICKOFF"]+=1
            continue
        if ko>now+timedelta(days=HORIZON_DAYS):
            reasons["PREGAME_HORIZON_OVER_10_DAYS"]+=1
            continue
        if bucket(sidecar["prior_snap_mean_observed"])!=sidecar["frozen_bucket"]:
            raise ValueError("first-seen PFR snap bucket differs from snap source")
        base_vector=base_catalog["vectors"].get(f["residual_group_key"])
        if base_vector is None or len(base_vector)!=int(f["training_residual_count"]):
            raise ValueError("original CDF empirical error vector mismatch")
        if not math.isclose(_scale(f["market"],float(f["original_point_mean"])),
                            float(f["original_scale"]),rel_tol=0,abs_tol=1e-9):
            raise ValueError("original model point-stat scale was modified")
        group=sidecar["snap_CDF_group"]
        if sidecar["fallback_to_original_full_CDF"]:
            if group!=sidecar["original_CDF_group"]:
                raise ValueError("false fallback group")
            snap_vector=base_vector
        else:
            if not group.endswith("|"+sidecar["frozen_bucket"]):
                raise ValueError("snap CDF group does not match recorded bucket")
            snap_vector=snap_catalog["fitted_normalized_residual_vectors"].get(group)
            if not snap_vector:
                raise ValueError("frozen snap CDF vector missing")
        if sidecar["original_CDF_group"]!=f["residual_group_key"]:
            raise ValueError("baseline residual position/model group changed")
        for line in PROP_THRESHOLD_GRID[f["market"]]:
            dummy={"game_id":k[0],"player_id":k[1],"market":k[2],
                   "synthetic_line":line}
            if line_key(dummy) in seen:continue
            mean=float(f["original_point_mean"])
            pb=over_probability(mean,line,f["market"],base_vector)
            ps=over_probability(mean,line,f["market"],snap_vector)
            if not 0<pb<1 or not 0<ps<1:
                raise ValueError("invalid original conditional half-point chance")
            record={
                "model":VERSION,"first_observed_at_utc":now.isoformat(),
                "source_full_first_seen_utc":f["first_observed_at_utc"],
                "source_snap_first_seen_utc":sidecar["first_observed_at_utc"],
                "source_prior_snap_first_seen_utc":sidecar["original_snap_context_first_seen_utc"],
                "game_id":f["game_id"],"season":f["season"],"week":f["week"],
                "kickoff_utc":f["kickoff_utc"],"team":f["team"],
                "player_id":f["player_id"],"player_name":f["player_name"],
                "position":f["position"],"market":f["market"],
                "synthetic_line":float(line),"full_game":True,
                "p_over_original_conditional_on_official_stat_row":round(pb,10),
                "p_over_snap_conditional_on_official_stat_row":round(ps,10),
                "original_2024_catalog_sha256":base_catalog["sha256"],
                "snap_2024_catalog_sha256":snap_catalog["sha256"],
                "original_full_residual_group":f["residual_group_key"],
                "snap_residual_group":group,
                "snap_bucket":sidecar["frozen_bucket"],
                "source_snaps_mean":float(sidecar["prior_snap_mean_observed"]),
                "line_is_actual_BetMGM_offer":False,
                "original_US_BetMGM_quote_origin":None,
                "p_prop_void":None,"profit_ev":None,
                "wager":False,"decision":"PASS",
            }
            record["frozen_row_sha256"]=sha_payload(record)
            additions.append(record)
            seen.add(line_key(record))
    if additions:
        dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("a",encoding="utf8") as out:
            for row in additions:out.write(json.dumps(
                row,sort_keys=True,allow_nan=False)+"\n")
    return {"original_player_market_sources":len(original),
            "qualified_snap_player_markets":len(side),
            "first_seen_lines_added":len(additions),
            "existing_line_originals":len(prior),
            "cumulative_first_seen_lines":len(prior)+len(additions),
            "exclude_reasons":dict(sorted(reasons.items()))}


def source_games(schedule,stats):
    games={}
    for g in schedule:
        gid=str(g.get("game_id") or "")
        if not gid:continue
        if gid in games:raise ValueError("duplicate source schedule game")
        games[gid]=g
    byplayer={}
    teams=defaultdict(lambda:defaultdict(set))
    for s in stats:
        gid=str(s.get("game_id") or "")
        pid=str(s.get("player_id") or "")
        tm=_team(s.get("team"))
        if not gid or not pid or pid in {"None","nan",""}:continue
        key=(gid,pid)
        if key in byplayer:raise ValueError("duplicated official player game stat row")
        byplayer[key]=s
        teams[gid][tm].add(pid)
    return games,byplayer,teams


def eligible_grades(frozen,schedules,stats,*,now):
    observed=utc(now)
    games,players,team_rows=source_games(schedules,stats)
    records=[]
    blocked=defaultdict(int)
    seen=set()
    for f in frozen:
        key=line_key(f)
        if key in seen:raise ValueError("duplicate pregame line original")
        seen.add(key)
        if f.get("model")!=VERSION or f.get("wager") is not False:
            raise ValueError("invalid frozen CDF line model")
        rowhash=f.get("frozen_row_sha256")
        if rowhash!=sha_payload({k:v for k,v in f.items()
                                 if k!="frozen_row_sha256"}):
            raise ValueError("tampered frozen original source")
        original=utc(f["first_observed_at_utc"])
        kickoff=utc(f["kickoff_utc"])
        if original>=kickoff:raise ValueError("late first-seen prop line")
        source=games.get(f["game_id"])
        if source is None:
            blocked["SCHEDULE_NOT_PUBLISHED"]+=1;continue
        official=schedule_kickoff_utc(source)
        if official is None or utc(official)!=kickoff:
            blocked["ORIGINAL_KICKOFF_CHANGED_AUDIT_REQUIRED"]+=1;continue
        if source.get("home_score") is None or source.get("away_score") is None:
            blocked["GAME_NOT_COMPLETED"]+=1;continue
        if observed<kickoff+timedelta(hours=MATURE_HOURS):
            blocked["OFFICIAL_STATS_NOT_72H_MATURE"]+=1;continue
        home=_team(source["home_team"]);away=_team(source["away_team"])
        n_home=len(team_rows[f["game_id"]][home])
        n_away=len(team_rows[f["game_id"]][away])
        if n_home<MIN_PER_TEAM or n_away<MIN_PER_TEAM or n_home+n_away<MIN_COMBINED:
            blocked["OFFICIAL_TEAM_SOURCE_COVERAGE_INSUFFICIENT"]+=1;continue
        statsrow=players.get((f["game_id"],f["player_id"]))
        if statsrow is None:
            blocked["PLAYER_STAT_ROW_UNOBSERVED_DO_NOT_ASSUME_UNDER_OR_VOID"]+=1
            continue
        if _team(statsrow.get("team"))!=_team(f["team"]):
            blocked["PLAYER_TEAM_IDENTITY_DISAGREES"]+=1;continue
        try:
            actual=float(statsrow.get(f["market"]))
        except (ValueError,TypeError,OverflowError):
            blocked["PLAYER_METRIC_NOT_RECORDED"]+=1;continue
        if not math.isfinite(actual):
            blocked["PLAYER_METRIC_NOT_FINITE"]+=1;continue
        line=float(f["synthetic_line"])
        if not math.isclose(line%1,.5,abs_tol=1e-9):
            raise ValueError("only original half-point research thresholds supported")
        truth=int(actual>line)
        records.append({
            "model":VERSION,"grade_source_captured_utc":observed.isoformat(),
            "game_id":f["game_id"],"player_id":f["player_id"],
            "market":f["market"],"synthetic_line":line,
            "original_first_seen_utc":f["first_observed_at_utc"],
            "original_row_sha256":f["frozen_row_sha256"],
            "kickoff_utc":f["kickoff_utc"],"team":f["team"],
            "official_stat_value":actual,
            "target_over_given_official_stat_row":truth,
            "p_original":float(f["p_over_original_conditional_on_official_stat_row"]),
            "p_snap":float(f["p_over_snap_conditional_on_official_stat_row"]),
            "source_home_distinct_player_rows":n_home,
            "source_away_distinct_player_rows":n_away,
            "source_record_is_BetMGM_settlement":False,
            "p_void":None,"book_quote":None,"actual_money_EV":None,
            "betting_decision":"PASS",
        })
    return records,dict(sorted(blocked.items()))


def append_grades(records,path):
    prev=read_jsonl(path)
    current={}
    for p in prev:
        key=line_key(p)
        if key not in current or p["revision"]>current[key]["revision"]:
            current[key]=p
    changed=[];corrected=0
    for r in records:
        k=line_key(r)
        old=current.get(k)
        if old is None:
            x={**r,"revision":1,"record_type":"ORIGINAL_MATURE_GRADE"}
        elif (old["target_over_given_official_stat_row"]!=r["target_over_given_official_stat_row"]
              or old["official_stat_value"]!=r["official_stat_value"]):
            if old["original_row_sha256"]!=r["original_row_sha256"]:
                raise ValueError("cannot rewrite original first-seen pair upon source revision")
            x={**r,"revision":old["revision"]+1,
               "record_type":"APPENDED_OFFICIAL_SOURCE_STAT_CORRECTION",
               "previous_official_stat_value":old["official_stat_value"]}
            corrected+=1
        else:continue
        changed.append(x)
        current[k]=x
    if changed:
        dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("a",encoding="utf8") as out:
            for r in changed:out.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    return {"prior_records":len(prev),"new_records":len(changed),
            "provider_corrections_appended":corrected,
            "total_saved_records":len(prev)+len(changed)}


def grade_metrics(grade_rows,*,draws=BOOTSTRAPS,seed=2026):
    latest={}
    for g in grade_rows:
        k=line_key(g)
        if k not in latest or int(g["revision"])>int(latest[k]["revision"]):
            latest[k]=g
    by_market=defaultdict(list)
    for g in latest.values():
        market=g["market"]
        if market not in PROP_THRESHOLD_GRID:
            raise ValueError("unregistered player-prop research market")
        y=int(g["target_over_given_official_stat_row"])
        if y not in (0,1):raise ValueError("invalid official conditional line outcome")
        pb=float(g["p_original"]);ps=float(g["p_snap"])
        if not 0<pb<1 or not 0<ps<1:raise ValueError("invalid original probabilities")
        by_market[market].append((g["game_id"],y,pb,ps))
    scores={}
    for market,obs in sorted(by_market.items()):
        n=len(obs)
        base=sum((p-y)**2 for gid,y,p,q in obs)/n
        snap=sum((q-y)**2 for gid,y,p,q in obs)/n
        grouped=defaultdict(list)
        for gid,y,p,q in obs:
            grouped[gid].append((q-y)**2-(p-y)**2)
        ci95=ci4=None
        if len(grouped)>=20:
            rng=random.Random(seed);keys=sorted(grouped);boot=[]
            for _ in range(draws):
                chosen=[keys[rng.randrange(len(keys))] for _ in keys]
                d=[v for gid in chosen for v in grouped[gid]]
                boot.append(sum(d)/len(d))
            boot.sort()
            ci95=[round(boot[int((draws-1)*.025)],8),
                  round(boot[int((draws-1)*.975)],8)]
            tail=.05/(2*FAMILYWISE_MARKETS)
            ci4=[round(boot[int((draws-1)*tail)],8),
                 round(boot[int((draws-1)*(1-tail))],8)]
        scores[market]={
            "distinct_source_games":len(grouped),
            "paired_player_market_synthetic_thresholds":n,
            "baseline_conditional_Brier":round(base,8),
            "snap_conditional_Brier":round(snap,8),
            "snap_minus_original_conditional_Brier":round(snap-base,8),
            "same_game_cluster_ci95":ci95,
            "four_market_adjusted_ci98_75":ci4,
            "forward_market_betting_edge_proven":False,
            "no_vig_actual_BetMGM_comparator":False,
        }
    return {"grade_counts":len(latest),"by_market":scores,
            "conditional_official_stat_row_only":True,
            "void_and_activation_unverified":True,
            "validated_BetMGM_profit":False,
            "wagers":0}


def run(*,freeze_only=False):
    now=datetime.now(timezone.utc).isoformat()
    full=read_jsonl("history/player_prop_full_distributions_2026.jsonl")
    snap=read_jsonl("history/player_prop_snap_cdf_sidecar_v1_2026.jsonl")
    base=json.loads(Path("history/player_prop_2024_residual_catalog_v1.json").read_text())
    snapcat=json.loads(Path("history/player_prop_prior_snap_cdf_2024_v1.json").read_text())
    if not full or not snap:raise ValueError("missing original CDF first-seen sources")
    frozen=freeze_pairs(full,snap,base,snapcat,now=now,path=FROZEN_PATH)
    if freeze_only:
        print(json.dumps({"model":VERSION,"stage":"PREGAME_FIRST_SEEN_ONLY",
                          "first_seen_capture_utc":now,"paired_originals":frozen,
                          "no_postgame_data_loaded":True,
                          "betting_decision":"PASS"},sort_keys=True))
        return frozen
    # First-seen paired lines must already be COMMITTED by the workflow before
    # any target-game provider source is queried. An outage cannot erase
    # the original forecast or make it post-kickoff.
    cli=NFLDataClient()
    schedules=cli.load_schedules([2026],refresh=True).to_dicts()
    stats=cli.load_player_stats([2026],refresh=True).to_dicts()
    gradeable,blocks=eligible_grades(read_jsonl(FROZEN_PATH),schedules,stats,now=now)
    written=append_grades(gradeable,GRADE_PATH)
    all_grades=read_jsonl(GRADE_PATH)
    grade=grade_metrics(all_grades)
    # The anchor is selected from original BASE probabilities and first-seen
    # before kickoff. Never generate or edit anchors after seeing outcomes.
    original_anchors=read_jsonl(
        "history/player_prop_primary_anchor_v1_2026.jsonl")
    anchor_grade=None
    if original_anchors:
        from scripts.nfl_prop_primary_anchor_score import anchor_results
        anchor_grade=anchor_results(
            original_anchors,read_jsonl(FROZEN_PATH),all_grades)
    report={
        "model":VERSION,"decision":"PASS","first_seen_capture_utc":now,
        "paired_originals":frozen,"source_grading_eligibility":blocks,
        "mature_official_stat_row_conditional_grades_now":len(gradeable),
        "append_only_grade_journal":written,"forward_performance":grade,
        "primary_one_line_per_player_game_market_anchored_scoring":anchor_grade,
        "primary_anchor_preregistered_source_count":len(original_anchors),
        "all_seven_lines_are_correlated_sensitivity_analysis":True,
        "original_full_and_snap_CDF_journals_unchanged":True,
        "2024_original_calibration_catalogs_unchanged":True,
        "source_grading_min_hours_after_kickoff":MATURE_HOURS,
        "source_actual_stat_rows_may_not_equal_sportsbook_settlements":True,
        "actual_US_BetMGM_account_quotes":0,"true_market_ev":None,
        "profitable_strategy_validated":False,"cash_bets":0,
    }
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-only",action="store_true",help="Commit before touching postgame source")
    a=p.parse_args()
    run(freeze_only=a.freeze_only)
