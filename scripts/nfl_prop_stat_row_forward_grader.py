"""Prospective grading of the frozen 2026 NFL *official-stat-row* proxy.

This is NOT activation, snap share, over/under win, push, VOID, cash return or
market edge. Source provider releases can lag; wait >=72h after kickoff and
require source game team coverage before grading absent player row as missing.
First grade stays immutable; if upstream revises an answer, append correction.

The 2024 fitted model coefficients and the first-seen 2026 pregame forecasts
are never touched. All results and decisions are RESEARCH_ONLY / PASS.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import nflreadpy as nfl

from nfl.data import NFLDataClient
from nfl.pregame import schedule_kickoff_utc
from scripts.nfl_prop_snap_stat_row_diagnostic import VERSION as FROZEN_MODEL
from scripts.nfl_moneyline_elo_research import _team

VERSION="NFL_PROP_2026_FROZEN_STAT_ROW_FORWARD_GRADE_V1"
MATURE_HOURS=72
MIN_TEAM_STAT_ROWS=4
MIN_BOTH_TEAM_STAT_ROWS=12
PRIOR_2024_RATE=0.8234786361674579

def utc(s):
    d=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    if d.tzinfo is None or d.utcoffset()!=timedelta(0):
        raise ValueError("source/original datetime must be UTC")
    return d.astimezone(timezone.utc)

def read_jsonl(path):
    p=Path(path)
    if not p.exists():return []
    return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]

def sha256_beta(beta):
    return hashlib.sha256(json.dumps(beta,sort_keys=True).encode()).hexdigest()

def source_rows_by_game(schedules,stats):
    matches={}
    for g in schedules:
        gid=str(g.get("game_id") or "")
        if not gid:continue
        if gid in matches:raise ValueError("duplicate official schedule game")
        matches[gid]=g
    present=defaultdict(set)
    team_coverage=defaultdict(lambda:defaultdict(set))
    for row in stats:
        gid=str(row.get("game_id") or "")
        player=str(row.get("player_id") or "")
        team=_team(row.get("team"))
        if not gid or not player or player in {"None","nan",""}:continue
        if (gid,player) in present[gid]:
            raise ValueError("duplicate official player game stat source")
        present[gid].add((gid,player))
        team_coverage[gid][team].add(player)
    return matches,present,team_coverage

def observations(frozen,schedules,stats,pinned,*,captured_utc):
    now=utc(captured_utc)
    expected_hash=sha256_beta(pinned["full_beta"])
    if (pinned.get("model")!=FROZEN_MODEL or
        pinned.get("development_year")!=2024 or
        expected_hash!=pinned.get("full_beta_sha256") or
        abs(float(pinned["development_stat_row_presence_rate"])-PRIOR_2024_RATE)>1e-12):
        raise ValueError("pinned original 2024 proxy calibration invalid")
    game_by_id,player_rows,team_rows=source_rows_by_game(schedules,stats)
    seen=set();candidate=[];block=defaultdict(int)
    for r in frozen:
        if (r.get("version")!=FROZEN_MODEL or r.get("wager") is not False
            or r.get("estimated_true_EV") is not None
            or r.get("p_player_active") is not None
            or r.get("p_prop_void") is not None
            or r.get("fit_beta_sha256")!=expected_hash):
            raise ValueError("2026 original is not approved immutable research source")
        key=(str(r["game_id"]),str(r["player_id"]))
        if key in seen:raise ValueError("duplicate frozen game-player pregame")
        seen.add(key)
        first=utc(r["first_observed_at_utc"]);kickoff=utc(r["kickoff_utc"])
        if first>=kickoff:
            raise ValueError("original first-seen forecast not pregame")
        p=float(r["p_official_stat_row_present_NOT_DNP"])
        if not math.isfinite(p) or not 0<p<1:raise ValueError("invalid binary pregame probability")
        g=game_by_id.get(key[0])
        if g is None:
            block["MISSING_SCHEDULE_GAME"]+=1
            continue
        official=schedule_kickoff_utc(g)
        if official is None or utc(official)!=kickoff:
            block["KICKOFF_CHANGED_REQUIRES_MANUAL_ORIGINAL_TIMELINE_AUDIT"]+=1
            continue
        if g.get("home_score") is None or g.get("away_score") is None:
            block["GAME_NOT_COMPLETED"]+=1
            continue
        if now<kickoff+timedelta(hours=MATURE_HOURS):
            block["SOURCE_MATURITY_72H_NOT_REACHED"]+=1
            continue
        home=_team(g.get("home_team"));away=_team(g.get("away_team"))
        h_n=len(team_rows[key[0]][home]);a_n=len(team_rows[key[0]][away])
        if (h_n<MIN_TEAM_STAT_ROWS or a_n<MIN_TEAM_STAT_ROWS or
            h_n+a_n<MIN_BOTH_TEAM_STAT_ROWS):
            block["INSUFFICIENT_BOTH_TEAM_OFFICIAL_STAT_SOURCE_COVERAGE"]+=1
            continue
        outcome=int(key in player_rows[key[0]])
        candidate.append({
            "model":VERSION,
            "original_model":r["version"],
            "original_first_observed_at_utc":r["first_observed_at_utc"],
            "first_source_grade_observed_utc":now.isoformat(),
            "game_id":key[0],"player_id":key[1],
            "kickoff_utc":r["kickoff_utc"],
            "p_row_record_present_original":p,
            "observed_official_player_stat_row_presence":outcome,
            "observed_source_home_team_stat_rows":h_n,
            "observed_source_away_team_stat_rows":a_n,
            "outcome_means_confirmed_DNP_or_void":False,
            "betmgm_prop_price":None,"money_ev":None,
            "betting_decision":"PASS",
        })
    return candidate,dict(sorted(block.items()))

def write_immutable_grades(candidates,path):
    old=read_jsonl(path)
    def key(r):return (str(r["game_id"]),str(r["player_id"]))
    previous={}
    for r in old:
        k=key(r)
        previous[k]=r  # last official correction is current label
    new=[]; corrections=0
    for r in candidates:
        k=key(r)
        prior=previous.get(k)
        if prior is None:
            fresh={**r,"revision":1,"record_type":"ORIGINAL_GRADE"}
        elif prior["observed_official_player_stat_row_presence"]!=r["observed_official_player_stat_row_presence"]:
            fresh={**r,"revision":int(prior["revision"])+1,
                   "record_type":"APPENDED_SOURCE_CORRECTION",
                   "previous_observed_row_presence":prior["observed_official_player_stat_row_presence"]}
            corrections+=1
        else:continue
        new.append(fresh)
        previous[k]=fresh
    if new:
        dest=Path(path)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("a",encoding="utf8") as f:
            for row in new:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    return {"previous_records":len(old),"new_records":len(new),
            "source_corrections_appended":corrections,
            "cumulative_records":len(old)+len(new)}

def score_grades(latest_grades,*,bootstraps=400,seed=822):
    latest={}
    for r in latest_grades:
        latest[(r["game_id"],r["player_id"])]=r
    pairs=list(latest.values())
    if not pairs:
        return {"n_games":0,"n_player_games":0,"calibration_sample_insufficient":True,
                "profitable_edge_proven":False}
    observed=[(r["game_id"],int(r["observed_official_player_stat_row_presence"]),
               float(r["p_row_record_present_original"])) for r in pairs]
    n=len(observed)
    rows=defaultdict(list)
    diff=[]
    for gid,y,p in observed:
        v=(p-y)**2-(PRIOR_2024_RATE-y)**2
        rows[gid].append(v);diff.append(v)
    brier_model=sum((p-y)**2 for _,y,p in observed)/n
    brier_prior=sum((PRIOR_2024_RATE-y)**2 for _,y,p in observed)/n
    ci=None
    if len(rows)>=20:
        rng=random.Random(seed);keys=sorted(rows);boot=[]
        for _ in range(bootstraps):
            picked=[keys[rng.randrange(len(keys))] for _ in keys]
            pts=[v for gid in picked for v in rows[gid]]
            boot.append(sum(pts)/len(pts))
        boot.sort()
        ci=[round(boot[int((bootstraps-1)*.025)],8),
            round(boot[int((bootstraps-1)*.975)],8)]
    return {
        "n_games":len(rows),"n_player_games":n,
        "mean_original_predicted_row_presence":round(sum(p for _,y,p in observed)/n,8),
        "observed_official_stats_row_presence_rate":round(sum(y for _,y,p in observed)/n,8),
        "original_model_forward_brier":round(brier_model,8),
        "fixed_2024_constant_rate_forward_brier":round(brier_prior,8),
        "game_clustered_brier_delta_model_minus_constant_ci95":ci,
        "calibration_sample_insufficient":len(rows)<20,
        "label_is_actually_bet_settlement":False,
        "profitable_edge_proven":False,
    }

def run():
    now=datetime.now(timezone.utc).isoformat()
    frozen=read_jsonl("history/player_prop_stat_row_proxy_v1_2026.jsonl")
    pinned=json.loads(Path("history/player_prop_stat_row_training_2024_v1.json").read_text(encoding="utf8"))
    if not frozen:raise ValueError("missing original 2026 player-game first-seen forecasts")
    client=NFLDataClient()
    # No ad-hoc live betting source: official game/source only.
    schedules=client.load_schedules([2026],refresh=True).to_dicts()
    players=client.load_player_stats([2026],refresh=True).to_dicts()
    eligible,blocks=observations(frozen,schedules,players,pinned,captured_utc=now)
    path="history/player_prop_stat_row_grades_v1_2026.jsonl"
    written=write_immutable_grades(eligible,path)
    grading=score_grades(read_jsonl(path))
    report={
        "model":VERSION,"decision":"PASS","source_observed_utc":now,
        "original_pregame_player_games":len(frozen),
        "source_outcomes_mature_72h_only":True,
        "official_stat_rows_are_NOT_player_active_or_bet_settlements":True,
        "source_condition_blocks":blocks,"eligible_grades_this_capture":len(eligible),
        "original_plus_appended_correction_journal":written,
        "forward_scoring":grading,
        "2024_original_training_coefficients_modified":False,
        "2026_original_probability_journal_modified":False,
        "source_original_bookmaker_price_verified":0,
        "true_EV":None,"wagers":0,"profitable_BetMGM_edge":False,
    }
    p=Path("reports/player_prop_stat_row_forward_grading.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True,indent=2))
    return report
if __name__=="__main__":run()
