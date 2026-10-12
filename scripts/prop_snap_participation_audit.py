"""Audit missing 2026 player market records using source-ID-matched PFR snaps.

Use nflverse source players.gsis_id -> players.pfr_id mapping, *never names*.
Snap evidence is observed POSTGAME, used only to study missing stat rows;
it must not be copied backward as a pregame participation forecast.
No missing stat is automatically a wager under, void, or active-player zero.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone,timedelta
import csv
import json
import math
from pathlib import Path

from nfl.pregame import schedule_kickoff_utc
from scripts.professional_method_audit import _utc

VERSION="PLAYER_MISSING_STAT_SNAP_ID_AUDIT_V1"


def crosswalk(players):
    """Validated unique GSIS-to-PFR mapping; ambiguous IDs do not map."""
    by_gsis={}
    by_pfr={}
    duplicates=set()
    for r in players:
        gsis=str(r.get("gsis_id") or "")
        pfr=str(r.get("pfr_id") or "")
        if not gsis or not pfr:continue
        if gsis in by_gsis and by_gsis[gsis]!=pfr:
            duplicates.add(gsis)
        by_gsis[gsis]=pfr
        by_pfr.setdefault(pfr,set()).add(gsis)
    ambiguous={g for g in by_gsis if len(by_pfr[by_gsis[g]])!=1}
    ambiguous|=duplicates
    return {g:p for g,p in by_gsis.items() if g not in ambiguous},len(ambiguous)


def snap_index(rows):
    index={}
    ambiguous=set()
    for r in rows:
        required=("game_id","pfr_player_id","team","offense_snaps")
        if any(k not in r for k in required):
            raise ValueError("missing required snap-count source keys")
        gid=str(r["game_id"] or "")
        pid=str(r["pfr_player_id"] or "")
        team=str(r["team"] or "")
        if not gid or not pid or not team:continue
        count=float(r["offense_snaps"]) if r["offense_snaps"] is not None else None
        if count is None or not math.isfinite(count) or count<0:continue
        key=(gid,pid)
        if key in index:ambiguous.add(key)
        index[key]=(team,count)
    return {k:v for k,v in index.items() if k not in ambiguous},len(ambiguous)


def audit(forecasts,schedules,stats,players,snaps,*,observed_utc):
    at=_utc(observed_utc)
    games={str(r["game_id"]):r for r in schedules}
    seen_stats={(str(r.get("game_id")),str(r.get("player_id")))
                for r in stats if r.get("game_id") and r.get("player_id")}
    mapping,ambiguous_players=crosswalk(players)
    snap_by_key,ambiguous_snaps=snap_index(snaps)
    reasons=Counter()
    groups=set()
    examples=[]
    graded_eligible=0
    raw_missing_thresholds=0
    for row in forecasts:
        gid=row.get("game_id")
        game=games.get(gid)
        if not game:
            reasons["UNKNOWN_GAME"]+=1
            continue
        kickoff=_utc(row["kickoff_utc"])
        confirmed=schedule_kickoff_utc(game)
        if confirmed is None or confirmed!=kickoff:
            reasons["KICKOFF_NOT_VERIFIED"]+=1
            continue
        if row.get("first_observed_at_utc") is None or _utc(row["first_observed_at_utc"])>=kickoff:
            raise ValueError("not original before-kickoff forecast")
        if kickoff+timedelta(hours=3)>at or game.get("home_score") is None or game.get("away_score") is None:
            reasons["NOT_FINAL_OR_TOO_EARLY"]+=1
            continue
        if (gid,str(row["player_id"])) in seen_stats:
            graded_eligible+=1
            continue
        raw_missing_thresholds+=1
        key=(gid,str(row["player_id"]),str(row["market"]))
        if key in groups:continue
        groups.add(key)
        player_id=str(row["player_id"])
        pfr=mapping.get(player_id)
        if pfr is None:
            result="NO_UNAMBIGUOUS_GSIS_PFR_CROSSWALK"
        else:
            snap=snap_by_key.get((gid,pfr))
            if snap is None:
                result="NO_VERIFIED_PFR_GAME_SNAP_RECORD"
            elif snap[0]!=row["team"]:
                result="PFR_SNAP_TEAM_MISMATCH"
            elif snap[1]>0:
                result="POSITIVE_OFFENSE_SNAPS_BUT_MISSING_STAT_ROW"
            else:
                result="ZERO_OFFENSE_SNAPS_DOES_NOT_PROVE_DNP"
        reasons[result]+=1
        if len(examples)<12:
            examples.append({
                "event_id":gid,"player_id":player_id,"market":row["market"],
                "classification":result,
            })
    return {
        "version":VERSION,"status":"POSTGAME_PARTICIPATION_RECONCILIATION_RESEARCH",
        "decision":"PASS",
        "observed_utc":at.isoformat(),
        "missing_original_threshold_rows":raw_missing_thresholds,
        "missing_distinct_player_game_market_groups":len(groups),
        "original_threshold_rows_with_official_stat":graded_eligible,
        "classification_groups":dict(sorted(reasons.items())),
        "player_identity_ambiguous_crosswalk_entries":ambiguous_players,
        "duplicate_pfr_game_snap_rows":ambiguous_snaps,
        "example_evidence":examples,
        "can_impute_missing_official_stats_as_zero":False,
        "source_original_pfr_snaps_asof_timestamp_known":False,
        "does_not_rewrite_original_predictions_or_grades":True,
        "bookmaker_void_rules_verified":False,
        "may_promote_model":False,
        "validated_betting_ev":False,
    }


def read_csv(path):
    with Path(path).open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    import argparse
    import nflreadpy as nfl
    from nfl.data import NFLDataClient
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--frozen",type=Path,
                    default=Path("history/player_prop_probability_forward.csv"))
    ap.add_argument("--report",type=Path,
                    default=Path("reports/player_snap_participation_audit.json"))
    a=ap.parse_args()
    if a.season!=2026:
        raise ValueError("frozen 2026 original source requires season 2026")
    now=datetime.now(timezone.utc).isoformat()
    source=read_csv(a.frozen)
    client=NFLDataClient()
    try:
        schedule=client.load_schedules([a.season],refresh=True).to_dicts()
        weekly=client.load_player_stats([a.season],refresh=True).to_dicts()
        players=nfl.load_players().to_dicts()
        snaps=nfl.load_snap_counts([a.season]).to_dicts()
        out=audit(source,schedule,weekly,players,snaps,observed_utc=now)
        out["source_error"]=None
        out["source_row_counts"]={
            "frozen_thresholds":len(source),"players_id_map":len(players),
            "PFR_snap_rows":len(snaps),"official_weekly_stats":len(weekly)}
    except (ValueError,RuntimeError,KeyError,TypeError,OSError) as error:
        out={
            "version":VERSION,"decision":"PASS","status":"SOURCE_UNAVAILABLE_OR_INVALID",
            "source_error":f"{type(error).__name__}: {str(error)[:300]}",
            "may_promote_model":False,"can_impute_missing_official_stats_as_zero":False,
            "observed_utc":now,
        }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(out,sort_keys=True,indent=2))
if __name__=="__main__":
    main()
