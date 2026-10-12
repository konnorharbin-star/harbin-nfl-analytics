"""Prospective NFL player-prop *single-anchor-line* integrity protocol.

In the existing frozen conditional CDF grid, seven half-point lines per
player/game/market are correlated. Select exactly ONE anchor ex ante:
the original/BASE CDF's half-point with P(OVER) closest to 0.50; on ties,
the lower numerical line. No postgame stat values, sportsbook prices or
candidate/snap CDF values inform selection.

Original 2026 4,886 paired lines are immutable. The new research-only
anchor is a separate append-only PRE-KICKOFF cohort with seven-line audit
and its original constituent SHA. No BetMGM offer or bet is asserted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from nfl.prop_probability import PROP_THRESHOLD_GRID
from scripts.nfl_prop_paired_cdf_forward_grader import (
    VERSION as PAIRED_VERSION,read_jsonl,sha_payload,utc,
)

VERSION="NFL_PROSPECTIVE_PROP_PRIMARY_ANCHOR_V1"
SOURCE=Path("history/player_prop_paired_cdf_lines_v1_2026.jsonl")
JOURNAL=Path("history/player_prop_primary_anchor_v1_2026.jsonl")
REPORT=Path("reports/player_prop_primary_anchor_v1_2026.json")

def group_key(r):
    return (str(r["game_id"]),str(r["player_id"]),str(r["market"]))

def _assert_complete_original_group(rows):
    original=rows[0]
    market=str(original["market"])
    expected=tuple(float(x) for x in PROP_THRESHOLD_GRID[market])
    if len(rows)!=len(expected) or tuple(sorted(float(x["synthetic_line"]) for x in rows))!=expected:
        raise ValueError("original paired player-market source does not contain exact full registered seven-line grid")
    fixed_fields=(
        "first_observed_at_utc","source_full_first_seen_utc",
        "source_snap_first_seen_utc","source_prior_snap_first_seen_utc",
        "game_id","player_id","market","team","position","kickoff_utc",
        "original_2024_catalog_sha256","snap_2024_catalog_sha256",
        "original_full_residual_group","snap_residual_group","snap_bucket",
        "source_snaps_mean",
    )
    for row in rows:
        if row.get("model")!=PAIRED_VERSION or row.get("wager") is not False:
            raise ValueError("not an original accepted research-only paired CDF line")
        if row.get("frozen_row_sha256")!=sha_payload({
            k:v for k,v in row.items() if k!="frozen_row_sha256"}):
            raise ValueError("original paired line SHA changed or missing")
        if any(row.get(k)!=original.get(k) for k in fixed_fields):
            raise ValueError("inconsistent same-player/game/stat sources across thresholds")
        if row["line_is_actual_BetMGM_offer"] is not False or row.get("p_prop_void") is not None:
            raise ValueError("invalid bookmaker provenance on synthetic source")
    ordered=sorted(rows,key=lambda x:float(x["synthetic_line"]))
    # The two 2024 fixed CDFs must be actual distributions (monotone tails).
    for prob_field in (
        "p_over_original_conditional_on_official_stat_row",
        "p_over_snap_conditional_on_official_stat_row",
    ):
        vals=[float(x[prob_field]) for x in ordered]
        if not all(math.isfinite(x) and 0<x<1 for x in vals):
            raise ValueError("nonfinite or invalid probability in a source CDF")
        if any(vals[i]+1e-10<vals[i+1] for i in range(len(vals)-1)):
            raise ValueError(f"original frozen {prob_field} has NONMONOTONE OVER probability")
    return ordered

def build_anchor_candidates(lines,*,now):
    at=utc(now)
    groups=defaultdict(list)
    for r in lines:groups[group_key(r)].append(r)
    anchors=[];excluded=Counter();risk=defaultdict(list)
    seen=set()
    for key,rows in sorted(groups.items()):
        ordered=_assert_complete_original_group(rows)
        original=ordered[0]
        first=utc(original["first_observed_at_utc"])
        ko=utc(original["kickoff_utc"])
        if not first<=at<ko:
            excluded["NOT_PREKICKOFF_AT_ANCHOR_REGISTRATION"]+=1
            continue
        # Original BASE probability exclusively decides the primary line.
        selected=min(ordered,key=lambda x:(
            round(abs(float(x["p_over_original_conditional_on_official_stat_row"])-.5),10),
            float(x["synthetic_line"]),
        ))
        item={
            "model":VERSION,
            "anchor_first_observed_at_utc":at.isoformat(),
            "paired_original_first_observed_at_utc":original["first_observed_at_utc"],
            "game_id":key[0],"player_id":key[1],"market":key[2],
            "team":original["team"],"position":original["position"],
            "kickoff_utc":original["kickoff_utc"],
            "anchor_synthetic_half_point_line":float(selected["synthetic_line"]),
            "p_over_original_base_CDF":float(
                selected["p_over_original_conditional_on_official_stat_row"]),
            "p_over_snap_challenger_CDF":float(
                selected["p_over_snap_conditional_on_official_stat_row"]),
            "constituent_paired_original_row_sha256":selected["frozen_row_sha256"],
            "complete_grid_original_row_sha256s":[x["frozen_row_sha256"] for x in ordered],
            "anchor_selection_rule":"MIN_ABS_ORIGINAL_P_OVER_MINUS_0_5_THEN_LOWER_LINE",
            "grid_threshold_count":len(ordered),
            "original_line_is_actual_BetMGM_offer":False,
            "bookmaker_quote_original_time":None,
            "p_bet_void":None,"bet_true_EV":None,"wager":False,
            "decision":"PASS",
        }
        item["anchor_record_sha256"]=sha_payload(item)
        if key in seen:raise ValueError("duplicate anchor player-game-market")
        seen.add(key)
        anchors.append(item)
        risk[key[2]].append(abs(
            item["p_over_original_base_CDF"]-item["p_over_snap_challenger_CDF"]))
    summary={
        "original_paired_line_count":len(lines),
        "complete_player_game_market_groups":len(groups),
        "eligible_anchored_player_game_market_groups":len(anchors),
        "excluded_game_market_group_reasons":dict(excluded),
        "by_market":{
            market:{
                "group_count":len(vals),
                "mean_abs_original_snap_pover_disagreement":round(
                    sum(vals)/len(vals),8),
                "max_abs_original_snap_pover_disagreement":round(max(vals),8),
            } for market,vals in sorted(risk.items())
        },
        "research_scope":"OFFICIAL_STAT_ROW_CONDITIONAL_SYNTHETIC_ANCHOR_ONLY",
        "actual_BetMGM_bookmaker_quotes":0,
        "prospective_bets":0,"true_EV":None,"decision":"PASS",
    }
    return anchors,summary

def append_first_seen(rows,path):
    path=Path(path)
    old=read_jsonl(path)
    registered={}
    for r in old:
        key=group_key(r)
        if key in registered:raise ValueError("duplicate frozen anchor in journal")
        if r.get("model")!=VERSION or r.get("anchor_record_sha256")!=sha_payload({
            k:v for k,v in r.items() if k!="anchor_record_sha256"}):
            raise ValueError("historical anchor original was changed")
        registered[key]=r
    new=[]
    for r in rows:
        key=group_key(r)
        if key in registered:continue
        if r.get("anchor_record_sha256")!=sha_payload({
            k:v for k,v in r.items() if k!="anchor_record_sha256"}):
            raise ValueError("candidate anchor record SHA wrong")
        registered[key]=r
        new.append(r)
    if new:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf8") as stream:
            for r in new:
                stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    return {"existing":len(old),"new_origination":len(new),"cumulative":len(old)+len(new)}

def run():
    now=datetime.now(timezone.utc).isoformat()
    original=read_jsonl(SOURCE)
    if not original:raise ValueError("missing original pregame paired player CDFs")
    candidates,summary=build_anchor_candidates(original,now=now)
    summary["version"]=VERSION
    summary["source_capture_utc"]=now
    summary["first_seen_written"]=append_first_seen(candidates,JOURNAL)
    summary["primary_sample_not_independent_across_markets_or_games"]=True
    summary["original_4886_probability_records_modified"]=False
    summary["true_BetMGM_approval"]=False
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(summary,sort_keys=True))
    return summary

if __name__=="__main__":
    run()
