"""2026 prospective single-anchor research scoring and selection-completeness audit.

Anchors use original conditional P(over) nearest 0.50 before kickoff, never
the snap challenger or observed outcomes. One line per player/game/market
reduces correlated seven-line pseudo replication but players across markets
and weeks are not wholly independent. Bootstrap game-cluster, familywise
Bonferroni across four markets; not sportsbook ROI or settlement.
"""
from __future__ import annotations
from collections import defaultdict
import math
import random

from nfl.prop_probability import PROP_THRESHOLD_GRID
from scripts.nfl_prop_paired_cdf_forward_grader import line_key,sha_payload,utc
from scripts.nfl_prop_primary_anchor_cohort import VERSION as ANCHOR_VERSION,group_key

MIN_CI_GAMES=20
FAMILY=4
BOOTSTRAPS=400

def latest_grade(rows):
    values={}
    for r in rows:
        k=line_key(r)
        last=values.get(k)
        if last is None or int(r["revision"])>int(last["revision"]):
            values[k]=r
    return values

def anchor_results(anchors,paired_originals,graded,*,seed=2026,draws=BOOTSTRAPS):
    originals={}
    for p in paired_originals:
        k=line_key(p)
        if k in originals:raise ValueError("duplicate original player-market line")
        if p["frozen_row_sha256"]!=sha_payload({
            x:v for x,v in p.items() if x!="frozen_row_sha256"}):
            raise ValueError("original paired CDF row corrupted")
        originals[k]=p
    graded_map=latest_grade(graded)
    groups=defaultdict(list)
    before_by_market=defaultdict(int)
    sample_by_market=defaultdict(int)
    counted=set()
    for anchor in anchors:
        key=group_key(anchor)
        if key in counted:raise ValueError("duplicate single pregame anchor")
        counted.add(key)
        if anchor.get("model")!=ANCHOR_VERSION:
            raise ValueError("unregistered anchor model")
        if anchor.get("anchor_record_sha256")!=sha_payload({
            x:v for x,v in anchor.items() if x!="anchor_record_sha256"}):
            raise ValueError("anchor content hash modified")
        if utc(anchor["anchor_first_observed_at_utc"])>=utc(anchor["kickoff_utc"]):
            raise ValueError("anchor selected after target kickoff")
        k=(key[0],key[1],key[2],float(anchor["anchor_synthetic_half_point_line"]))
        original=originals.get(k)
        if original is None:
            raise ValueError("original anchor threshold not found in first-seen pair")
        if (anchor["constituent_paired_original_row_sha256"]!=original["frozen_row_sha256"]
            or anchor["paired_original_first_observed_at_utc"]!=original["first_observed_at_utc"]):
            raise ValueError("anchor didn't bind exact original frozen forecast")
        if (anchor["p_over_original_base_CDF"]!=original["p_over_original_conditional_on_official_stat_row"]
            or anchor["p_over_snap_challenger_CDF"]!=original["p_over_snap_conditional_on_official_stat_row"]):
            raise ValueError("anchor probabilities altered after first-seen source")
        if original["market"] not in PROP_THRESHOLD_GRID:
            raise ValueError("unregistered market")
        # Verify exact non-hindsight probability-minimum criterion again.
        complete=[originals.get((key[0],key[1],key[2],float(t)))
                  for t in PROP_THRESHOLD_GRID[original["market"]]]
        if any(v is None for v in complete):
            raise ValueError("anchor no longer has all seven original lines")
        expected=min(complete,key=lambda r:(
            round(abs(r["p_over_original_conditional_on_official_stat_row"]-.5),10),
            r["synthetic_line"],
        ))
        if float(expected["synthetic_line"])!=k[3]:
            raise ValueError("anchor selection was not the predeclared original CDF median criterion")
        before_by_market[key[2]]+=1
        result=graded_map.get(k)
        if result is None:continue
        if result["original_row_sha256"]!=original["frozen_row_sha256"]:
            raise ValueError("grade refers to replaced originally frozen probability")
        y=int(result["target_over_given_official_stat_row"])
        if y not in (0,1):raise ValueError("not an observed binary over outcome")
        sample_by_market[key[2]]+=1
        groups[key[2]].append((key[0],y,
            float(anchor["p_over_original_base_CDF"]),
            float(anchor["p_over_snap_challenger_CDF"])))
    results={}
    for market in sorted(before_by_market):
        obs=groups[market]
        n=len(obs)
        metric={
            "preregistered_original_player_game_market_anchors":before_by_market[market],
            "mature_source_valid_graded_anchors":sample_by_market[market],
            "not_yet_valid_graded_anchor_groups":before_by_market[market]-sample_by_market[market],
            "independent_games_with_a_valid_grade":len({g for g,y,a,b in obs}),
            "original_conditional_base_Brier":None,
            "snap_conditional_challenger_Brier":None,
            "snap_minus_original_Brier":None,
            "game_cluster_ci95":None,
            "four_market_familywise_ci98_75":None,
            "source_row_presence_conditional_not_sportsbook_settlement":True,
            "profitable_BetMGM_edge_validated":False,
        }
        if obs:
            raw=[(q-y)**2-(p-y)**2 for g,y,p,q in obs]
            metric["original_conditional_base_Brier"]=round(
                sum((p-y)**2 for g,y,p,q in obs)/n,8)
            metric["snap_conditional_challenger_Brier"]=round(
                sum((q-y)**2 for g,y,p,q in obs)/n,8)
            metric["snap_minus_original_Brier"]=round(sum(raw)/n,8)
            by_game=defaultdict(list)
            for g,y,p,q in obs:by_game[g].append((q-y)**2-(p-y)**2)
            if len(by_game)>=MIN_CI_GAMES and draws>=80:
                rng=random.Random(seed)
                keys=sorted(by_game)
                boot=[]
                for _ in range(draws):
                    chosen=[keys[rng.randrange(len(keys))] for _ in keys]
                    differences=[v for g in chosen for v in by_game[g]]
                    boot.append(sum(differences)/len(differences))
                boot.sort()
                ci=lambda p:round(boot[min(draws-1,int((draws-1)*p))],8)
                metric["game_cluster_ci95"]=[ci(.025),ci(.975)]
                lower=.05/(2*FAMILY)
                metric["four_market_familywise_ci98_75"]=[ci(lower),ci(1-lower)]
        results[market]=metric
    return {
        "version":"NFL_2026_PROP_ANCHOR_PROSPECTIVE_SCORE_V1",
        "anchor_selection_rule":"BASE_P_OVER_CLOSEST_TO_50_PERCENT_BEFORE_KICKOFF",
        "original_single_anchor_groups":len(anchors),
        "one_anchor_per_player_game_market_not_independent_across_markets":True,
        "score_only_explicit_official_player_stat_rows":True,
        "scoreboard":results,
        "proven_accepted_BetMGM_quotes":0,
        "true_EV":None,"bet_approved":False,"decision":"PASS",
    }
