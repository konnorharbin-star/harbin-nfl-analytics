"""Prospective, exact-contract v1-v2 NFL player prop forecast comparison.

Only *original* grade journals are admissible, and only where both versions
had a pregame frozen probability for the identical game, player, market, line.
Game-bootstrap CI respects shared matchup shocks; player/stat-threshold
samples are not independent bets. No sportsbook odds/ROI or model promotion.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,math,random
from pathlib import Path
from scripts.professional_method_audit import _utc

VERSION="PAIRED_NFL_FIRST_SEEN_V1_V2_PROSPECTIVE_V1"
V1_VERSION="NFL_FROZEN_PROP_FORWARD_SCORING_V1"
V2_VERSION="PROP_PARTICIPATION_CONDITIONAL_V2_FORWARD_GRADE_V1"
MIN_COMPARABLE_GAMES=40
MIN_DISTINCT_PLAYER_MARKETS=500
BOOTSTRAPS=1500


def _validate(entry, version):
    if entry.get("version")!=version:
        raise ValueError("wrong model source version")
    if not entry.get("forecast_id") or not entry.get("event_id"):
        raise ValueError("source result missing contract identity")
    kickoff=_utc(entry["kickoff_utc"])
    first=_utc(entry["first_forecast_utc"])
    settled=_utc(entry["first_scored_observed_utc"])
    if not first<kickoff<settled:
        raise ValueError("retroactive forecast or pre-kickoff result")
    if entry.get("verified_betmgm_quote") is not False or entry.get("betting_roi") is not None:
        raise ValueError("unexpectedly attributed real bet to synthetic forecast")
    x=float(entry["p_over"])
    y=int(entry["actual_over"])
    actual=float(entry["actual_stat"])
    line=float(entry["line"])
    if not (math.isfinite(x) and 0<=x<=1 and y in (0,1) and
            math.isfinite(actual) and math.isfinite(line)):
        raise ValueError("invalid scored probability or stat")
    if abs(line%1-.5)>1e-9 or y!=int(actual>line):
        raise ValueError("invalid recorded half-point target outcome")
    if abs(float(entry["brier_over"])-(x-y)**2)>1e-7:
        raise ValueError("stored score differs from original frozen probability")
    return (
        str(entry["event_id"]),str(entry["player_id"]),
        str(entry["market"]),line,
    )


def _index(rows,version):
    result={}
    for row in rows:
        key=_validate(row,version)
        if key in result:raise ValueError("duplicate original contract in graded cohort")
        result[key]=row
    return result


def _bootstrap_game_ci(grouped, seed, iters=BOOTSTRAPS):
    games=sorted(grouped)
    if len(games)<8:return None
    rng=random.Random(seed)
    values=[]
    for _ in range(iters):
        selected=[games[rng.randrange(len(games))] for _ in games]
        weighted=[v for event in selected for v in grouped[event]]
        values.append(sum(weighted)/len(weighted))
    values.sort()
    return [round(values[int((iters-1)*.025)],9),
            round(values[int((iters-1)*.975)],9)]


def compare(first, second):
    v1=_index(first,V1_VERSION)
    v2=_index(second,V2_VERSION)
    overlap=sorted(set(v1)&set(v2))
    diffs=defaultdict(lambda:defaultdict(list))
    games=set()
    groups=set()
    for key in overlap:
        a,b=v1[key],v2[key]
        if (
            a["actual_over"]!=b["actual_over"] or
            a["actual_stat"]!=b["actual_stat"] or
            a["kickoff_utc"]!=b["kickoff_utc"]
        ):
            raise ValueError("different official outcomes or kickoff for same contract")
        event,pid,market,line=key
        games.add(event)
        groups.add((event,pid,market))
        # Calculate from ORIGINAL frozen P; do not trust aggregate stored scores.
        diffs[market][event].append(
            (float(b["p_over"])-int(b["actual_over"]))**2-
            (float(a["p_over"])-int(a["actual_over"]))**2
        )
    results={}
    for market,by_game in sorted(diffs.items()):
        arr=[x for rows in by_game.values() for x in rows]
        point=sum(arr)/len(arr)
        ci=_bootstrap_game_ci(by_game,seed=20261012)
        market_groups=len({(k[0],k[1]) for k in overlap if k[2]==market})
        statistically_valid=(
            len(by_game)>=MIN_COMPARABLE_GAMES and
            market_groups>=MIN_DISTINCT_PLAYER_MARKETS and
            ci is not None and ci[1]<0
        )
        results[market]={
            "paired_thresholds":len(arr),
            "unique_games":len(by_game),
            "distinct_player_game_clusters":market_groups,
            "brier_delta_v2_minus_v1":round(point,9),
            "game_cluster_bootstrap_ci95":ci,
            "minimum_sample_thresholds_met":(
                len(by_game)>=MIN_COMPARABLE_GAMES and
                market_groups>=MIN_DISTINCT_PLAYER_MARKETS
            ),
            "improved_beyond_95pct_cluster_interval":statistically_valid,
            "independent_betmgm_market_outperformance_proven":False,
            "eligible_for_betting_approval":False,
        }
    return {
        "version":VERSION,
        "status":"NO_PAIRED_SETTLED_CONTRACTS" if not overlap
                 else "PROSPECTIVE_UNPRICED_PAIRED_RESEARCH",
        "decision":"PASS",
        "original_v1_graded_rows":len(first),
        "original_v2_graded_rows":len(second),
        "matched_exact_contracts":len(overlap),
        "matched_player_game_market_clusters":len(groups),
        "matched_distinct_games":len(games),
        "by_market":results,
        "reference_market_prices_used":False,
        "source_quote_origin_times_verified":False,
        "betmgm_account_executable_quote_verified":False,
        "profitable_betting_edge":None,
        "betting_recommendations":[],
        "selection_bias_notice":"Only matching outcome-graded contracts included; missing-player participation and void unresolved.",
        "multiple_comparison_caution":"Four player markets and correlated threshold grids; no point estimate is an actual BetMGM profit.",
    }


def read(path):
    file=Path(path)
    if not file.exists():return []
    return [json.loads(x) for x in file.read_text(encoding="utf8").splitlines() if x.strip()]


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1",type=Path,
                   default=Path("history/player_prop_outcomes_2026.jsonl"))
    p.add_argument("--v2",type=Path,
                   default=Path("history/prop_participation_outcomes_v2_2026.jsonl"))
    p.add_argument("--report",type=Path,
                   default=Path("reports/paired_v1_v2_prop_forward.json"))
    a=p.parse_args()
    report=compare(read(a.v1),read(a.v2))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
