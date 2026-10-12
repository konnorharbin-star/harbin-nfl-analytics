"""Research-only v2 player prop calibration: do not treat unobserved players as zeros.

2024 development sample: compare unconditional v1 residual errors with the
participation-proxy version that FITS only *explicit* official player rows.
2025 previously inspected historical diagnostic: evaluate BOTH against the
SAME confirmed stat-row population and SAME independent point predictions.
2026 current forecasts, if any, are frozen independently before kickoff.
No account odds, verified snaps, void rules or validated BetMGM profit.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from nfl.data import NFLDataClient
from nfl.prop_challenger import METRICS, forecast_week, source_frames
from nfl.prop_probability import (
    PROP_THRESHOLD_GRID, _errors_for, fit_empirical_errors, over_probability,
)

VERSION="PROP_PARTICIPATION_CONDITIONAL_V2_RESEARCH"
DEFAULT_OPTIONS={key:(5,1.0) for key in METRICS}
MIN_TRAIN_MARKET=160
MIN_DIAGNOSTIC_GROUPS=60
BOOTSTRAPS=1000

FREEZE_FIELDS=(
    "version","first_observed_at_utc","game_id","season","week",
    "kickoff_utc","team","player_id","player_name","position","market",
    "synthetic_line","p_over","p_under","calibration_basis",
    "train_residual_count","player_participation_verified",
    "bookmaker","american_odds","model_ev","recommended_bet",
)


def samples(games, stats, season, options):
    """Same pre-week forecasts for both calibration populations.

    Eligibility never uses future target-week rows. At scoring time, observed
    stat row means explicit verified game/team/player match in source_frames.
    Missing source row is *unknown*, not an observed zero or known non-player.
    """
    outcomes={(x["game_id"],x["player_id"]):x for x in stats}
    complete={g["game_id"] for g in games if g["season"]==season and g["completed"]}
    weeks=sorted({g["week"] for g in games
                  if g["season"]==season and g["completed"] and g["week"]>=4})
    observed=[]
    legacy_zero=[]
    unresolved=[]
    for week in weeks:
        for pred in forecast_week(games,stats,season=season,week=week,
                                  options=options):
            if pred["game_id"] not in complete:continue
            obj=outcomes.get((pred["game_id"],pred["player_id"]))
            if obj is None:
                unresolved.append(pred)
                legacy_zero.append((pred,0.0))
                continue
            value=obj.get(pred["market"])
            if value is None or not math.isfinite(float(value)):
                unresolved.append(pred)
                continue
            if obj["team"]!=pred["team"]:
                raise ValueError("observed player/team identity mismatch")
            actual=float(value)
            observed.append((pred,actual))
            legacy_zero.append((pred,actual))
    return {
        "participation_proxy":observed,
        "legacy_zero_imputation":legacy_zero,
        "unresolved_without_matching_stat":unresolved,
    }


def fit_both(training):
    available=training["participation_proxy"]
    pooled=training["legacy_zero_imputation"]
    models={
        "v1_legacy_zero":fit_empirical_errors(pooled),
        "v2_stats_row_conditional":fit_empirical_errors(available),
    }
    for name,source in (("legacy",pooled),("conditional",available)):
        counts=Counter(p["market"] for p,y in source)
        lacking={m:n for m,n in counts.items() if n<MIN_TRAIN_MARKET}
        if lacking or set(counts)!=set(METRICS):
            raise ValueError("insufficient 2024 "+name+" market residuals: "+str(lacking))
    return models


def _binary(y,p):
    return (p-y)**2


def evaluate(samples_2025,models,*,bootstrap=BOOTSTRAPS,seed=414):
    """Paired same-row analysis with game-cluster resampling, no tuning to 2025."""
    records=defaultdict(list)
    for prediction,actual in samples_2025["participation_proxy"]:
        market=prediction["market"]
        line_grid=PROP_THRESHOLD_GRID[market]
        refs={}
        for key,errors in models.items():
            pair=_errors_for(errors,method="challenger",market=market,
                             position=prediction["position"])
            if pair is None:break
            refs[key]=pair[0]
        if len(refs)!=len(models):continue
        for line in line_grid:
            actual_over=float(actual>line)
            estimates={
                key:over_probability(
                    float(prediction["challenger_mean"]),line,market,residuals
                ) for key,residuals in refs.items()
            }
            delta=_binary(actual_over,estimates["v2_stats_row_conditional"])-_binary(
                actual_over,estimates["v1_legacy_zero"]
            )
            records[market].append({
                "game_id":prediction["game_id"],
                "player_id":prediction["player_id"],
                "line":line,"actual_over":int(actual_over),
                "p_legacy":estimates["v1_legacy_zero"],
                "p_conditional":estimates["v2_stats_row_conditional"],
                "brier_legacy":_binary(actual_over,estimates["v1_legacy_zero"]),
                "brier_conditional":_binary(actual_over,estimates["v2_stats_row_conditional"]),
                "delta":delta,
            })
    rng=random.Random(seed)
    results={}
    for market,rows in sorted(records.items()):
        by_game=defaultdict(list)
        by_player_game=set()
        for r in rows:
            by_game[r["game_id"]].append(r)
            by_player_game.add((r["game_id"],r["player_id"]))
        games=sorted(by_game)
        n=len(rows)
        old=sum(r["brier_legacy"] for r in rows)/n
        new=sum(r["brier_conditional"] for r in rows)/n
        boot=None
        if bootstrap and len(games)>=8:
            vals=[]
            for _ in range(bootstrap):
                selected=[games[rng.randrange(len(games))] for _ in games]
                sampled=[r for g in selected for r in by_game[g]]
                vals.append(sum(r["delta"] for r in sampled)/len(sampled))
            vals.sort()
            boot=[round(vals[int((bootstrap-1)*0.025)],8),
                  round(vals[int((bootstrap-1)*0.975)],8)]
        results[market]={
            "2025_source_stat_player_groups":len(by_player_game),
            "2025_unique_games":len(games),
            "correlated_synthetic_thresholds":n,
            "2025_brier_v1_zero_imputed_training":round(old,8),
            "2025_brier_v2_stat_row_conditional_training":round(new,8),
            "2025_brier_delta_v2_minus_v1":round(new-old,8),
            "2025_game_cluster_bootstrap_ci95":boot,
            "2025_is_pristine_holdout":False,
            "2025_eligible_for_promotion":False,
            "v2_better_point_diagnostic":new<old,
        }
    return results


def forward_curves(predictions,model):
    rows=[]
    for pred in predictions:
        market=pred["market"]
        subset=_errors_for(model,method="challenger",market=market,
                           position=pred["position"])
        if subset is None:continue
        errors,source=subset
        for threshold in PROP_THRESHOLD_GRID[market]:
            p=over_probability(float(pred["challenger_mean"]),
                               threshold,market,errors)
            row={
                "game_id":pred["game_id"],"season":pred["season"],
                "week":pred["week"],"kickoff_utc":pred["kickoff_utc"],
                "team":pred["team"],"player_id":pred["player_id"],
                "player_name":pred["player_name"],"position":pred["position"],
                "market":market,"synthetic_line":threshold,
                "p_over":round(p,6),"p_under":round(1-p,6),
                "calibration_basis":source,
                "train_residual_count":len(errors),
                "player_participation_verified":False,
                "bookmaker":None,"american_odds":None,"model_ev":None,
                "recommended_bet":False,
            }
            rows.append(row)
    return rows


def freeze_first_seen(entries,*,observed_utc,path):
    """Original 2026 history remains unchanged; no backfill or overwrite."""
    now=datetime.fromisoformat(observed_utc.replace("Z","+00:00"))
    if now.tzinfo is None:raise ValueError("collector time must be UTC-aware")
    history=[]
    file=Path(path)
    if file.exists():
        with file.open(newline="",encoding="utf8") as f:
            history=list(csv.DictReader(f))
    def key(row):
        return (str(row["game_id"]),str(row["player_id"]),
                str(row["market"]),float(row["synthetic_line"]))
    existing={key(r) for r in history}
    if len(existing)!=len(history):
        raise ValueError("duplicate past frozen challenger")
    created=[]
    seen=set()
    for r in entries:
        k=key(r)
        if k in existing:continue
        if k in seen:raise ValueError("duplicate candidate freeze threshold")
        seen.add(k)
        kickoff=datetime.fromisoformat(str(r["kickoff_utc"]).replace("Z","+00:00"))
        if kickoff.tzinfo is None or kickoff<=now:
            continue
        created.append({
            "version":VERSION,
            "first_observed_at_utc":now.astimezone(timezone.utc).isoformat(),
            **r,
        })
    if created:
        file.parent.mkdir(parents=True,exist_ok=True)
        with file.open("a",newline="",encoding="utf8") as stream:
            writer=csv.DictWriter(stream,fieldnames=FREEZE_FIELDS)
            if not history:writer.writeheader()
            for row in created:writer.writerow(row)
    return {"existing":len(history),"new":len(created),
            "total":len(history)+len(created)}


def run(*,season=2026,week=6,freeze_path=None,report_path=None):
    cli=NFLDataClient()
    years=[2024,2025,season]
    schedules=cli.load_schedules(years,refresh=True)
    stats=cli.load_player_stats(years,refresh=True)
    games,players,source=source_frames(schedules,stats)
    choices=DEFAULT_OPTIONS.copy()
    train=samples(games,players,2024,choices)
    prior=fit_both(train)
    diagnostic=samples(games,players,2025,choices)
    result=evaluate(diagnostic,prior)
    now=datetime.now(timezone.utc)
    candidates=forecast_week(games,players,season=season,week=week,
                              options=choices,future_only_at=now)
    curves=forward_curves(candidates,prior["v2_stats_row_conditional"])
    append={"new":0,"existing":0,"total":0}
    if freeze_path is not None:
        append=freeze_first_seen(curves,observed_utc=now.isoformat(),
                                 path=freeze_path)
    report={
        "version":VERSION,"decision":"PASS","research_only":True,
        "captured_utc":now.isoformat(),
        "train_year":2024,"diagnostic_year":2025,
        "2025_previously_inspected":True,
        "2026_not_used_for_fit":True,
        "predefined_point_forecast_options":choices,
        "training_participation_source":"MATCHED_OFFICIAL_STAT_ROW_ONLY",
        "full_snap_verification":False,
        "probability_condition":"P(OVER | eligible original forecast AND official player stats row)",
        "participation_or_void_model_validated":False,
        "2024_matched_player_market_predictions":len(train["participation_proxy"]),
        "2024_unresolved_missing_player_predictions":len(train["unresolved_without_matching_stat"]),
        "2024_old_training_with_missing_imputed_zero":len(train["legacy_zero_imputation"]),
        "2025_matched_player_market_predictions":len(diagnostic["participation_proxy"]),
        "2025_unresolved_missing_player_predictions":len(diagnostic["unresolved_without_matching_stat"]),
        "replay_2025":result,
        "forward_synthetic_lines_created":len(curves),
        "forward_frozen_records":append,
        "source_rows":source,
        "approved_betting_model":False,
        "current_account_betmgm_prices":None,
        "profitability_verified":False,
        "release_state":"UNPRICED_SHADOW_NOT_DEPLOYED_FOR_BETS",
    }
    if report_path is not None:
        p=Path(report_path);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--week",type=int,default=6)
    p.add_argument("--freeze",type=Path,default=Path("history/prop_participation_shadow_v2_2026.csv"))
    p.add_argument("--report",type=Path,default=Path("reports/prop_participation_v2_replay.json"))
    a=p.parse_args()
    report=run(season=a.season,week=a.week,freeze_path=a.freeze,report_path=a.report)
    print(json.dumps({k:report[k] for k in (
        "version","release_state","2024_matched_player_market_predictions",
        "2024_unresolved_missing_player_predictions","forward_frozen_records",
        "replay_2025",
    )},indent=2,sort_keys=True))
if __name__=="__main__":
    main()
