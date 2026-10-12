"""Freeze full pregame NFL player-stat empirical distributions, not price grids.

Research-only extension of v2. Reuses 2024 official-stat-row-conditional
(challenger mean, scale, complete normalized empirical residual vector) with the
predefined options. Once frozen, *any* observed threshold is evaluated against
the original archived 2024 vector and original pregame point forecast WITHOUT
a re-fit or interpolating pre-frozen fixed-grid probabilities.

This source does NOT authenticate BetMGM quotes, participation, DNP/void,
player GSIS crosswalks or market calibration. All outputs PASS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path

from nfl.data import NFLDataClient
from nfl.prop_challenger import METRICS, forecast_week, source_frames
from nfl.prop_probability import (
    MIN_POSITION_RESIDUALS, MIN_MARKET_RESIDUALS, PROP_THRESHOLD_GRID,
    _scale, over_probability,
)
from scripts.auto_freeze_participation_v2 import future_weeks
from scripts.participation_conditional_prop_v2 import (
    DEFAULT_OPTIONS, fit_both, samples,
)

VERSION = "NFL_PLAYER_PROP_FROZEN_FULL_DISTRIBUTION_V1"
CALIBRATION_VERSION = "PROP_V2_2024_STATS_ROW_CONDITIONAL_EMPIRICAL_V1"


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,
                      separators=(",",":")).encode("utf8")


def hash_object(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _utc(raw):
    if isinstance(raw,datetime):
        at=raw
    elif isinstance(raw,str):
        at=datetime.fromisoformat(raw.replace("Z","+00:00"))
    else:
        raise ValueError("UTC timestamp required")
    if at.tzinfo is None or at.utcoffset()!=timedelta(0):
        raise ValueError("UTC-aware timestamp required")
    return at.astimezone(timezone.utc)


def create_calibration(games,players):
    """2024 development only. Never use 2025/2026 targets to fit."""
    train=samples(games,players,2024,DEFAULT_OPTIONS)
    models=fit_both(train)
    grouped=models["v2_stats_row_conditional"]
    vectors={}
    for (method,market,position),errors in sorted(grouped.items()):
        if method=="challenger":
            vectors[f"{market}|{position}"]=[float(x) for x in errors]
    content={
        "calibration_version":CALIBRATION_VERSION,
        "training_year":2024, "training_outcome_population":"EXPLICIT_OFFICIAL_STAT_ROW",
        "training_not_an_independent_profitability_backtest":True,
        "player_stat_row_groups":len(train["participation_proxy"]),
        "predefined_player_point_options":{k:list(v) for k,v in DEFAULT_OPTIONS.items()},
        "vectors":vectors,
        "probability_scheme":"EMPIRICAL_SMOOTHED_TAIL_V2_CONDITIONAL",
        "book_prices_used":False,
    }
    content["sha256"]=hash_object(content)
    return content


def load_or_create_calibration(path, *, games=None, players=None):
    path=Path(path)
    if path.exists():
        result=json.loads(path.read_text(encoding="utf8"))
        digest=result.pop("sha256",None)
        if digest!=hash_object(result):
            raise ValueError("immutable 2024 residual catalog corrupted")
        result["sha256"]=digest
        if result.get("calibration_version") != CALIBRATION_VERSION:
            raise ValueError("immutable residual catalog version changed")
        return result,False
    if games is None or players is None:
        raise ValueError("source rows required to create first calibration")
    result=create_calibration(games,players)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Refuse replacing even if racing with another run.
    with path.open("x",encoding="utf8") as f:
        f.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result,True


def group_for(market,position,catalog):
    vectors=catalog["vectors"]
    direct=f"{market}|{position}"
    pool=f"{market}|ALL"
    if direct in vectors and len(vectors[direct])>=MIN_POSITION_RESIDUALS:
        return direct
    if pool in vectors and len(vectors[pool])>=MIN_MARKET_RESIDUALS:
        return pool
    return None


def candidate_distributions(predictions,catalog):
    output=[]
    for p in predictions:
        market=str(p["market"])
        position=str(p["position"])
        if market not in PROP_THRESHOLD_GRID:
            raise ValueError("unsupported player-stat market")
        group=group_for(market,position,catalog)
        if group is None:
            continue
        mean=float(p["challenger_mean"])
        if not math.isfinite(mean) or mean<0:
            raise ValueError("invalid original player-stat mean")
        output.append({
            "model":VERSION,"game_id":str(p["game_id"]),
            "season":int(p["season"]),"week":int(p["week"]),
            "kickoff_utc":str(p["kickoff_utc"]),
            "team":str(p["team"]),"player_id":str(p["player_id"]),
            "player_name":str(p["player_name"]),"position":position,
            "market":market,
            "original_point_mean":mean,"original_scale":_scale(market,mean),
            "residual_catalog_sha256":catalog["sha256"],
            "residual_group_key":group,
            "training_residual_count":len(catalog["vectors"][group]),
            "participation_and_void_verified":False,
            "bookmaker":None,"original_book_quote_time":None,
            "accepted_book_price":None,"true_ev":None,"decision":"PASS",
        })
    return output


def freeze_first_seen(rows, *, observed_utc, path):
    now=_utc(observed_utc)
    path=Path(path)
    original=[]
    if path.exists():
        original=[json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
    def key(row):
        return (row["game_id"],row["player_id"],row["market"])
    existing={key(x) for x in original}
    if len(existing)!=len(original):
        raise ValueError("duplicate frozen player-game-market originals")
    created=[]
    seen=set()
    for row in rows:
        k=key(row)
        if k in existing:
            continue
        if k in seen:
            raise ValueError("duplicate new player-game-market")
        seen.add(k)
        kickoff=_utc(row["kickoff_utc"])
        if not now<kickoff:
            continue
        if row["model"]!=VERSION or row["decision"]!="PASS":
            raise ValueError("invalid shadow model lineage")
        created.append({"first_observed_at_utc":now.isoformat(),**row})
    if created:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf8") as f:
            for r in created:
                f.write(json.dumps(r,sort_keys=True,allow_nan=False)+"\n")
    return {"existing":len(original),"new":len(created),"total":len(original)+len(created),
            "new_player_game_market_groups":len(created)}


def exact_line_probabilities(snapshot,catalog,line,*,observed_utc,
                             original_book_quote_utc=None):
    """Research-only probabilities; never compute ticket EV from editorial prices."""
    now=_utc(observed_utc)
    freeze=_utc(snapshot["first_observed_at_utc"])
    kickoff=_utc(snapshot["kickoff_utc"])
    if not freeze<=now<kickoff:
        raise ValueError("as-of outside original forecast-to-kickoff window")
    if snapshot["model"]!=VERSION or snapshot["residual_catalog_sha256"]!=catalog["sha256"]:
        raise ValueError("frozen distribution does not match immutable original catalog")
    market=snapshot["market"]
    group=snapshot["residual_group_key"]
    if not group.startswith(market+"|") or group not in catalog["vectors"]:
        raise ValueError("residual identity mismatch")
    errors=catalog["vectors"][group]
    if len(errors)!=snapshot["training_residual_count"] or not errors:
        raise ValueError("calibration sample mismatch")
    mean=float(snapshot["original_point_mean"])
    if not math.isclose(float(snapshot["original_scale"]),_scale(market,mean),abs_tol=1e-9):
        raise ValueError("modified stat scale")
    if isinstance(line,bool):
        raise ValueError("invalid line")
    price_line=float(line)
    if not math.isfinite(price_line) or price_line<0:
        raise ValueError("invalid nonnegative statistic line")
    if original_book_quote_utc is not None:
        quoted=_utc(original_book_quote_utc)
        if not freeze<=quoted<=now<kickoff:
            raise ValueError("original forecast must precede quote origin and kickoff")
    # For discrete NFL stat X: X>n uses n+.5; X>=n uses n-.5.
    if math.isclose(price_line % 1,.5,abs_tol=1e-9):
        over=over_probability(mean,price_line,market,errors)
        under=1-over
        push=0.
    elif float(price_line).is_integer():
        n=int(price_line)
        over=over_probability(mean,n+.5,market,errors)
        at_least=over_probability(mean,n-.5,market,errors) if n>0 else 1.
        under=1-at_least
        push=max(0.,at_least-over)
    else:
        raise ValueError("stat-line must be integer or half-point")
    assert abs(over+under+push-1.)<1e-9
    return {
        "model":VERSION,"decision":"PASS","game_id":snapshot["game_id"],
        "player_id":snapshot["player_id"],"market":market,"line":price_line,
        "original_forecast_utc":freeze.isoformat(),
        "query_observed_utc":now.isoformat(),
        "p_over_conditional_on_official_stat_row":round(over,9),
        "p_under_conditional_on_official_stat_row":round(under,9),
        "p_push_conditional_on_official_stat_row":round(push,9),
        "bookmaker_original_quote_utc_independently_verified":False,
        "betmgm_account_quote_accepted":False,
        "participation_or_void_verified":False,
        "calibrated_market_edge":False,"estimated_true_ev":None,
        "warning":"Full frozen empirical distribution is conditional on observed stats row; not an executable wager price.",
    }


def run(*,season=2026,horizon_days=10,
        calibrations_path=Path("history/player_prop_2024_residual_catalog_v1.json"),
        shadow_path=Path("history/player_prop_full_distributions_2026.jsonl"),
        report_path=Path("reports/player_prop_full_distribution_research.json")):
    if season!=2026:
        raise ValueError("distribution snapshot lineage registered only for 2026")
    now=datetime.now(timezone.utc)
    cli=NFLDataClient()
    years=[2024,2025,season]
    schedules=cli.load_schedules(years,refresh=True)
    stats=cli.load_player_stats(years,refresh=True)
    games,players,source=source_frames(schedules,stats)
    catalog,is_new=load_or_create_calibration(calibrations_path,games=games,players=players)
    weeks=future_weeks(schedules.to_dicts(),observed_utc=now.isoformat(),season=season,horizon_days=horizon_days)
    snapshots=[]
    for w in weeks:
        fresh=forecast_week(games,players,season=season,week=w["week"],
                            options=DEFAULT_OPTIONS,future_only_at=now)
        snapshots+=candidate_distributions(fresh,catalog)
    freeze=freeze_first_seen(snapshots,observed_utc=now,path=shadow_path)
    report={
        "model":VERSION,"decision":"PASS","research_only":True,
        "training_year":2024,"2025_previously_inspected":True,
        "2026_outcomes_used_to_fit":False,"calibration_newly_frozen":is_new,
        "catalog_sha256":catalog["sha256"],"catalog_groups":len(catalog["vectors"]),
        "future_eligible_weeks":[x["week"] for x in weeks],
        "first_seen_source_capture_utc":now.isoformat(),
        "forecast_groups_candidates":len(snapshots),
        "forward_records":freeze,
        "source_rows":source,
        "old_synthetic_v1_v2_journals_modified":False,
        "betmgm_executable_prices":0,
        "profitable_market_edge_confirmed":False,
        "approved_bets":0,"cash_wagers":0,
    }
    file=Path(report_path)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--horizon-days",type=int,default=10)
    ap.add_argument("--calibration",type=Path,
                    default=Path("history/player_prop_2024_residual_catalog_v1.json"))
    ap.add_argument("--shadow",type=Path,
                    default=Path("history/player_prop_full_distributions_2026.jsonl"))
    ap.add_argument("--report",type=Path,
                    default=Path("reports/player_prop_full_distribution_research.json"))
    args=ap.parse_args()
    result=run(season=args.season,horizon_days=args.horizon_days,
               calibrations_path=args.calibration,shadow_path=args.shadow,
               report_path=args.report)
    print(json.dumps({k:result[k] for k in (
        "model","decision","catalog_sha256","catalog_groups",
        "future_eligible_weeks","forward_records","approved_bets",
    )},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
