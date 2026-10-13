"""NFL correlated-score empirical-residual research only; NOT calibrated, nor a bet."""
from __future__ import annotations
import json, math, random
from datetime import UTC, datetime, timedelta

MIN_GAMES = 64
MIN_WEEKS = 8

def utc(value):
    try:
        t = datetime.fromisoformat(value.replace("Z","+00:00"))
        if t.tzinfo is None: raise ValueError()
        return t.astimezone(UTC)
    except (TypeError, AttributeError, ValueError) as e:
        raise ValueError("INVALID_UTC_TIME") from e

def number(obj, key, minimum=None):
    x=obj.get(key)
    if isinstance(x,bool): raise ValueError("INVALID_"+key)
    try: v=float(x)
    except (ValueError,TypeError) as e: raise ValueError("INVALID_"+key) from e
    if not math.isfinite(v) or (minimum is not None and v<minimum):
        raise ValueError("INVALID_"+key)
    return v

def percentile(xs, p):
    a=sorted(xs)
    return a[max(0,min(len(a)-1,math.ceil(len(a)*p)-1))]

def summary(xs):
    return {"mean":sum(xs)/len(xs),"median":percentile(xs,.5),
            "p10":percentile(xs,.1),"p90":percentile(xs,.9)}

def score_shadow(forecast, history, *, at, seed=20261012, draws=10000,
                 total_line=None, home_spread=None):
    """Paired historical final-score error bootstrap (NOT a drive simulator)."""
    if at.tzinfo is None: raise ValueError("UNZONED_ASOF")
    at=at.astimezone(UTC)
    if not isinstance(draws,int) or isinstance(draws,bool) or draws<1000:
        raise ValueError("INVALID_DRAWS")
    if not isinstance(seed,int) or isinstance(seed,bool):
        raise ValueError("INVALID_SEED")
    if not isinstance(forecast,dict) or forecast.get("record_kind")!="FROZEN_FIRST_SEEN_FORECAST":
        raise ValueError("ORIGINAL_FROZEN_FORECAST_REQUIRED")
    if forecast.get("league")!="NFL" or not forecast.get("event_id") or not forecast.get("content_sha256"):
        raise ValueError("FORECAST_IDENTITY_MISSING")
    first=utc(forecast["captured_utc"]); kick=utc(forecast["kickoff_utc"])
    if not first<kick or not first<=at<kick:
        raise ValueError("FORECAST_NOT_PREKICKOFF_AT_ASOF")
    home=number(forecast,"independent_home_points",0)
    away=number(forecast,"independent_away_points",0)
    if abs(home+away-number(forecast,"independent_projected_total"))>1e-6 or abs(home-away-number(forecast,"independent_projected_home_margin"))>1e-6:
        raise ValueError("FORECAST_SCORE_INCOHERENT")
    if not isinstance(history,list): raise ValueError("MISSING_HISTORY")
    pairs=[]; weeks=set(); ids=set()
    for row in history:
        if not isinstance(row,dict) or row.get("game_id") in ids or not row.get("game_id") or row["game_id"]==forecast["event_id"]:
            raise ValueError("HISTORY_ID_INVALID_OR_DUPLICATE")
        ids.add(row["game_id"])
        if row.get("independent_final_source_verified") is not True or not row.get("original_forecast_sha256") or not row.get("final_source_sha256"):
            raise ValueError("UNVERIFIED_HISTORICAL_SOURCE")
        ft=utc(row["forecast_captured_utc"])
        ko=utc(row["kickoff_utc"])
        grade=utc(row["final_first_verified_utc"])
        if not(ft<ko and ko+timedelta(hours=72)<=grade<=first):
            raise ValueError("LOOKAHEAD_OR_IMMATURE_HISTORICAL_GRADE")
        wk=(int(row["season"]),int(row["week"]))
        if wk[0]<2000 or not 1<=wk[1]<=22: raise ValueError("INVALID_HISTORY_WEEK")
        weeks.add(wk)
        eh=number(row,"actual_home_points",0)-number(row,"forecast_home_points",0)
        ea=number(row,"actual_away_points",0)-number(row,"forecast_away_points",0)
        if not (number(row,"actual_home_points").is_integer() and number(row,"actual_away_points").is_integer()):
            raise ValueError("INVALID_FINAL_SCORE")
        pairs.append((eh,ea))
    if len(pairs)<MIN_GAMES or len(weeks)<MIN_WEEKS:
        raise ValueError("INSUFFICIENT_VERIFIED_GAMES")
    for line in (total_line,home_spread):
        if line is not None and (isinstance(line,bool) or not isinstance(line,(int,float)) or not math.isfinite(line)):
            raise ValueError("INVALID_THRESHOLD")
    rand=random.Random(seed)
    hs=[]; aws=[]; ts=[]; ms=[]
    for _ in range(draws):
        eh,ea=rand.choice(pairs)
        h=max(0,math.floor(home+eh+.5))
        a=max(0,math.floor(away+ea+.5))
        hs.append(h);aws.append(a);ts.append(h+a);ms.append(h-a)
    out={"status":"UNVALIDATED_PAIRED_SCORE_SHADOW","method":"EMPIRICAL_PAIRED_RESIDUAL_REPLAY",
         "game_id":forecast["event_id"],"model_version":forecast["model_version"],
         "original_forecast_sha256":forecast["content_sha256"],
         "asof_utc":at.isoformat(),"original_capture_utc":first.isoformat(),
         "training_games":len(pairs),"training_weeks":len(weeks),
         "draws":draws,"seed":seed,"score_distributions":{
         "home":summary(hs),"away":summary(aws),"total":summary(ts),"home_margin":summary(ms)},
         "home_win":sum(m>0 for m in ms)/draws,"away_win":sum(m<0 for m in ms)/draws,
         "tie":sum(m==0 for m in ms)/draws,"thresholds":{},
         "is_drive_or_player_simulator":False,"accuracy_improvement_validated":False,
         "real_book_quote":None,"estimated_ev":None,"action":"PASS_RESEARCH_ONLY",
         "wager_authorized":False,"training_source_claims_independently_verified_by_code":False}
    if total_line is not None:
        out["thresholds"]["total"]={"line":total_line,
          "over":sum(t>total_line for t in ts)/draws,
          "push":sum(t==total_line for t in ts)/draws,
          "under":sum(t<total_line for t in ts)/draws}
    if home_spread is not None:
        out["thresholds"]["home_spread"]={"line":home_spread,
          "cover":sum(m+home_spread>0 for m in ms)/draws,
          "push":sum(m+home_spread==0 for m in ms)/draws,
          "fail":sum(m+home_spread<0 for m in ms)/draws}
    return out
