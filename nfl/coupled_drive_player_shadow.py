"""NFL joint DRIVE+PLAYER research shadow, with source-required prior drives.

No book odds, no live NFL source downloader, no refitting frozen original
forecasts and no market edge claims. Drive scoring, pass/run opportunities and
player props come from the SAME bootstrap drive draws.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from math import ceil, exp, floor, isfinite
import random

SPEC = "NFL_JOINT_DRIVE_PLAYER_RESEARCH_V1"
MIN_PRIOR_GAMES = 40
MIN_PRIOR_WEEKS = 6

def _utc(t):
    try:
        d = datetime.fromisoformat(t.replace("Z","+00:00"))
        if d.tzinfo is None: raise ValueError()
        return d.astimezone(UTC)
    except (ValueError,AttributeError,TypeError) as e:
        raise ValueError("BAD_TIME_OR_NO_UTC_OFFSET") from e

def _number(x, *, minimum=None):
    if isinstance(x,bool): raise ValueError("BAD_NUMBER")
    try: v=float(x)
    except (TypeError,ValueError,OverflowError) as e:
        raise ValueError("BAD_NUMBER") from e
    if not isfinite(v) or (minimum is not None and v<minimum):
        raise ValueError("BAD_NUMBER")
    return v

def _int(x, *, minimum=0):
    v=_number(x,minimum=minimum)
    if int(v)!=v: raise ValueError("COUNT_NOT_INTEGER")
    return int(v)

def _quantile(v,p):
    s=sorted(v)
    return s[min(len(s)-1,max(0,ceil(len(s)*p)-1))]

def _describe(v):
    return {"mean":sum(v)/len(v),"p10":_quantile(v,.10),
            "median":_quantile(v,.50),"p90":_quantile(v,.90)}

def _validate_history(drives, *, first_seen):
    if not isinstance(drives,list): raise ValueError("DRIVES_REQUIRED")
    seen=set();games={};library=[]
    for d in drives:
        if not isinstance(d,dict): raise ValueError("BAD_DRIVE")
        game=d.get("game_id"); drive=d.get("drive_id"); offense=d.get("team")
        key=(game,drive)
        if not game or drive is None or not offense or key in seen: raise ValueError("DUPLICATE_DRIVE_OR_ID")
        seen.add(key)
        if d.get("completed_nflverse_schedule_game") is not True or not d.get("source_sha256"):
            raise ValueError("NFLVERSE_SOURCE_COMPLETION_NOT_ATTESTED")
        kickoff=_utc(d.get("kickoff_utc"))
        verified=_utc(d.get("verified_at_utc"))
        if not (kickoff+timedelta(hours=72)<=verified<=first_seen):
            raise ValueError("LOOKAHEAD_OR_UNMATURED_DRIVE")
        week=(_int(d.get("season"),minimum=2000),_int(d.get("week"),minimum=1))
        if week[1]>22: raise ValueError("BAD_WEEK")
        previous=games.get(game)
        if previous and previous[:3]!=(week[0],week[1],kickoff): raise ValueError("CONFLICTING_GAME_IDENTITY")
        games[game]=(week[0],week[1],kickoff)
        score=_int(d.get("points"))
        if score not in (0,3,7): raise ValueError("UNSUPPORTED_DRIVE_SCORE")
        pa=_int(d.get("pass_attempts")); co=_int(d.get("completions"))
        ya=_int(d.get("passing_yards")); ru=_int(d.get("rush_attempts"))
        ry=int(_number(d.get("rushing_yards")))
        if co>pa or (co==0 and ya!=0) or pa+ru==0 and score>0:
            raise ValueError("INVALID_DRIVE_PBP_ACCOUNTING")
        library.append({"game":game,"team":offense,"points":score,
                        "pass_attempts":pa,"completions":co,
                        "passing_yards":ya,"rush_attempts":ru,"rushing_yards":ry})
    weeks={(s,w) for s,w,_ in games.values()}
    if len(games)<MIN_PRIOR_GAMES or len(weeks)<MIN_PRIOR_WEEKS:
        raise ValueError("INSUFFICIENT_VERIFIED_DRIVE_GAMES")
    pergame=defaultdict(lambda:defaultdict(int))
    for d in library: pergame[d["game"]][d["team"]]+=1
    possible=[v for v in pergame.values() if len(v)==2 and all(4<=n<=30 for n in v.values())]
    if len(possible)<MIN_PRIOR_GAMES: raise ValueError("INCOMPLETE_TWO_TEAM_DRIVE_LIBRARY")
    return library,[tuple(v.values()) for v in possible],len(games),len(weeks)

def _role_weights(roles, *, team, kind, first_seen):
    eligible=[r for r in roles if r.get("team")==team and r.get("role") in ("QB","RB","WR","TE","FB")]
    key="target_share" if kind=="target" else "carry_share"
    weighted=[]
    for r in eligible:
        if not r.get("player_id") or _utc(r.get("observed_at_utc"))>first_seen:
            raise ValueError("PLAYER_ROSTER_LOOKAHEAD_OR_MISSING_ID")
        if r.get("activity") not in ("CONFIRMED_ACTIVE","UNKNOWN"):
            raise ValueError("PLAYER_ACTIVITY_STATUS_INVALID")
        weight=_number(r.get(key,0),minimum=0)
        if weight>1: raise ValueError("PLAYER_SHARE_TOO_LARGE")
        if weight: weighted.append((str(r["player_id"]),weight))
    total=sum(w for _,w in weighted)
    if total==0: return []
    if total>1.000001: raise ValueError("PLAYER_SHARES_EXCEED_TEAM")
    return weighted+([("OTHER",max(0,1-total))] if total<1 else [])

def _choose(weights, rand):
    x=rand.random()*sum(w for _,w in weights)
    for name,w in weights:
        x-=w
        if x<=0: return name
    return weights[-1][0]

def _split(total,n,rand):
    if n==0:
        if total!=0: raise ValueError("NONZERO_YARDS_WITHOUT_ATTEMPT")
        return []
    if total==0:return [0]*n
    sign=-1 if total<0 else 1
    raw=[rand.expovariate(1) for _ in range(n)]
    scale=sum(raw)
    chunks=[int(abs(total)*x/scale) for x in raw]
    remainder=abs(total)-sum(chunks)
    for i in sorted(range(n),key=lambda i:raw[i]/scale*abs(total)-chunks[i],reverse=True)[:remainder]:
        chunks[i]+=1
    return [sign*c for c in chunks]

def _tilt_weights(drives, target_ppd):
    # Entropic projection of prior P(points), *not* validated predicted accuracy.
    if not .05<=target_ppd<=6.95: raise ValueError("UNSUPPORTED_EXPECTED_POINTS_PER_DRIVE")
    vals={p:sum(x["points"]==p for x in drives) for p in (0,3,7)}
    if not all(vals.values()): raise ValueError("MISSING_REPLAY_SCORING_CLASS")
    lo,hi=-8.0,8.0
    for _ in range(60):
        mid=(lo+hi)/2
        terms={p:vals[p]*exp(mid*p) for p in vals}
        expected=sum(p*w for p,w in terms.items())/sum(terms.values())
        if expected<target_ppd:lo=mid
        else:hi=mid
    midpoint=(lo+hi)/2
    # Calculate only three weights, normalize for safe repeated random draws.
    class_weights={p:vals[p]*exp(midpoint*p) for p in vals}
    total=sum(class_weights.values())
    by_score=defaultdict(list)
    for row in drives:by_score[row["points"]].append(row)
    return [(p,class_weights[p]/total,by_score[p]) for p in (0,3,7)]

def _sample_drive(weighted,rng):
    r=rng.random()
    for p,w,rows in weighted:
        r-=w
        if r<=0:return rng.choice(rows)
    return rng.choice(weighted[-1][2])

def simulate_joint(
    forecast, drives, players, *, asof, draws=2000, seed=42,
    queries=(),
):
    """Game/props research simulation, requires original pregame score and past drives."""
    if asof.tzinfo is None: raise ValueError("UNZONED_ASOF")
    at=asof.astimezone(UTC)
    if isinstance(draws,bool) or not isinstance(draws,int) or not 1000<=draws<=100000:
        raise ValueError("DRAWS_OUT_OF_RANGE")
    if isinstance(seed,bool) or not isinstance(seed,int):raise ValueError("INVALID_SEED")
    if not isinstance(forecast,dict) or forecast.get("record_kind")!="FROZEN_FIRST_SEEN_FORECAST":
        raise ValueError("FROZEN_FORECAST_REQUIRED")
    original=_utc(forecast.get("captured_utc"))
    kickoff=_utc(forecast.get("kickoff_utc"))
    if not original<=at<kickoff or original>=kickoff or not forecast.get("content_sha256"):
        raise ValueError("FORECAST_NOT_ORIGINAL_PREKICKOFF")
    home,away=forecast.get("home_team"),forecast.get("away_team")
    if not home or not away or home==away:raise ValueError("INVALID_GAME_IDENTITY")
    eh=_number(forecast.get("independent_home_points"),minimum=0)
    ea=_number(forecast.get("independent_away_points"),minimum=0)
    if abs(eh+ea-_number(forecast.get("independent_projected_total")))>1e-6:
        raise ValueError("FORECAST_TOTAL_INCOHERENT")
    if abs(eh-ea-_number(forecast.get("independent_projected_home_margin")))>1e-6:
        raise ValueError("FORECAST_MARGIN_INCOHERENT")
    lib,pace,n_games,n_weeks=_validate_history(drives,first_seen=original)
    # Model training never sees the target original score, kickoff or season result.
    by_score_home={}; player_weights={}
    for team in (home,away):
        player_weights[team]={
            "target":_role_weights(players,team=team,kind="target",first_seen=original),
            "carry":_role_weights(players,team=team,kind="carry",first_seen=original),
        }
        if not player_weights[team]["target"] or not player_weights[team]["carry"]:
            raise ValueError("MISSING_TEAM_PLAYER_SHARE_COVERAGE")
    mean_pace=sum((v[0]+v[1])/2 for v in pace)/len(pace)
    for team,points in ((home,eh),(away,ea)):
        by_score_home[team]=_tilt_weights(lib,points/mean_pace)
    rand=random.Random(seed)
    series=defaultdict(list)
    for _ in range(draws):
        counts=rand.choice(pace)
        if rand.random()<.5:counts=counts[::-1]
        perteam={}
        for team,n in zip((home,away),counts):
            rows=[_sample_drive(by_score_home[team],rand) for j in range(n)]
            pts=sum(x["points"] for x in rows)
            pa=sum(x["pass_attempts"] for x in rows)
            co=sum(x["completions"] for x in rows)
            py=sum(x["passing_yards"] for x in rows)
            ru=sum(x["rush_attempts"] for x in rows)
            ry=sum(x["rushing_yards"] for x in rows)
            receiving=defaultdict(lambda:{"receptions":0,"receiving_yards":0})
            for yd in _split(py,co,rand):
                name=_choose(player_weights[team]["target"],rand)
                receiving[name]["receptions"]+=1
                receiving[name]["receiving_yards"]+=yd
            rushing=defaultdict(lambda:{"carries":0,"rushing_yards":0})
            for yd in _split(ry,ru,rand):
                name=_choose(player_weights[team]["carry"],rand)
                rushing[name]["carries"]+=1
                rushing[name]["rushing_yards"]+=yd
            if sum(p["receptions"] for p in receiving.values())!=co:
                raise AssertionError("RECEIVING_COUNTS_DO_NOT_RECONCILE")
            if sum(p["receiving_yards"] for p in receiving.values())!=py:
                raise AssertionError("RECEIVING_YARDS_DO_NOT_RECONCILE")
            if sum(p["carries"] for p in rushing.values())!=ru:
                raise AssertionError("RUSHING_COUNTS_DO_NOT_RECONCILE")
            if sum(p["rushing_yards"] for p in rushing.values())!=ry:
                raise AssertionError("RUSH_YARDS_DO_NOT_RECONCILE")
            perteam[team]=pts
            for k,v in (("points",pts),("pass_attempts",pa),("completions",co),
                        ("passing_yards",py),("rush_attempts",ru),("rushing_yards",ry)):
                series[f"{team}:{k}"].append(v)
            for player,v in receiving.items():
                for stat,x in v.items():series[f"{team}:{player}:{stat}"].append(x)
            for player,v in rushing.items():
                for stat,x in v.items():series[f"{team}:{player}:{stat}"].append(x)
            # Missing player recorded as 0 simulated usage (not actual betting DNP).
            for role in players:
                if role.get("team")!=team:continue
                person=role["player_id"]
                for stat in ("receptions","receiving_yards","carries","rushing_yards"):
                    key=f"{team}:{person}:{stat}"
                    if len(series[key])<=_:series[key].append(0)
        h,a=perteam[home],perteam[away]
        series["TOTAL"].append(h+a);series["HOME_MARGIN"].append(h-a)
    result={"spec":SPEC,"status":"UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH",
            "game_id":forecast["event_id"],"forecast_sha256":forecast["content_sha256"],
            "asof_utc":at.isoformat(),"source_model":forecast["model_version"],
            "training_games":n_games,"training_weeks":n_weeks,"draws":draws,"seed":seed,
            "distribution":{key:_describe(v) for key,v in sorted(series.items()) if len(v)==draws},
            "home_win":sum(x>0 for x in series["HOME_MARGIN"])/draws,
            "away_win":sum(x<0 for x in series["HOME_MARGIN"])/draws,
            "tie":sum(x==0 for x in series["HOME_MARGIN"])/draws,
            "query_thresholds":[],"assumed_pregame_player_activity":"UNVERIFIED",
            "player_stats_conditional_scenarios_only":True,
            "excluded_from_official_forecast":True,"accuracy_improvement_validated":False,
            "source_attestations_independently_authenticated":False,
            "game_model_not_yet_qb_roof_weather_or_ot_validated":True,
            "estimated_ev":None,"wager_authorized":False,"action":"PASS_RESEARCH_ONLY"}
    for q in queries:
        metric=q.get("metric")
        if metric not in series or len(series[metric])!=draws:raise ValueError("UNKNOWN_OR_INCOMPLETE_QUERY_METRIC")
        threshold=_number(q.get("line"))
        z=series[metric]
        result["query_thresholds"].append({
            "metric":metric,"line":threshold,
            "over":sum(y>threshold for y in z)/draws,
            "push":sum(y==threshold for y in z)/draws,
            "under":sum(y<threshold for y in z)/draws,
            "betting_ev":None})
    return result
