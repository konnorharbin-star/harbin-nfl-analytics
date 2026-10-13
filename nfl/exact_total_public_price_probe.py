"""Exact original-source NFL full-game total vs public price research probe.

Only recomputes threshold distributions using identical frozen historical drives,
roles, seed, draws and first-seen forecast. Public aggregator prices cannot
prove bookmaker execution or pregame offer-origin time. Raw model EV is NOT
a calibrated or profitable betting recommendation.
"""
from __future__ import annotations
from datetime import datetime,UTC
from math import isfinite,sqrt
from hashlib import sha256
import json

from nfl.coupled_drive_player_shadow import simulate_joint

SPEC="NFL_TOTAL_EXACT_PUBLIC_PRICE_RESEARCH_V1"

def _utc(value):
    if not isinstance(value,str):raise ValueError("MISSING_DATETIME")
    try:d=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:raise ValueError("INVALID_DATETIME") from exc
    if d.tzinfo is None:raise ValueError("UNZONED_DATETIME")
    return d.astimezone(UTC)

def _decimal(american):
    if isinstance(american,bool) or not isinstance(american,int) or (-100<american<100):
        raise ValueError("INVALID_AMERICAN_PRICE")
    return 1+(american/100 if american>0 else 100/abs(american))

def _wilson(p,n,z=1.96):
    if not isinstance(n,int) or n<1:raise ValueError("INVALID_MONTE_CARLO_SIZE")
    center=p+z*z/(2*n)
    margin=z*sqrt((p*(1-p)+z*z/(4*n))/n)
    denom=1+z*z/n
    return [max(0,(center-margin)/denom),min(1,(center+margin)/denom)]

def compare(original,replay,quotes,*,now,simulate=simulate_joint):
    """Research dislocations are hypotheses; no positive-EV certification."""
    if now.tzinfo is None:raise ValueError("NAIVE_RESEARCH_CLOCK")
    at=now.astimezone(UTC)
    if original.get("spec")!="NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1":
        raise ValueError("WRONG_ORIGINAL_FOOTBALL_ARTIFACT")
    if replay.get("spec")!="NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1":
        raise ValueError("WRONG_FROZEN_REPLAY_INPUTS")
    if not isinstance(quotes,list):raise ValueError("QUOTE_ROWS_REQUIRED")
    captured=_utc(original["capture_utc"])
    if _utc(replay["capture_utc"])!=captured or captured>at:
        raise ValueError("ORIGINAL_REPLAY_CAPTURE_MISMATCH")
    if (original["season"],original["week"])!=(replay["season"],replay["week"]):
        raise ValueError("SEASON_WEEK_MISMATCH")
    games={g["game_id"]:g for g in original["games"]
           if g.get("status")=="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH"}
    inputs={g["game_id"]:g for g in replay["games"]}
    if len(games)!=sum(g.get("status")=="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH"
                      for g in original["games"]) or len(inputs)!=len(replay["games"]):
        raise ValueError("DUPLICATE_ORIGINAL_GAME")
    if set(games)!=set(inputs):raise ValueError("INPUT_COHORT_IDENTITY_MISMATCH")
    reports=[]
    seen=set()
    for q in quotes:
        if q.get("market")!="NFL_FULL_GAME_TOTAL":
            raise ValueError("TOTAL_MARKET_ONLY")
        if q.get("period")!="REGULATION_AND_OVERTIME":
            raise ValueError("MISMATCHED_MARKET_PERIOD")
        gid=q["game_id"]
        if gid in seen:raise ValueError("DUPLICATE_MARKET_CONTRACT")
        seen.add(gid)
        if gid not in games:raise ValueError("NO_FROZEN_MATCHING_GAME")
        g=games[gid];inputs_g=inputs[gid];f=inputs_g["forecast"]
        if g["forecast_sha256"]!=f["content_sha256"] or (
            g["frozen_football_score_input"]["content_sha256"]!=f["content_sha256"]
        ) or g["game_id"]!=f["event_id"]:
            raise ValueError("FROZEN_FORECAST_IDENTITY_MISMATCH")
        if (f["home_team"]!=q["home_team"] or f["away_team"]!=q["away_team"]):
            raise ValueError("WRONG_HOME_AWAY_OR_EVENT_ID")
        model_time=_utc(f["captured_utc"])
        quote_time=_utc(q["collector_observed_utc"])
        kickoff=_utc(f["kickoff_utc"])
        if not model_time<=captured<=quote_time<=at<kickoff:
            raise ValueError("NONPROSPECTIVE_OR_FUTURE_QUOTE")
        if not isinstance(q.get("source_url"),str) or not q["source_url"].startswith("https://"):
            raise ValueError("MISSING_SOURCE_URL")
        if q.get("source_quote_origin_utc") is not None or q.get("account_executable") is not False:
            raise ValueError("PUBLIC_QUOTE_NOT_AUTHENTICATED_EXECUTION")
        line=q.get("line")
        if isinstance(line,bool) or not isinstance(line,(int,float)) or not isfinite(line) or (
            line<0 or abs(line*2-round(line*2))>1e-8
        ):raise ValueError("INVALID_TOTAL_LINE")
        over,under=q["over_american"],q["under_american"]
        over_d,under_d=_decimal(over),_decimal(under)
        replayed=simulate(
            f,inputs_g["drives"],inputs_g["roles"],asof=captured,
            draws=g["draws"],seed=g["seed"],
            queries=[{"metric":"TOTAL","line":line}],
        )
        same_original=("TOTAL","HOME_MARGIN")
        for key in same_original:
            a=replayed["distribution"][key]["mean"]
            b=g["distribution"][key]["mean"]
            if abs(a-b)>1e-9:
                raise ValueError("FROZEN_REPLAY_MISMATCH")
        if replayed["forecast_sha256"]!=g["forecast_sha256"]:
            raise ValueError("MODEL_IDENTITY_CHANGED_DURING_REPLAY")
        probs=replayed["query_thresholds"][0]
        if abs(probs["over"]+probs["under"]+probs["push"]-1)>1e-8:
            raise ValueError("BAD_REPLAY_PROBABILITIES")
        implied_over=1/over_d
        implied_under=1/under_d
        # Market no-vig only if the SAME sportsbook actually offered both.
        # This public aggregator quote does NOT certify that; never infer.
        p_over,p_under,p_push=probs["over"],probs["under"],probs["push"]
        candidates={}
        for side,win,loss,price,decimal_odds,implied in (
            ("over",p_over,p_under,over,over_d,implied_over),
            ("under",p_under,p_over,under,under_d,implied_under),
        ):
            net=decimal_odds-1
            raw_ev=win*net-loss  # refund push; assumes ordinary total settlement
            candidates[side]={
                "line":line,"american_price":price,
                "model_raw_win_probability":win,
                "model_raw_push_probability":p_push,
                "model_raw_loss_probability":loss,
                "advertised_price_break_even_win_probability_if_no_push":implied,
                "model_minus_implied_no_push_gap":win-implied,
                "hypothetical_raw_model_ev_per_unit":raw_ev,
                "monte_carlo_only_win_interval_95":_wilson(win,g["draws"]),
                "calibration_uncertainty_quantified":False,
                "price_execution_verified":False,
                "bookmaker_rules_independently_verified":False,
                "validated_profitable_edge":False,
                "bet_authorized":False,
            }
        reports.append({
            "game_id":gid,"matchup":f"{f['away_team']} @ {f['home_team']}",
            "model_original_sha256":f["content_sha256"],
            "model_capture_utc":model_time.isoformat(),
            "source_quote_url":q["source_url"],
            "public_collector_observed_utc":quote_time.isoformat(),
            "source_quote_origin_time":None,
            "book_origin_and_same_book_pair_verified":False,
            "line":line,"source_quote_descriptor":q.get("source_description"),
            "research_model_total_mean":replayed["distribution"]["TOTAL"]["mean"],
            "samples":g["draws"],"sides":candidates,
            "settlement_model_includes_all_unusual_ot_scoring":False,
            "decision":"PASS_MODEL_NOT_CALIBRATED_AND_PUBLIC_QUOTE_UNEXECUTABLE",
        })
    return {
        "spec":SPEC,"status":"RESEARCH_ONLY_NOT_PROFITABLE_EDGE_PROOF",
        "observed_utc":at.isoformat(),"games_examined":len(reports),
        "distinct_independent_forward_grades_generated":0,
        "historical_market_superiority_validated":False,
        "price_origin_authenticated":False,
        "positive_ev_betting_recommendations":0,
        "betting_authorized":False,"action":"PASS",
        "quote_pair_observations":reports,
    }
