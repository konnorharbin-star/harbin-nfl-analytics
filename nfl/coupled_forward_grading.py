"""Grade frozen NFL joint game/prop research without retrospectively refitting.

Source: a PREKICKOFF GitHub Actions JSON artifact, mature postgame nflverse
schedules and player weekly game rows. The nflverse scoreboard is not official
NFL gamebook verification, and absent player rows are NEVER scored as zero.
"""
from __future__ import annotations
from datetime import UTC, datetime, timedelta
import math

from nfl.coupled_nflverse_asof import kickoff_utc

SPEC="NFL_JOINT_DRIVE_PLAYER_FORWARD_GRADE_RESEARCH_V1"
MIN_GAMES_FOR_INDEPENDENT_REVIEW=128
# Map research metric names to the actual nflverse weekly player stat fields.
# Never interpret an absent row or absent column as a zero or prop settlement.
PLAYER_STATS={
    "receptions":"receptions",
    "receiving_yards":"receiving_yards",
    "receiving_touchdowns":"receiving_tds",
    "carries":"carries",
    "rushing_yards":"rushing_yards",
    "rushing_touchdowns":"rushing_tds",
    "passing_attempts":"attempts",
    "passing_completions":"completions",
    "passing_yards":"passing_yards",
    "passing_touchdowns":"passing_tds",
    "passing_interceptions":"passing_interceptions",
}

def clock(x):
    if not isinstance(x,str):raise ValueError("TIMESTAMP_MISSING")
    try: dt=datetime.fromisoformat(x.replace("Z","+00:00"))
    except ValueError as e:raise ValueError("INVALID_TIMESTAMP") from e
    if dt.tzinfo is None:raise ValueError("UNZONED_TIMESTAMP")
    return dt.astimezone(UTC)

def num(x):
    if isinstance(x,bool):raise ValueError("NUMERIC_BOOL")
    try:v=float(x)
    except (ValueError,TypeError) as e:raise ValueError("NUMERIC_INVALID") from e
    if not math.isfinite(v):raise ValueError("NUMERIC_NONFINITE")
    return v

def grade_snapshot(snapshot, finals, player_rows, *, at):
    if at.tzinfo is None:raise ValueError("UNZONED_GRADE_CLOCK")
    now=at.astimezone(UTC)
    if snapshot.get("spec")!="NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1":
        raise ValueError("WRONG_FORECAST_FAMILY")
    captured=clock(snapshot.get("capture_utc"))
    if captured>now:raise ValueError("SNAPSHOT_LATER_THAN_GRADING")
    final_map={}
    for f in finals:
        gid=f.get("game_id")
        if not gid:continue
        if gid in final_map:raise ValueError("DUPLICATE_FINAL_GAME")
        final_map[gid]=f
    # Public nflverse load_player_stats(season) is WEEKLY and normally has
    # NO game_id. Use a strict REG season/week/recent_team -> schedule game
    # lookup, and only if exactly one valid fixture fits. Preserve original
    # game_id rows as a supported explicit-source path, with the same identity
    # checks downstream. Never match by player name or assume missing = zero.
    team_games={}
    ambiguous_team_weeks=set()
    for f in finals:
        if f.get("game_type")!="REG":
            continue
        gid=f.get("game_id")
        season,week=f.get("season"),f.get("week")
        # Some provided scoreboard test rows omit season/week; weekly joins
        # cannot use these without an explicit season/week schedule identity.
        if not gid or season is None or week is None:
            continue
        for team in (f.get("home_team"),f.get("away_team")):
            if not team:continue
            key=(str(season),str(week),str(team))
            if key in team_games and team_games[key]!=gid:
                ambiguous_team_weeks.add(key)
            team_games[key]=gid
    player_map={};invalid_players=set()
    identity_uncertain_rows=0
    for p in player_rows:
        pid=p.get("player_id")
        if not pid:continue
        team=p.get("recent_team") or p.get("team")
        if not team:continue
        gid=p.get("game_id")
        if gid is None:
            season,week=p.get("season"),p.get("week")
            if season is None or week is None:
                identity_uncertain_rows+=1
                continue
            match=(str(season),str(week),str(team))
            if match in ambiguous_team_weeks:
                identity_uncertain_rows+=1
                continue
            gid=team_games.get(match)
            if gid is None:
                identity_uncertain_rows+=1
                continue
        f=final_map.get(gid)
        if not f or f.get("game_type")!="REG":
            identity_uncertain_rows+=1
            continue
        if team not in (f.get("home_team"),f.get("away_team")):
            identity_uncertain_rows+=1
            continue
        if p.get("season_type") not in (None,"REG"):
            identity_uncertain_rows+=1
            continue
        if (p.get("season") is not None and f.get("season") is not None
            and str(p["season"])!=str(f["season"])):
            identity_uncertain_rows+=1
            continue
        if (p.get("week") is not None and f.get("week") is not None
            and str(p["week"])!=str(f["week"])):
            identity_uncertain_rows+=1
            continue
        opponent=f["away_team"] if team==f["home_team"] else f["home_team"]
        if p.get("opponent_team") not in (None,opponent):
            identity_uncertain_rows+=1
            continue
        key=(str(gid),str(pid))
        if key in player_map:invalid_players.add(key)
        player_map[key]=p
    rows=[];counts={"not_mature":0,"final_unverified":0,"identity_error":0,
                    "invalid_original":0,"graded":0,"player_stat_rows_not_eligible":0}
    seen=set()
    for record in snapshot.get("games",[]):
        gid=record.get("game_id")
        if not gid or gid in seen:raise ValueError("DUPLICATE_OR_EMPTY_PREGAME_GAME")
        seen.add(gid)
        if record.get("status")!="UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH":
            counts["invalid_original"]+=1
            continue
        original=record.get("frozen_football_score_input") or {}
        try:
            kickoff=clock(original["kickoff_utc"])
            first=clock(original["captured_utc"])
            if original.get("event_id")!=gid or not original.get("content_sha256"):
                raise ValueError("BROKEN_FROZEN_FORECAST_IDENTITY")
            if not(first<=captured<kickoff) or record.get("game_id")!=gid:
                raise ValueError("POSTKICKOFF_OR_UNFROZEN")
        except (ValueError,KeyError):
            counts["invalid_original"]+=1
            continue
        if now<kickoff+timedelta(hours=72):
            counts["not_mature"]+=1
            continue
        f=final_map.get(gid)
        if not f or f.get("game_type")!="REG" or f.get("home_score") is None or f.get("away_score") is None:
            counts["final_unverified"]+=1
            continue
        try:
            if kickoff_utc(f)!=kickoff or f.get("home_team")!=original["home_team"] or f.get("away_team")!=original["away_team"]:
                counts["identity_error"]+=1
                continue
            home_actual=num(f["home_score"]);away_actual=num(f["away_score"])
            if min(home_actual,away_actual)<0 or not home_actual.is_integer() or not away_actual.is_integer():
                raise ValueError("BAD_FINAL_SCORE")
            total=home_actual+away_actual;margin=home_actual-away_actual
            expected=original.get("independent_projected_total")
            expected_margin=original.get("independent_projected_home_margin")
            dist=record["distribution"]
            shadow_total=num(dist["TOTAL"]["mean"])
            shadow_margin=num(dist["HOME_MARGIN"]["mean"])
            baseline_total=num(expected)
            baseline_margin=num(expected_margin)
            probs=[num(record[key]) for key in ("home_win","away_win","tie")]
            if abs(sum(probs)-1)>1e-6 or min(probs)<0:
                raise ValueError("INVALID_PREDICTIVE_PROBABILITY")
            truth=[int(margin>0),int(margin<0),int(margin==0)]
            multiclass_brier=sum((a-b)**2 for a,b in zip(probs,truth))
        except (KeyError,TypeError,ValueError):
            counts["invalid_original"]+=1
            continue
        prop_stats=[]
        for key,info in dist.items():
            parts=key.split(":")
            if len(parts)!=3 or parts[2] not in PLAYER_STATS:continue
            team,pid,stat=parts
            source_key=(gid,pid)
            p=player_map.get(source_key)
            if source_key in invalid_players or p is None:
                counts["player_stat_rows_not_eligible"]+=1
                continue
            source_field=PLAYER_STATS[stat]
            if (p.get("recent_team") or p.get("team"))!=team or p.get(source_field) is None:
                counts["player_stat_rows_not_eligible"]+=1
                continue
            try:
                actual=num(p[source_field])
                forecast_value=num(info["mean"])
            except (ValueError,KeyError,TypeError):
                counts["player_stat_rows_not_eligible"]+=1
                continue
            prop_stats.append({"player_id":pid,"team":team,"stat":stat,
                               "source_field":source_field,
                               "actual":actual,"predicted":forecast_value,
                               "absolute_error":abs(actual-forecast_value),
                               "status":"RESEARCH_SOURCE_ROW_ONLY_NOT_BET_SETTLEMENT"})
        rows.append({
            "game_id":gid,"model_original_sha256":original["content_sha256"],
            "forecast_captured_utc":first.isoformat(),"kickoff_utc":kickoff.isoformat(),
            "graded_asof_utc":now.isoformat(),"final_home":int(home_actual),
            "final_away":int(away_actual),"actual_total":int(total),"actual_margin":int(margin),
            "canonical_total_mae":abs(baseline_total-total),
            "shadow_total_mae":abs(shadow_total-total),
            "canonical_margin_mae":abs(baseline_margin-margin),
            "shadow_margin_mae":abs(shadow_margin-margin),
            "total_mae_delta_baseline_minus_shadow":abs(baseline_total-total)-abs(shadow_total-total),
            "margin_mae_delta_baseline_minus_shadow":abs(baseline_margin-margin)-abs(shadow_margin-margin),
            "shadow_three_way_brier":multiclass_brier,
            "qualified_player_stat_rows":prop_stats,
            "independently_official_final_confirmed":False,
            "eligible_for_profitability_registry":False})
        counts["graded"]+=1
    n=len(rows)
    metrics=None
    if n:
        metrics={k:sum(r[k] for r in rows)/n for k in (
            "canonical_total_mae","shadow_total_mae","canonical_margin_mae",
            "shadow_margin_mae","total_mae_delta_baseline_minus_shadow",
            "margin_mae_delta_baseline_minus_shadow","shadow_three_way_brier"
        )}
        metrics["n_independent_games"]=n
        metrics["unique_player_stats_graded"]=sum(len(r["qualified_player_stat_rows"]) for r in rows)
    return {"spec":SPEC,"status":"RESEARCH_GRADE_NOT_PROFIT_EVIDENCE",
            "asof_utc":now.isoformat(),"original_forecast_capture_utc":captured.isoformat(),
            "counts":counts,"metrics":metrics,"games":rows,
            "weekly_rows_with_uncertain_identity":identity_uncertain_rows,
            "player_join_rule":"EXACT_REG_SEASON_WEEK_TEAM_SCHEDULE_ID",
            "player_grades_are_bet_settlement_evidence":False,
            "promotion_ready":False,"minimum_forward_games":MIN_GAMES_FOR_INDEPENDENT_REVIEW,
            "official_NFL_gamebook_crosscheck_done":False,
            "player_void_rules_verified":False,
            "sportsbook_price_and_roi_graded":False,
            "no_original_predictions_modified":True,"wager_authorized":False}
