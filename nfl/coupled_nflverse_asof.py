"""Source-as-of nflverse PBP -> replayable paired drive/role research inputs.

The final game's actual nflverse schedule and the original PBP source are
captured at run time; files are NOT retroactively presented as historical
first-seen archives. Excludes any game younger than 72h at model capture.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from zoneinfo import ZoneInfo

import polars as pl

from nfl.coupled_drive_player_shadow import _utc, _number
from nfl.ratings import fit_pregame_fair_score


def kickoff_utc(game: dict) -> datetime:
    """nflverse gameday/gametime are Eastern local, including DST."""
    original=game.get("kickoff_utc")
    if original:
        return _utc(str(original))
    date=game.get("gameday")
    clock=game.get("gametime")
    if not date or not clock:
        raise ValueError("MISSING_SOURCE_KICKOFF_DATE_TIME")
    try:
        local=datetime.fromisoformat(str(date)+"T"+str(clock))
    except ValueError as e:
        raise ValueError("MALFORMED_SOURCE_KICKOFF") from e
    if local.tzinfo is None:
        local=local.replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(UTC)


def prepare_asof_source(schedules, pbp, player_stats, *,
                        season: int, week: int, game_id: str, at: datetime):
    """Build real-source read-only research inputs with cutoff provenance.

    No sportsbook data. No claimed independently verified QB starter/active
    player or original upstream publish time.
    """
    if at.tzinfo is None: raise ValueError("UNZONED_SOURCE_CAPTURE")
    at=at.astimezone(UTC)
    if not isinstance(season,int) or not isinstance(week,int):raise ValueError("BAD_WEEK")
    schedule_rows=schedules.to_dicts()
    target=[r for r in schedule_rows if str(r.get("game_id"))==game_id
            and r.get("season")==season and r.get("week")==week]
    if len(target)!=1:raise ValueError("MISSING_OR_DUPLICATE_SCHEDULE_GAME")
    game=target[0]
    kickoff=kickoff_utc(game)
    if at>=kickoff or game.get("game_type")!="REG":
        raise ValueError("SOURCE_NOT_PROSPECTIVE_REGULAR_SEASON_GAME")
    home,away=game["home_team"],game["away_team"]
    if home==away:raise ValueError("INVALID_TEAMS")
    eligible={}
    for row in schedule_rows:
        if row.get("game_type")!="REG" or row.get("game_id")==game_id:
            continue
        try: ko=kickoff_utc(row)
        except ValueError:continue
        if ko+timedelta(hours=72)>at:
            continue
        if row.get("home_score") is None or row.get("away_score") is None:
            continue
        if row.get("season")>season or (
            row.get("season")==season and row.get("week",999)>=week):
            continue
        gid=str(row.get("game_id"))
        if gid in eligible:raise ValueError("DUPLICATED_SCHEDULE_GAME")
        eligible[gid]=(row,ko)
    if len(eligible)<40:
        raise ValueError("INSUFFICIENT_MATURE_SCHEDULE_GAMES")
    # Force the scoring baseline to fit ONLY source-mature, prior weeks.
    # Do not alter published original scoring forecast files.
    eligible_ids=set(eligible)
    strict_schedule=schedules.filter(pl.col("game_id").cast(pl.String).is_in(list(eligible_ids|{game_id})))
    fair=fit_pregame_fair_score(strict_schedule,season,week)
    projection=fair.project(home,away)
    sha=sha256(json.dumps({
        "asof":at.isoformat(),"game_id":game_id,
        "model_home":projection.home_points,"model_away":projection.away_points,
        "eligible_prior_games":sorted(eligible_ids)},sort_keys=True).encode()).hexdigest()
    forecast={"record_kind":"FROZEN_FIRST_SEEN_FORECAST","league":"NFL",
              "event_id":game_id,"season":season,"week":week,
              "home_team":home,"away_team":away,
              "model_version":"ASOF_NFL_CANONICAL_RIDGE_RESEARCH_V1",
              "captured_utc":at.isoformat(),"kickoff_utc":kickoff.isoformat(),
              "independent_home_points":projection.home_points,
              "independent_away_points":projection.away_points,
              "independent_projected_total":projection.total,
              "independent_projected_home_margin":projection.home_margin,
              "content_sha256":sha,
              "quarterback_starters_verified":False,
              "injury_weather_stadium_asof_verified":False,
              "pregame_source_vintage_asserted":False}
    required={
        "game_id","fixed_drive","fixed_drive_result","posteam",
        "pass_attempt","complete_pass","passing_yards",
        "rush_attempt","rushing_yards","play_type",
    }
    if not required.issubset(set(pbp.columns)):
        raise ValueError("PBP_MISSING_DRIVE_COLUMNS:"+str(sorted(required-set(pbp.columns))))
    # Using only the select columns reduces iteration/memory overhead.
    filtered=pbp.filter(pbp.get_column("game_id").is_in(list(eligible_ids))).select(sorted(required))
    bundles=defaultdict(list)
    for row in filtered.iter_rows(named=True):
        gid=str(row.get("game_id") or "")
        did=row.get("fixed_drive")
        if not gid or did is None:continue
        bundles[(gid,str(did))].append(row)
    library=[];rejects=Counter()
    for (gid,did),plays in bundles.items():
        snap,ko=eligible[gid]
        eligible_plays=[p for p in plays if p.get("play_type") in ("pass","run")
                        and p.get("posteam")]
        if not eligible_plays:
            rejects["NO_SCRIMMAGE_OFFENSE"]+=1
            continue
        team_counts=Counter(str(p["posteam"]) for p in eligible_plays)
        team,n=team_counts.most_common(1)[0]
        if n!=len(eligible_plays):
            rejects["MULTIPLE_POSTEAM_FIXED_DRIVE"]+=1
            continue
        results=Counter(str(p.get("fixed_drive_result") or "") for p in plays
                        if p.get("fixed_drive_result"))
        if not results:
            rejects["NO_FIXED_DRIVE_RESULT"]+=1
            continue
        result=results.most_common(1)[0][0]
        points=7 if result=="Touchdown" else 3 if result=="Field goal" else 0
        count_pass=lambda key:sum(float(p.get(key) or 0) for p in eligible_plays)
        pa=round(count_pass("pass_attempt"))
        co=round(count_pass("complete_pass"))
        py=round(count_pass("passing_yards"))
        ra=round(count_pass("rush_attempt"))
        ry=round(count_pass("rushing_yards"))
        if co>pa or co==0 and py!=0 or py<0 or pa+ra==0:
            rejects["INVALID_DRIVE_PASS_RUSH_ACCOUNTING"]+=1
            continue
        library.append({
            "game_id":gid,"drive_id":did,"team":team,
            "season":int(snap["season"]),"week":int(snap["week"]),
            "kickoff_utc":ko.isoformat(),
            "verified_at_utc":at.isoformat(),
            "completed_nflverse_schedule_game":True,
            "source_sha256":sha256((gid+"|"+did+"|"+team+"|"+str(pa)+str(co)+str(py)).encode()).hexdigest(),
            "points":points,"pass_attempts":pa,"completions":co,
            "passing_yards":py,"rush_attempts":ra,"rushing_yards":ry,
        })
    # Prior player game-level source rows are read at the same time, not
    # independently-certified active lineups or verified participation.
    previous=player_stats.filter(
        (player_stats.get_column("season")==season)&
        (player_stats.get_column("week")<week)&
        (player_stats.get_column("week")>=max(1,week-4))
    )
    required_stats={"team","player_id","position","season","week","targets","carries"}
    if not required_stats.issubset(set(previous.columns)):
        raise ValueError("PLAYER_STATS_MISSING_COLUMNS:"+str(sorted(required_stats-set(previous.columns))))
    role_totals=defaultdict(lambda:[0.,0.])
    for r in previous.select(sorted(required_stats)).iter_rows(named=True):
        t=r["team"]
        if t not in (home,away) or not r.get("player_id") or r.get("position") not in ("QB","RB","WR","TE","FB"):
            continue
        key=(t,str(r["player_id"]),str(r["position"]))
        role_totals[key][0]+=max(0,float(r.get("targets") or 0))
        role_totals[key][1]+=max(0,float(r.get("carries") or 0))
    roles=[]
    for team in (home,away):
        target_total=sum(v[0] for k,v in role_totals.items() if k[0]==team)
        rush_total=sum(v[1] for k,v in role_totals.items() if k[0]==team)
        if target_total<=0 or rush_total<=0:raise ValueError("NO_VERIFIED_HISTORICAL_ROLE_DENOMINATOR")
        for (t,p,pos),(tar,car) in role_totals.items():
            if t!=team or tar==0 and car==0:continue
            roles.append({"team":team,"player_id":p,"role":pos,
                          "target_share":tar/target_total,
                          "carry_share":car/rush_total,
                          "observed_at_utc":at.isoformat(),
                          "activity":"UNKNOWN"})
    if not roles:raise ValueError("PLAYER_ROLE_COVERAGE_NONE")
    provenance={"source":"nflreadpy_NFLVERSE_CURRENT_AT_CAPTURE","captured_utc":at.isoformat(),
                "eligible_schedule_games":len(eligible),"derived_drives":len(library),
                "derive_drive_rejects":dict(rejects),"role_records":len(roles),
                "independent_original_provider_publish_time":None,
                "player_activity_inactive_status":None,
                "no_historical_first_seen_archives_claimed":True}
    return forecast,library,roles,provenance
