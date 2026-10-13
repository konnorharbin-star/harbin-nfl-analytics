"""Forward grading guardrail: synthetic data is never betting proof."""
from copy import deepcopy
from datetime import UTC,datetime,timedelta
import pytest
from nfl.coupled_forward_grading import grade_snapshot

CAP=datetime(2026,10,13,11,21,tzinfo=UTC)
KICK=datetime(2026,10,18,20,5,tzinfo=UTC)
ASOF=KICK+timedelta(days=4)

def orig():
    return {"spec":"NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1","capture_utc":CAP.isoformat(),
      "games":[{"game_id":"2026_06_BUF_LA",
        "status":"UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH",
        "frozen_football_score_input":{
          "event_id":"2026_06_BUF_LA","captured_utc":CAP.isoformat(),
          "kickoff_utc":KICK.isoformat(),"home_team":"LA","away_team":"BUF",
          "content_sha256":"a"*64,"independent_projected_total":45,
          "independent_projected_home_margin":2},
        "distribution":{"TOTAL":{"mean":48},"HOME_MARGIN":{"mean":3},
             "LA:00-0000001:receiving_yards":{"mean":78},
             "LA:00-0000002:receptions":{"mean":3.5}},
        "home_win":0.65,"away_win":0.34,"tie":0.01}]}

def finals():
    return [{"game_id":"2026_06_BUF_LA","gameday":"2026-10-18","gametime":"16:05",
      "game_type":"REG","home_team":"LA","away_team":"BUF",
      "home_score":28,"away_score":24}]

def players():
    return [{"game_id":"2026_06_BUF_LA","team":"LA",
             "player_id":"00-0000001","receiving_yards":90}]

def test_pairstored_grade_and_missing_player_never_zero():
    s=orig();f=finals();p=players()
    copies=deepcopy((s,f,p))
    report=grade_snapshot(s,f,p,at=ASOF)
    assert (s,f,p)==copies
    assert report["counts"]["graded"]==1
    assert report["counts"]["player_stat_rows_not_eligible"]==1
    row=report["games"][0]
    assert row["canonical_total_mae"]==7
    assert row["shadow_total_mae"]==4
    assert row["total_mae_delta_baseline_minus_shadow"]==3
    assert row["canonical_margin_mae"]==2
    assert row["shadow_margin_mae"]==1
    assert row["qualified_player_stat_rows"][0]["absolute_error"]==12
    assert len(row["qualified_player_stat_rows"])==1
    assert row["eligible_for_profitability_registry"] is False
    assert report["promotion_ready"] is False
    assert report["wager_authorized"] is False

def test_no_early_or_missing_final_grade():
    assert grade_snapshot(orig(),finals(),players(),at=KICK+timedelta(hours=71))["counts"]["graded"]==0
    assert grade_snapshot(orig(),[],players(),at=ASOF)["counts"]["final_unverified"]==1
    bad=finals();bad[0]["away_team"]="OTHER"
    assert grade_snapshot(orig(),bad,players(),at=ASOF)["counts"]["identity_error"]==1

def test_anti_backfill_cutoff_and_duplicate_game():
    s=orig()
    s["games"][0]["frozen_football_score_input"]["captured_utc"]=(KICK+timedelta(minutes=2)).isoformat()
    assert grade_snapshot(s,finals(),players(),at=ASOF)["counts"]["invalid_original"]==1
    s=orig();s["games"].append(deepcopy(s["games"][0]))
    with pytest.raises(ValueError,match="DUPLICATE"):
        grade_snapshot(s,finals(),players(),at=ASOF)
    s=orig();s["games"][0]["home_win"]=3
    assert grade_snapshot(s,finals(),players(),at=ASOF)["counts"]["invalid_original"]==1

def test_no_economic_claims_from_synthetic_gains():
    result=grade_snapshot(orig(),finals(),players(),at=ASOF)
    assert result["metrics"]["total_mae_delta_baseline_minus_shadow"]>0
    assert result["promotion_ready"] is False
    assert result["official_NFL_gamebook_crosscheck_done"] is False
    assert result["sportsbook_price_and_roi_graded"] is False

def test_qb_passing_touchdowns_and_interceptions_use_exact_stat_columns():
    snap=orig()
    stat=snap["games"][0]["distribution"]
    stat["LA:00-0000001:passing_touchdowns"]={"mean":1.3}
    stat["LA:00-0000001:passing_interceptions"]={"mean":0.7}
    stat["LA:00-0000001:passing_completions"]={"mean":22.0}
    stat["LA:00-0000001:passing_attempts"]={"mean":32.0}
    stat["LA:00-0000002:receiving_touchdowns"]={"mean":0.2}
    source=players()
    source[0].update({"passing_tds":2,"passing_interceptions":1,
                      "completions":20,"attempts":31})
    out=grade_snapshot(snap,finals(),source,at=ASOF)
    scored={x["stat"]:x for x in out["games"][0]["qualified_player_stat_rows"]}
    assert scored["passing_touchdowns"]["actual"]==2
    assert scored["passing_touchdowns"]["source_field"]=="passing_tds"
    assert scored["passing_interceptions"]["actual"]==1
    assert scored["passing_completions"]["source_field"]=="completions"
    assert scored["passing_attempts"]["source_field"]=="attempts"
    assert "receiving_touchdowns" not in scored
    assert out["counts"]["player_stat_rows_not_eligible"]==2
    assert out["promotion_ready"] is False

def test_real_nflverse_weekly_player_rows_without_game_id_are_scored():
    s=orig()
    score=finals()
    score[0].update({"season":2026,"week":6})
    rows=[
        {"player_id":"00-0000001","recent_team":"LA","opponent_team":"BUF",
         "season":2026,"week":6,"season_type":"REG",
         "receiving_yards":90,"passing_tds":2,"passing_interceptions":1},
        {"player_id":"00-0000002","recent_team":"LA","opponent_team":"BUF",
         "season":2026,"week":6,"season_type":"REG","receptions":4},
    ]
    s["games"][0]["distribution"]["LA:00-0000001:passing_touchdowns"]={"mean":1.3}
    report=grade_snapshot(s,score,rows,at=ASOF)
    stats={x["stat"]:x for x in report["games"][0]["qualified_player_stat_rows"]}
    assert len(stats)==3
    assert stats["receiving_yards"]["actual"]==90
    assert stats["passing_touchdowns"]["actual"]==2
    assert stats["receptions"]["actual"]==4
    assert report["player_join_rule"]=="EXACT_REG_SEASON_WEEK_TEAM_SCHEDULE_ID"
    assert report["counts"]["player_stat_rows_not_eligible"]==0
    assert report["promotion_ready"] is False

def test_weekly_results_wrong_opponent_week_or_postseason_cannot_leak():
    s=orig()
    actual=finals()
    actual[0].update({"season":2026,"week":6})
    cases=(
      {"season":2026,"week":5,"recent_team":"LA","opponent_team":"BUF"},
      {"season":2026,"week":6,"recent_team":"LA","opponent_team":"OTHER"},
      {"season":2026,"week":6,"recent_team":"LA","opponent_team":"BUF",
       "season_type":"POST"},
      {"season":2027,"week":6,"recent_team":"LA","opponent_team":"BUF"},
      {"season":2026,"week":6,"recent_team":"OTHER","opponent_team":"BUF"},
    )
    for change in cases:
        row={"player_id":"00-0000001","receiving_yards":900}
        row.update(change)
        result=grade_snapshot(s,actual,[row],at=ASOF)
        assert len(result["games"][0]["qualified_player_stat_rows"])==0,change
        assert result["counts"]["player_stat_rows_not_eligible"]==2

def test_multiple_player_rows_same_game_pid_ambiguous_not_double_counted():
    s=orig()
    official=finals()
    official[0].update({"season":2026,"week":6})
    p={"player_id":"00-0000001","recent_team":"LA",
       "season":2026,"week":6,"opponent_team":"BUF","receiving_yards":90}
    result=grade_snapshot(s,official,[p,dict(p)],at=ASOF)
    assert result["counts"]["graded"]==1
    assert result["counts"]["player_stat_rows_not_eligible"]==2
    assert result["games"][0]["qualified_player_stat_rows"]==[]

def test_ambiguous_same_week_team_cannot_join_any_player_row():
    s=orig()
    f=finals()
    f[0].update({"season":2026,"week":6})
    f.append(dict(f[0],game_id="OTHER_ID",away_team="NYG"))
    p={"player_id":"00-0000001","recent_team":"LA",
       "season":2026,"week":6,"opponent_team":"BUF","receiving_yards":90}
    r=grade_snapshot(s,f,[p],at=ASOF)
    assert r["counts"]["graded"]==1
    assert r["weekly_rows_with_uncertain_identity"]==1
    assert r["games"][0]["qualified_player_stat_rows"]==[]
