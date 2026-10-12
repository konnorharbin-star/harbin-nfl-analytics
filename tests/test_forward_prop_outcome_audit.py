"""Synthetic regression: first-seen NFL prop calibration, no fabricated bets."""
import copy
from datetime import datetime,timezone
import json
import pytest
from scripts.forward_prop_outcome_audit import grade,append_results

WHEN="2026-10-13T04:00:00Z"

def schedule(**updates):
    x={"game_id":"2026_05_BUF_LA","season":2026,"week":5,
       "game_type":"REG","gameday":"2026-10-12","gametime":"20:15",
       "home_team":"LA","away_team":"BUF","home_score":24,"away_score":21}
    x.update(updates)
    return x

def forecast(line="4.5",prob=".60",**updates):
    x={"first_observed_at_utc":"2026-10-12T02:00:00Z",
       "game_id":"2026_05_BUF_LA","season":"2026","week":"5",
       "kickoff_utc":"2026-10-13T00:15:00Z",
       "team":"LA","player_id":"P001","player_name":"SYNTHETIC",
       "market":"receptions","synthetic_line":line,
       "probability_over":prob,"probability_under":str(1-float(prob))}
    x.update(updates)
    return x

def stat(**updates):
    x={"season":2026,"week":5,"recent_team":"LA","player_id":"P001",
       "receptions":7,"receiving_yards":82,"rushing_yards":0,
       "passing_yards":0}
    x.update(updates)
    return x

def run(fs=None,ss=None,gs=None,now=WHEN,previous=None):
    return grade(fs if fs is not None else [forecast()],
                 gs if gs is not None else [schedule()],
                 ss if ss is not None else [stat()],
                 captured_utc=now,previous=previous)

def test_true_postfinal_grade_and_brier_but_never_roi():
    scored,meta=run()
    assert meta["status"]=="SCORED_RESEARCH"
    assert meta["graded_new_threshold_rows"]==1
    assert meta["new_independent_player_game_clusters"]==1
    assert scored[0]["actual_over"]==1
    assert scored[0]["actual_stat"]==7
    assert scored[0]["brier_over"]==.16
    assert scored[0]["betting_roi"] is None
    assert not scored[0]["actual_bet_placed"]
    assert not meta["independently_validated_betmgm_positive_ev"]

def test_thresholds_cluster_per_player_game():
    s,m=run(fs=[forecast("4.5"),forecast("6.5",prob=".4")])
    assert len(s)==2
    assert m["new_independent_player_game_clusters"]==1
    assert m["score_metrics_by_market"]["receptions"]["graded_correlated_threshold_rows"]==2

def test_before_final_time_never_grades():
    s,m=run(now="2026-10-13T02:00:00Z")
    assert not s and m["ungraded_reason_counts"]["INSUFFICIENT_TIME_AFTER_KICKOFF"]==1

def test_unsettled_game_with_score_null_never_grades():
    s,m=run(gs=[schedule(home_score=None,away_score=None)])
    assert not s and m["ungraded_reason_counts"]["GAME_NOT_FINAL"]==1

def test_dnp_or_missing_player_row_not_silently_zero():
    s,m=run(ss=[])
    assert not s and m["ungraded_reason_counts"]["MISSING_PLAYER_STATS_OR_DNP"]==1

def test_player_stat_zero_is_graduated_when_explicit():
    s,m=run(ss=[stat(receptions=0)])
    assert s[0]["actual_over"]==0 and s[0]["brier_over"]==.36

def test_wrong_team_prediction_rejected():
    with pytest.raises(ValueError,match="does not play"):
        run(fs=[forecast(team="KC")])

def test_after_kickoff_forecast_rejected():
    with pytest.raises(ValueError,match="not actually frozen"):
        run(fs=[forecast(first_observed_at_utc="2026-10-13T01:00:00Z")])

def test_integer_line_cannot_be_graded_as_halfpoint():
    with pytest.raises(ValueError,match="integer threshold"):
        run(fs=[forecast(line="5")])

def test_duplicate_frozen_thresholds_rejected():
    x=forecast()
    with pytest.raises(ValueError,match="duplicate original"):
        run(fs=[x,x])

def test_previous_result_is_not_overwritten(tmp_path):
    a,_=run()
    p=tmp_path/"scored.jsonl"
    assert append_results(p,a)==1
    with pytest.raises(ValueError,match="overwrite"):
        append_results(p,a)
    after,meta=run(previous=a)
    assert not after and meta["ungraded_reason_counts"]["ALREADY_GRADED"]==1

def test_invalid_prediction_fails_closed():
    with pytest.raises(ValueError,match="bad frozen probability"):
        run(fs=[forecast(prob=".95",probability_under=".20")])

def test_noninteger_official_stats_blocked():
    with pytest.raises(ValueError,match="non-integer"):
        run(ss=[stat(receptions=7.1)])

def test_kickoff_change_not_silently_backfilled():
    s,m=run(gs=[schedule(gametime="20:25")])
    assert not s and m["ungraded_reason_counts"]["KICKOFF_CHANGED_OR_UNVERIFIED"]==1

def test_wrong_official_week_does_not_link():
    with pytest.raises(ValueError,match="season week"):
        run(gs=[schedule(week=6)])

def test_all_missing_results_remain_pass():
    s,m=run(gs=[])
    assert s==[]
    assert m["decision"]=="PASS"
    assert m["status"]=="NO_FINAL_RESULT_ELIGIBLE"
