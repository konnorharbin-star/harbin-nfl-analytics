"""No overstatement of synthetic prop calibration as BetMGM EV."""
import copy
import pytest
from scripts.paired_v1_v2_forward_prop_audit import compare,V1_VERSION,V2_VERSION


def row(version,p=.55,actual=1,market="receptions",event="2026_06_BUF_LV"):
    return {
        "version":version,"forecast_id":"synthetic123",
        "event_id":event,"player_id":"P01","market":market,
        "line":4.5,"actual_stat":5,"actual_over":actual,
        "p_over":p,"brier_over":(p-actual)**2,
        "first_forecast_utc":"2026-10-12T03:30:00Z",
        "kickoff_utc":"2026-10-18T20:25:00Z",
        "first_scored_observed_utc":"2026-10-19T04:00:00Z",
        "verified_betmgm_quote":False,
        "betting_roi":None,
    }


def test_future_no_graded_v2_is_not_a_bet_or_backfilled():
    out=compare([row(V1_VERSION)],[])
    assert out["status"]=="NO_PAIRED_SETTLED_CONTRACTS"
    assert out["matched_exact_contracts"]==0
    assert out["profitable_betting_edge"] is None
    assert out["betting_recommendations"]==[]


def test_improvement_is_only_a_diagnostic_until_independent_validation():
    o=compare([row(V1_VERSION,p=.35)],[row(V2_VERSION,p=.7)])
    assert o["matched_exact_contracts"]==1
    assert o["by_market"]["receptions"]["brier_delta_v2_minus_v1"]<0
    assert o["by_market"]["receptions"]["eligible_for_betting_approval"] is False
    assert not o["by_market"]["receptions"]["minimum_sample_thresholds_met"]


def test_mismatched_player_ids_or_line_do_not_pair():
    a=row(V1_VERSION)
    b=row(V2_VERSION)
    b["line"]=5.5
    b["actual_over"]=0
    b["brier_over"]=(b["p_over"]-0)**2
    assert compare([a],[b])["matched_exact_contracts"]==0


def test_same_market_different_official_result_aborts():
    a=row(V1_VERSION)
    b=row(V2_VERSION)
    b["actual_stat"]=6
    with pytest.raises(ValueError,match="different official outcomes"):
        compare([a],[b])


def test_pregame_model_created_postkickoff_aborts():
    a=row(V1_VERSION)
    a["first_forecast_utc"]="2026-10-18T20:26:00Z"
    with pytest.raises(ValueError,match="retroactive"):
        compare([a],[])


def test_bad_probability_stored_score_aborts():
    a=row(V1_VERSION)
    a["brier_over"]=.01
    with pytest.raises(ValueError,match="stored score"):
        compare([a],[])


def test_incomplete_dnp_outcome_not_an_under_bet():
    a=row(V1_VERSION)
    a["actual_stat"]=None
    with pytest.raises((TypeError,ValueError)):
        compare([a],[])


def test_duplicate_grade_does_not_create_independent_sample():
    a=row(V1_VERSION)
    with pytest.raises(ValueError,match="duplicate original contract"):
        compare([a,a],[])


def test_whole_number_pushed_prop_illegal_for_original_shadow():
    a=row(V1_VERSION)
    a["line"]=5
    with pytest.raises(ValueError,match="half-point"):
        compare([a],[])


def test_market_replay_keeps_four_separate_stat_types():
    markets=("receptions","receiving_yards","rushing_yards","passing_yards")
    va=[];vb=[]
    for market in markets:
        va.append(row(V1_VERSION,p=.4,market=market))
        vb.append(row(V2_VERSION,p=.6,market=market))
    result=compare(va,vb)
    assert len(result["by_market"])==4
    assert result["matched_exact_contracts"]==4
    assert result["matched_player_game_market_clusters"]==4
    assert result["matched_distinct_games"]==1


def test_mismatch_v1_v2_source_versions_refused():
    a=row(V1_VERSION)
    b=row("FAKE")
    with pytest.raises(ValueError,match="wrong model source version"):
        compare([a],[b])
