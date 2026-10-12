"""V2 first-seen immutable forecast grader: no retrospective bets."""
import pytest

from scripts.grade_participation_prop_v2 import (
    SHADOW_VERSION,GRADE_VERSION,adapt_original_source,grade_new,
)

def original():
    return {
        "version":SHADOW_VERSION,"game_id":"2026_06_BUF_LV",
        "season":"2026","week":"6",
        "first_observed_at_utc":"2026-10-12T03:27:52Z",
        "kickoff_utc":"2026-10-18T20:25:00Z",
        "team":"BUF","player_id":"P001","player_name":"Fixture",
        "position":"WR","market":"receptions","synthetic_line":"4.5",
        "p_over":"0.58","p_under":"0.42",
        "recommended_bet":"False","american_odds":"","bookmaker":"",
    }

def game(score=21):
    return {
        "game_id":"2026_06_BUF_LV","game_type":"REG",
        "season":2026,"week":6,"gameday":"2026-10-18","gametime":"16:25",
        "home_team":"LV","away_team":"BUF",
        "home_score":score,"away_score":24,
    }

def player():
    return {"player_id":"P001","team":"BUF","game_id":"2026_06_BUF_LV",
            "season":2026,"week":6,"season_type":"REG","receptions":6}

def test_v2_original_remains_unpriced_and_separate_from_v1():
    mapped=adapt_original_source([original()])
    assert mapped[0]["probability_over"]=="0.58"
    assert mapped[0]["first_observed_at_utc"]=="2026-10-12T03:27:52Z"
    assert mapped[0]["recommended_bet"]=="False"

def test_v2_cannot_grade_future_unplayed():
    entries,report=grade_new([original()],[game(score=None)],[],
                              observed_utc="2026-10-13T04:00:00Z",existing=[])
    assert entries==[] and report["decision"]=="PASS"
    assert report["approved_betting_model"] is False

def test_v2_original_after_outcome_scores_once():
    entries,report=grade_new([original()],[game()],[player()],
                              observed_utc="2026-10-19T05:00:00Z",existing=[])
    assert len(entries)==1
    assert entries[0]["version"]==GRADE_VERSION
    assert entries[0]["original_frozen_model_version"]==SHADOW_VERSION
    assert entries[0]["actual_over"]==1
    assert report["new_independent_player_game_clusters"]==1
    assert entries[0]["betting_roi"] is None
    assert report["bookmaker_account_price_verified"] is False

def test_v2_cannot_regrade_original_settlement():
    entries,_=grade_new([original()],[game()],[player()],
                         observed_utc="2026-10-19T05:00:00Z",existing=[])
    second,report=grade_new([original()],[game()],[player()],
                            observed_utc="2026-10-20T05:00:00Z",existing=entries)
    assert second==[]
    assert report["cumulative_graded_threshold_rows"]==1
    assert report["status"]=="SCORED_RESEARCH_CUMULATIVE"

def test_unverified_no_stats_player_is_not_a_losing_under():
    entries,report=grade_new([original()],[game()],[],
                              observed_utc="2026-10-19T05:00:00Z",existing=[])
    assert not entries
    assert report["ungraded_reason_counts"]["MISSING_PLAYER_STATS_OR_DNP"]==1

def test_wrong_version_cannot_overwrite_v2_study():
    bad=original();bad["version"]="prop_empirical_error_cdf_v1"
    with pytest.raises(ValueError,match="non-v2"):
        adapt_original_source([bad])

def test_fabricated_sportsbook_price_in_shadow_refused():
    bad=original();bad["american_odds"]="-110"
    with pytest.raises(ValueError,match="research-only"):
        adapt_original_source([bad])

def test_hard_block_existing_wrong_model_version():
    with pytest.raises(ValueError,match="another scoring model"):
        grade_new([original()],[game()],[player()],
                   observed_utc="2026-10-19T05:00:00Z",
                   existing=[{"version":"v1","forecast_id":"x"}])

def test_duplicate_original_player_threshold_not_accepted():
    with pytest.raises(ValueError,match="duplicate"):
        adapt_original_source([original(),original()])
