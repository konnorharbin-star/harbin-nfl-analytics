"""Prospective anchor scoring is pregame, source-complete, non-bookmaker."""
import copy
import pytest
from scripts.nfl_prop_primary_anchor_score import anchor_results
from scripts.nfl_prop_primary_anchor_cohort import (
    build_anchor_candidates,VERSION as ANCHOR_VERSION
)
from scripts.nfl_prop_paired_cdf_forward_grader import sha_payload
from tests.test_nfl_prop_primary_anchor_cohort import source_lines


def first_seen():
    originals=source_lines()
    a,_=build_anchor_candidates(originals,now="2026-10-12T21:35:00Z")
    return originals,a


def test_one_line_per_player_market_and_zero_mature_scores():
    original,anchors=first_seen()
    result=anchor_results(anchors,original,[])
    assert result["original_single_anchor_groups"]==1
    assert result["scoreboard"]["receptions"]["preregistered_original_player_game_market_anchors"]==1
    assert result["scoreboard"]["receptions"]["mature_source_valid_graded_anchors"]==0
    assert result["scoreboard"]["receptions"]["not_yet_valid_graded_anchor_groups"]==1
    assert result["scoreboard"]["receptions"]["original_conditional_base_Brier"] is None
    assert result["true_EV"] is None
    assert result["bet_approved"] is False


def grade(original,anchor,truth=1,revision=1):
    selected=next(x for x in original
        if x["synthetic_line"]==anchor["anchor_synthetic_half_point_line"])
    return {
        "game_id":selected["game_id"],
        "player_id":selected["player_id"],
        "market":selected["market"],
        "synthetic_line":selected["synthetic_line"],
        "original_row_sha256":selected["frozen_row_sha256"],
        "p_original":selected["p_over_original_conditional_on_official_stat_row"],
        "p_snap":selected["p_over_snap_conditional_on_official_stat_row"],
        "target_over_given_official_stat_row":truth,
        "revision":revision,
    }


def test_anchor_score_once_not_all_correlated_thresholds():
    original,anchors=first_seen()
    x=grade(original,anchors[0])
    result=anchor_results(anchors,original,[x])
    metric=result["scoreboard"]["receptions"]
    assert metric["mature_source_valid_graded_anchors"]==1
    assert metric["independent_games_with_a_valid_grade"]==1
    assert metric["game_cluster_ci95"] is None
    assert metric["four_market_familywise_ci98_75"] is None
    assert metric["original_conditional_base_Brier"]==pytest.approx((.56-1)**2)
    assert metric["snap_conditional_challenger_Brier"]==pytest.approx((.52-1)**2)


def test_outcome_provider_correction_uses_latest_revision():
    original,anchors=first_seen()
    a=grade(original,anchors[0],truth=1,revision=1)
    b=grade(original,anchors[0],truth=0,revision=2)
    out=anchor_results(anchors,original,[a,b])
    assert out["scoreboard"]["receptions"]["original_conditional_base_Brier"]==pytest.approx(.56**2)


def test_score_never_selects_anchor_after_target():
    original,anchors=first_seen()
    bad=copy.deepcopy(anchors)
    bad[0]["anchor_first_observed_at_utc"]="2026-10-13T00:16:00Z"
    bad[0]["anchor_record_sha256"]=sha_payload({
        k:v for k,v in bad[0].items() if k!="anchor_record_sha256"})
    with pytest.raises(ValueError,match="after target kickoff"):
        anchor_results(bad,original,[])


def test_score_cannot_swap_constituent_pregame_probability():
    original,anchors=first_seen()
    tampered=copy.deepcopy(anchors)
    tampered[0]["p_over_snap_challenger_CDF"]=.999
    tampered[0]["anchor_record_sha256"]=sha_payload({
        k:v for k,v in tampered[0].items() if k!="anchor_record_sha256"})
    with pytest.raises(ValueError,match="anchor probabilities altered"):
        anchor_results(tampered,original,[])


def test_score_blocks_outcome_wrong_initial_original_sha():
    original,anchors=first_seen()
    bad=grade(original,anchors[0])
    bad["original_row_sha256"]="different"
    with pytest.raises(ValueError,match="replaced originally frozen"):
        anchor_results(anchors,original,[bad])


def test_anchor_frozen_source_has_all_seven_original_lines():
    original,anchors=first_seen()
    with pytest.raises(ValueError,match="all seven original lines"):
        anchor_results(anchors,original[:-1],[])


def test_anchor_selection_not_modified_by_outcome():
    original,anchors=first_seen()
    score1=anchor_results(anchors,original,[grade(original,anchors[0],truth=0)])
    score2=anchor_results(anchors,original,[grade(original,anchors[0],truth=1)])
    assert score1["anchor_selection_rule"]==score2["anchor_selection_rule"]
    assert score1["original_single_anchor_groups"]==score2["original_single_anchor_groups"]==1
