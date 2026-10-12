"""Regress original v2 versus frozen full-CDF source integrity."""
import pytest
from scripts.nfl_prop_original_consistency_audit import reconcile
from scripts.nfl_player_prop_full_distribution import (
    VERSION, CALIBRATION_VERSION, hash_object,
)
from nfl.prop_probability import over_probability, _scale


def setup():
    catalog={"calibration_version":CALIBRATION_VERSION,"training_year":2024,
             "vectors":{"rushing_yards|RB":[(i-50)/40 for i in range(100)]}}
    catalog["sha256"]=hash_object(catalog)
    full=[{"model":VERSION,
        "residual_catalog_sha256":catalog["sha256"],
        "game_id":"2026_05_BUF_LA","player_id":"ID","market":"rushing_yards",
        "residual_group_key":"rushing_yards|RB",
        "original_point_mean":70.,"original_scale":_scale("rushing_yards",70),
        "first_observed_at_utc":"2026-10-12T04:36:00Z",
        "kickoff_utc":"2026-10-13T00:15:00Z"}]
    row=[]
    for line in (49.5,64.5,79.5):
        row.append({"game_id":"2026_05_BUF_LA","player_id":"ID",
           "market":"rushing_yards",
           "synthetic_line":str(line),
           "p_over":round(over_probability(
                70.,line,"rushing_yards",catalog["vectors"]["rushing_yards|RB"]),6),
           "first_observed_at_utc":"2026-10-12T03:27:00Z",
           "kickoff_utc":"2026-10-13T00:15:00+00:00"})
    return row,full,catalog


def test_exact_same_originals_reconcile_at_csv_precision():
    v,f,c=setup()
    r=reconcile(v,f,c)
    assert r["overlap_exact_line_pairs"]==3
    assert r["threshold_probability_differences_gt_tol"]==0
    assert r["candidate_cross_version_equivalence_proven"]
    assert r["source_originals_modified"] is False
    assert r["bookmaker_EV_or_profitability_validated"] is False


def test_original_prob_change_reports_conflict_and_no_promotion():
    v,f,c=setup()
    v[0]["p_over"]=.01
    r=reconcile(v,f,c)
    assert r["threshold_probability_differences_gt_tol"]==1
    assert not r["candidate_cross_version_equivalence_proven"]
    assert r["largest_difference_examples"]


def test_source_cohort_gaps_reported():
    v,f,c=setup()
    f.clear()
    r=reconcile(v,f,c)
    assert r["v2_thresholds_without_full_distribution_group"]==3
    assert r["overlap_exact_line_pairs"]==0
    assert r["decision"]=="PASS"


def test_source_duplicates_are_rejected():
    v,f,c=setup()
    with pytest.raises(ValueError,match="duplicate full"):
        reconcile(v,f+f,c)
    with pytest.raises(ValueError,match="duplicate v2"):
        reconcile(v+v,f,c)


def test_catalog_mutation_blocks_claimed_reconciliation():
    v,f,c=setup()
    c["vectors"]["rushing_yards|RB"][0]=100.
    with pytest.raises(ValueError,match="hash"):
        reconcile(v,f,c)


def test_late_origination_is_not_pregame_forecast():
    v,f,c=setup()
    f[0]["first_observed_at_utc"]="2026-10-13T00:16:00Z"
    with pytest.raises(ValueError,match="late original"):
        reconcile(v,f,c)


def test_original_kickoff_match_required():
    v,f,c=setup()
    f[0]["kickoff_utc"]="2026-10-14T00:15:00Z"
    with pytest.raises(ValueError,match="different actual kickoff"):
        reconcile(v,f,c)


def test_no_bet_recommendation_from_equivalence():
    v,f,c=setup()
    x=reconcile(v,f,c)
    assert x["decision"]=="PASS"
    assert x["wagers"]==0
    assert not x["user_BetMGM_account_quote_verified"]
    assert not x["outcomes_used_for_reconciliation"]
