"""Future-only NFL prop readiness tests. No outcomes, bookmakers or wagers."""
import copy
from datetime import datetime, timezone
import json

import pytest

from scripts.nfl_prop_forward_maturity_readiness import readiness, run
from scripts.nfl_prop_primary_anchor_cohort import build_anchor_candidates
from scripts.nfl_prop_paired_cdf_forward_grader import sha_payload
from test_nfl_prop_primary_anchor_cohort import source_lines


def original_game():
    paired=source_lines()
    a,_=build_anchor_candidates(paired,now="2026-10-12T21:35:00Z")
    return a,paired


def two_games():
    a, p=original_game()
    second=copy.deepcopy(p)
    for r in second:
        r["game_id"]="2026_06_NEXT_TEST"
        r["kickoff_utc"]="2026-10-18T20:15:00Z"
        r["frozen_row_sha256"]=sha_payload({
            k:v for k,v in r.items() if k!="frozen_row_sha256"})
    anchors2,_=build_anchor_candidates(second,now="2026-10-12T21:35:00Z")
    return a+anchors2,p+second


def test_time_before_kickoff_is_never_scored():
    a,p=original_game()
    out=readiness(a,p,now="2026-10-12T23:00:00Z")
    assert out["unique_original_games"]==1
    assert out["original_primary_single_line_anchors"]==1
    assert out["original_paired_synthetic_thresholds"]==7
    assert out["markets"]["receptions"]["not_yet_kicked_off"]==1
    assert out["time_elapsed_anchor_count_not_same_as_source_graded_count"]==0
    assert out["next_time_boundary_utc"]=="2026-10-16T00:15:00+00:00"
    assert out["earliest_time_window_opened_utc"]=="2026-10-16T00:15:00+00:00"
    assert out["true_betmgm_ev"] is None and out["decision"]=="PASS"


def test_after_kickoff_before_72h_is_blocked():
    a,p=original_game()
    out=readiness(a,p,now="2026-10-13T02:30:00Z")
    assert out["markets"]["receptions"]["between_kickoff_and_72h"]==1
    assert out["markets"]["receptions"]["72h_elapsed_source_not_verified"]==0
    assert out["looked_up_postgame_scores"] is False
    assert out["looked_up_player_stats"] is False
    assert out["markets"]["receptions"]["official_stat_source_eligible_anchors"] is None


def test_at_exact_72h_cutoff_is_only_time_eligible_not_official_grade():
    a,p=original_game()
    out=readiness(a,p,now="2026-10-16T00:15:00Z")
    assert out["markets"]["receptions"]["72h_elapsed_source_not_verified"]==1
    assert out["time_elapsed_anchor_count_not_same_as_source_graded_count"]==1
    assert out["markets"]["receptions"]["graded_from_official_stat_rows"] is None
    assert out["account_accepted_betmgm_prices"]==0
    assert out["true_betmgm_ev"] is None
    assert out["next_time_boundary_utc"] is None


def test_multiple_games_have_distinct_maturity_windows_and_one_anchor_each():
    a,p=two_games()
    out=readiness(a,p,now="2026-10-16T00:15:00Z")
    assert out["unique_original_games"]==2
    assert out["original_primary_single_line_anchors"]==2
    assert out["original_paired_synthetic_thresholds"]==14
    assert out["markets"]["receptions"]["72h_elapsed_source_not_verified"]==1
    assert out["markets"]["receptions"]["not_yet_kicked_off"]==1
    assert out["next_time_boundary_utc"]=="2026-10-21T20:15:00+00:00"
    assert out["game_readiness"][0]["preregistered_anchors"]==1


def test_detect_tampered_original_line_or_anchor_sha():
    a,p=original_game()
    p[0]["synthetic_line"]=99.5
    with pytest.raises(ValueError,match="SHA|corrupt|original"):
        readiness(a,p,now="2026-10-13T02:30:00Z")
    a,p=original_game()
    a[0]["p_over_original_base_CDF"]=.01
    with pytest.raises(ValueError,match="anchor content hash"):
        readiness(a,p,now="2026-10-13T02:30:00Z")


def test_incomplete_original_grid_fails_closed():
    a,p=original_game()
    with pytest.raises(ValueError,match="all seven original lines"):
        readiness(a,p[:-1],now="2026-10-13T02:30:00Z")


def test_duplicate_anchor_or_inconsistent_game_kickoff_fails():
    a,p=original_game()
    with pytest.raises(ValueError,match="duplicate"):
        readiness(a+a,p,now="2026-10-13T02:30:00Z")


def test_non_utc_time_rejected_not_assumed():
    a,p=original_game()
    with pytest.raises(ValueError,match="UTC"):
        readiness(a,p,now="2026-10-13T00:00:00-05:00")


def test_read_only_report_does_not_change_original_frozen_inputs(tmp_path):
    a,p=two_games()
    fa=tmp_path/"anchors.jsonl"
    fp=tmp_path/"paired.jsonl"
    report=tmp_path/"calendar.json"
    fa.write_text("".join(json.dumps(row)+"\n" for row in a))
    fp.write_text("".join(json.dumps(row)+"\n" for row in p))
    before_a=fa.read_bytes();before_p=fp.read_bytes()
    out=run(now="2026-10-13T02:30:00Z",anchors_path=fa,
            paired_path=fp,report_path=report)
    assert out["decision"]=="PASS"
    assert report.exists() and report.stat().st_size>0
    assert fa.read_bytes()==before_a and fp.read_bytes()==before_p


def test_missing_source_is_not_fabricated_as_zero_grade(tmp_path):
    empty=tmp_path/"none.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError,match="no frozen first-seen"):
        run(now="2026-10-13T02:30:00Z",anchors_path=empty,
            paired_path=empty,report_path=tmp_path/"report.json")
