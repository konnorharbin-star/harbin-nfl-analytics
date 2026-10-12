"""Non-hindsight ex-ante one-line NFL synthetic prop CDF anchor guards."""
import copy
import json
import pytest

from nfl.prop_probability import PROP_THRESHOLD_GRID
from scripts.nfl_prop_paired_cdf_forward_grader import VERSION as PAIRED,sha_payload
from scripts.nfl_prop_primary_anchor_cohort import (
    VERSION,build_anchor_candidates,append_first_seen
)

def source_lines(*,base_p=None,snap_p=None):
    lines=PROP_THRESHOLD_GRID["receptions"]
    if base_p is None:
        base_p=(.89,.70,.56,.44,.25,.10,.02)
    if snap_p is None:
        snap_p=(.90,.73,.52,.41,.23,.11,.02)
    out=[]
    for line,p,q in zip(lines,base_p,snap_p):
        r={
            "model":PAIRED,"game_id":"2026_05_BUF_LA",
            "player_id":"TEST_GSIS","market":"receptions",
            "synthetic_line":line,"position":"WR","team":"BUF",
            "first_observed_at_utc":"2026-10-12T21:20:25Z",
            "source_full_first_seen_utc":"2026-10-12T04:36:20Z",
            "source_snap_first_seen_utc":"2026-10-12T12:05:38Z",
            "source_prior_snap_first_seen_utc":"2026-10-12T11:32:18Z",
            "kickoff_utc":"2026-10-13T00:15:00Z",
            "p_over_original_conditional_on_official_stat_row":p,
            "p_over_snap_conditional_on_official_stat_row":q,
            "original_2024_catalog_sha256":"original","snap_2024_catalog_sha256":"snap",
            "original_full_residual_group":"receptions|WR",
            "snap_residual_group":"receptions|WR|HIGH_35_PLUS",
            "snap_bucket":"HIGH_35_PLUS","source_snaps_mean":45.,
            "line_is_actual_BetMGM_offer":False,
            "p_prop_void":None,
            "wager":False,
        }
        r["frozen_row_sha256"]=sha_payload(r)
        out.append(r)
    return out


def test_picks_near_half_from_original_never_candidate():
    rows=source_lines()
    found,meta=build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")
    assert meta["eligible_anchored_player_game_market_groups"]==1
    assert meta["original_paired_line_count"]==7
    assert found[0]["anchor_synthetic_half_point_line"]==3.5
    assert found[0]["p_over_original_base_CDF"]==.56
    assert not found[0]["wager"]
    assert found[0]["p_bet_void"] is None
    assert found[0]["bet_true_EV"] is None
    # Change candidate probability at all lines: selected BASE anchor is invariant.
    snap_changed=source_lines(snap_p=(.99,.98,.97,.90,.89,.79,.15))
    updated,_=build_anchor_candidates(snap_changed,now="2026-10-12T21:35:00Z")
    assert updated[0]["anchor_synthetic_half_point_line"]==3.5


def test_tie_chooses_smaller_synthetic_line():
    rows=source_lines(base_p=(.95,.75,.55,.45,.30,.10,.02))
    anchors,_=build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")
    assert anchors[0]["anchor_synthetic_half_point_line"]==3.5


def test_incomplete_or_duplicate_seven_grid_fails_closed():
    rows=source_lines()
    with pytest.raises(ValueError,match="exact full registered seven-line"):
        build_anchor_candidates(rows[:-1],now="2026-10-12T21:35:00Z")
    with pytest.raises(ValueError,match="exact full registered seven-line"):
        build_anchor_candidates(rows+rows[:1],now="2026-10-12T21:35:00Z")


def test_nonmonotone_probabilities_fail_before_registration():
    for field in ("p_over_original_conditional_on_official_stat_row",
                  "p_over_snap_conditional_on_official_stat_row"):
        rows=source_lines()
        rows[2][field]=.98
        rows[2]["frozen_row_sha256"]=sha_payload({
            k:v for k,v in rows[2].items() if k!="frozen_row_sha256"})
        with pytest.raises(ValueError,match="NONMONOTONE"):
            build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")


def test_original_sha_tamper_or_mismatched_sources_blocked():
    rows=source_lines()
    rows[0]["source_snaps_mean"]=100.
    with pytest.raises(ValueError,match="SHA"):
        build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")
    rows=source_lines()
    rows[0]["source_snaps_mean"]=100.
    rows[0]["frozen_row_sha256"]=sha_payload({
        k:v for k,v in rows[0].items() if k!="frozen_row_sha256"})
    with pytest.raises(ValueError,match="inconsistent"):
        build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")


def test_no_anchor_first_seen_after_kickoff():
    rows=source_lines()
    got,meta=build_anchor_candidates(rows,now="2026-10-13T00:16:00Z")
    assert got==[]
    assert meta["excluded_game_market_group_reasons"]["NOT_PREKICKOFF_AT_ANCHOR_REGISTRATION"]==1


def test_anchor_source_journal_first_seen_immutable(tmp_path):
    rows=source_lines()
    a,_=build_anchor_candidates(rows,now="2026-10-12T21:35:00Z")
    path=tmp_path/"anchors.jsonl"
    assert append_first_seen(a,path)["new_origination"]==1
    original=path.read_text()
    alter=copy.deepcopy(a)
    alter[0]["p_over_snap_challenger_CDF"]=.02
    alter[0]["anchor_record_sha256"]=sha_payload({
        k:v for k,v in alter[0].items() if k!="anchor_record_sha256"})
    assert append_first_seen(alter,path)["new_origination"]==0
    assert path.read_text()==original
    corrupted=json.loads(original.strip())
    corrupted["p_over_original_base_CDF"]=.999
    path.write_text(json.dumps(corrupted)+"\n")
    with pytest.raises(ValueError,match="historical anchor original"):
        append_first_seen(a,path)


def test_grouped_markets_are_reported_without_independence_claim():
    rows=source_lines()
    other=copy.deepcopy(rows)
    for r in other:
        r["game_id"]="2026_05_OTHER"
        r["frozen_row_sha256"]=sha_payload({
            k:v for k,v in r.items() if k!="frozen_row_sha256"})
    anchors,summary=build_anchor_candidates(rows+other,now="2026-10-12T21:35:00Z")
    assert len(anchors)==2
    assert summary["by_market"]["receptions"]["group_count"]==2
    assert summary["actual_BetMGM_bookmaker_quotes"]==0
    assert summary["true_EV"] is None
