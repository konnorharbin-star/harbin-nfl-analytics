"""2026 NFL prop conditional forward grader: provenance, leakage, no wagering."""
import copy
import json
import pytest
from nfl.prop_probability import PROP_THRESHOLD_GRID,_scale
from scripts.nfl_player_prop_full_distribution import (
    VERSION as FULL_VERSION,hash_object,
)
from scripts.nfl_prop_snap_conditional_cdf import VERSION as SNAP_VERSION
from scripts.nfl_prop_paired_cdf_forward_grader import (
    VERSION,freeze_pairs,eligible_grades,append_grades,grade_metrics,
    read_jsonl,validate_catalogs,line_key,sha_payload,
)


def catalogs():
    vals=[(i-100)/40 for i in range(201)]
    b={"training_year":2024,
       "vectors":{"receptions|WR":vals,"receiving_yards|WR":vals}}
    b["sha256"]=hash_object(b)
    s={"development_year":2024,
       "base_full_distribution_catalog_sha256":b["sha256"],
       "fitted_normalized_residual_vectors":{
         "receptions|WR|HIGH_35_PLUS":[v+.08 for v in vals],
         "receiving_yards|WR|HIGH_35_PLUS":[v+.08 for v in vals],
       }}
    s["sha256"]=hash_object(s)
    return b,s


def raw_pair():
    b,s=catalogs()
    full={
        "model":FULL_VERSION,"game_id":"2026_05_BUF_LA","season":2026,
        "week":5,"kickoff_utc":"2026-10-13T00:15:00+00:00",
        "first_observed_at_utc":"2026-10-12T04:36:20+00:00",
        "team":"BUF","player_id":"GSIS-ONE","player_name":"Test WR",
        "position":"WR","market":"receptions",
        "original_point_mean":3.5,"original_scale":_scale("receptions",3.5),
        "residual_catalog_sha256":b["sha256"],
        "training_residual_count":len(b["vectors"]["receptions|WR"]),
        "residual_group_key":"receptions|WR",
    }
    side={
        "model":SNAP_VERSION,"game_id":full["game_id"],
        "player_id":full["player_id"],"market":full["market"],
        "position":full["position"],"kickoff_utc":full["kickoff_utc"],
        "first_observed_at_utc":"2026-10-12T12:05:38+00:00",
        "original_snap_context_first_seen_utc":"2026-10-12T11:32:18+00:00",
        "original_forecast_mean":full["original_point_mean"],
        "original_forecast_scale":full["original_scale"],
        "base_2024_CDF_sha256":b["sha256"],
        "snap_2024_CDF_sha256":s["sha256"],
        "frozen_bucket":"HIGH_35_PLUS","prior_snap_mean_observed":44.,
        "original_CDF_group":"receptions|WR",
        "snap_CDF_group":"receptions|WR|HIGH_35_PLUS",
        "fallback_to_original_full_CDF":False,
    }
    return [full],[side],b,s


def frozen_rows(tmp_path):
    f,s,b,c=raw_pair()
    path=tmp_path/"original.jsonl"
    result=freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=path)
    assert result["first_seen_lines_added"]==7
    return read_jsonl(path)


def schedule():
    return [{
        "game_id":"2026_05_BUF_LA","season":2026,"week":5,
        "game_type":"REG","gameday":"2026-10-12","gametime":"20:15",
        "home_team":"LA","away_team":"BUF",
        "home_score":24,"away_score":19,
    }]


def stats(observed=True):
    players=[]
    for team in ("BUF","LA"):
        for i in range(7):
            players.append({"game_id":"2026_05_BUF_LA",
                "team":team,"player_id":f"{team}{i}","receptions":i})
    if observed:
        players.append({"game_id":"2026_05_BUF_LA","team":"BUF",
                        "player_id":"GSIS-ONE","receptions":4})
    return players


def test_original_matched_freeze_seven_thresholds_before_kickoff(tmp_path):
    rows=frozen_rows(tmp_path)
    assert {r["synthetic_line"] for r in rows}==set(PROP_THRESHOLD_GRID["receptions"])
    assert len({line_key(r) for r in rows})==7
    assert all(r["wager"] is False and r["decision"]=="PASS" for r in rows)
    assert all(r["line_is_actual_BetMGM_offer"] is False for r in rows)
    assert all(r["p_prop_void"] is None and r["profit_ev"] is None for r in rows)
    assert all(r["frozen_row_sha256"]==sha_payload({
          k:v for k,v in r.items() if k!="frozen_row_sha256"}) for r in rows)


def test_repeat_freeze_never_overwrites_first_seen(tmp_path):
    f,s,b,c=raw_pair();p=tmp_path/"journal.jsonl"
    old=freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=p)
    assert old["first_seen_lines_added"]==7
    first=p.read_text()
    second=freeze_pairs(f,s,b,c,now="2026-10-12T22:10:00Z",path=p)
    assert second["first_seen_lines_added"]==0
    assert second["existing_line_originals"]==7 and p.read_text()==first


def test_original_2024_catalog_integrity_and_source_identity(tmp_path):
    f,s,b,c=raw_pair()
    b["vectors"]["receptions|WR"][0]=1000
    with pytest.raises(ValueError,match="calibration SHA"):
        freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=tmp_path/"x")
    f,s,b,c=raw_pair()
    s[0]["original_forecast_mean"]=50
    with pytest.raises(ValueError,match="identical base contracts"):
        freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=tmp_path/"x")


def test_source_after_model_first_seen_required(tmp_path):
    f,s,b,c=raw_pair()
    s[0]["original_snap_context_first_seen_utc"]="2026-10-12T12:06:00Z"
    r=freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=tmp_path/"x")
    assert r["first_seen_lines_added"]==0
    assert r["exclude_reasons"]["ORIGINAL_SOURCE_AFTER_FREEZE_OR_KICKOFF"]==1


def test_model_cannot_first_freeze_after_game(tmp_path):
    f,s,b,c=raw_pair()
    r=freeze_pairs(f,s,b,c,now="2026-10-13T00:16:00Z",path=tmp_path/"x")
    assert r["first_seen_lines_added"]==0


def test_snap_bucket_is_not_changeable_after_first_seen(tmp_path):
    f,s,b,c=raw_pair()
    s[0]["prior_snap_mean_observed"]=22
    with pytest.raises(ValueError,match="bucket differs"):
        freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=tmp_path/"x")


def test_pre_kickoff_grade_block_and_unfinished_game(tmp_path):
    rows=frozen_rows(tmp_path)
    future,reason=eligible_grades(rows,schedule(),stats(),
        now="2026-10-12T21:15:00Z")
    assert future==[] and reason["OFFICIAL_STATS_NOT_72H_MATURE"]==7
    incomplete=[dict(schedule()[0],home_score=None)]
    o,r=eligible_grades(rows,incomplete,stats(),now="2026-10-17T01:00:00Z")
    assert o==[] and r["GAME_NOT_COMPLETED"]==7


def test_official_stat_row_missing_is_not_seven_under_wins(tmp_path):
    rows=frozen_rows(tmp_path)
    result,blocks=eligible_grades(rows,schedule(),stats(False),
        now="2026-10-17T01:00:00Z")
    assert result==[] and blocks["PLAYER_STAT_ROW_UNOBSERVED_DO_NOT_ASSUME_UNDER_OR_VOID"]==7


def test_completed_confirmed_player_stat_matches_exact_synthetic_lines(tmp_path):
    rows=frozen_rows(tmp_path)
    result,blocks=eligible_grades(rows,schedule(),stats(True),
        now="2026-10-17T01:00:00Z")
    assert not blocks and len(result)==7
    assert next(r for r in result if r["synthetic_line"]==3.5)["target_over_given_official_stat_row"]==1
    assert next(r for r in result if r["synthetic_line"]==4.5)["target_over_given_official_stat_row"]==0
    assert all(r["official_stat_value"]==4 for r in result)
    assert all(not r["source_record_is_BetMGM_settlement"] for r in result)
    assert all(r["betting_decision"]=="PASS" and r["actual_money_EV"] is None for r in result)


def test_both_teams_stat_source_gate(tmp_path):
    rows=frozen_rows(tmp_path)
    result,blocks=eligible_grades(rows,schedule(),stats(True)[:5],
        now="2026-10-17T01:00:00Z")
    assert not result
    assert blocks["OFFICIAL_TEAM_SOURCE_COVERAGE_INSUFFICIENT"]==7


def test_kickoff_or_frozen_hash_tamper_requires_audit(tmp_path):
    rows=frozen_rows(tmp_path)
    bad=[dict(r,synthetic_line=1.5 if r["synthetic_line"]!=1.5 else 2.5) for r in rows]
    with pytest.raises(ValueError,match="duplicate|tampered"):
        eligible_grades(bad,schedule(),stats(),now="2026-10-17T01:00:00Z")
    revised=[dict(schedule()[0],gameday="2026-10-13")]
    _,blocks=eligible_grades(rows,revised,stats(),now="2026-10-17T01:00:00Z")
    assert blocks["ORIGINAL_KICKOFF_CHANGED_AUDIT_REQUIRED"]==7


def test_append_only_original_and_correction_history(tmp_path):
    rows=frozen_rows(tmp_path)
    a,_=eligible_grades(rows,schedule(),stats(True),now="2026-10-17T01:00:00Z")
    path=tmp_path/"grades.jsonl"
    first=append_grades(a,path)
    assert first["new_records"]==7
    assert append_grades(a,path)["new_records"]==0
    changed=copy.deepcopy(a)
    for r in changed:
        r["official_stat_value"]=5.
        r["target_over_given_official_stat_row"]=int(5>r["synthetic_line"])
    next=append_grades(changed,path)
    assert next["provider_corrections_appended"]==7
    all_records=read_jsonl(path)
    assert len(all_records)==14
    assert all(r["revision"]==2 for r in all_records[7:])


def test_grouped_game_bootstrap_requires_20_independent_games(tmp_path):
    rows=frozen_rows(tmp_path)
    graded,_=eligible_grades(rows,schedule(),stats(),now="2026-10-17T01:00:00Z")
    for r in graded:r["revision"]=1
    x=grade_metrics(graded)
    assert x["grade_counts"]==7
    assert x["by_market"]["receptions"]["distinct_source_games"]==1
    assert x["by_market"]["receptions"]["same_game_cluster_ci95"] is None
    assert x["by_market"]["receptions"]["four_market_adjusted_ci98_75"] is None
    assert not x["validated_BetMGM_profit"] and x["wagers"]==0


def test_sources_from_two_pregame_models_cannot_be_mixed(tmp_path):
    f,s,b,c=raw_pair()
    s[0]["base_2024_CDF_sha256"]="fake"
    with pytest.raises(ValueError,match="catalog mismatch"):
        freeze_pairs(f,s,b,c,now="2026-10-12T21:10:00Z",path=tmp_path/"x")
