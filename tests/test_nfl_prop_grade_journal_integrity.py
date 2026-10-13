"""Synthetic-only 2026 first-seen player-stat grade journal trust boundary."""
from copy import deepcopy
import json

import pytest

from scripts.nfl_prop_grade_journal_integrity import (
    validate_grade_journal, record_hash,
)
from scripts.nfl_prop_paired_cdf_forward_grader import (
    append_grades, eligible_grades, read_jsonl,
)
from test_nfl_prop_paired_cdf_forward_grader import frozen_rows, schedule, stats


def first_grade(tmp_path):
    frozen=frozen_rows(tmp_path)
    grades,blocked=eligible_grades(
        frozen,schedule(),stats(),now="2026-10-17T01:00:00Z")
    assert len(grades)==7 and not blocked
    path=tmp_path/"graded.jsonl"
    created=append_grades(grades,path,originals=frozen)
    assert created["new_records"]==7
    return frozen,grades,path


def test_first_seen_grade_signed_and_verified_against_frozen_sha(tmp_path):
    originals,_,path=first_grade(tmp_path)
    graded=read_jsonl(path)
    verified=validate_grade_journal(graded,originals)
    assert len(verified)==7
    assert all(x["revision"]==1 and
               x["grade_record_sha256"]==record_hash(x) and
               x["previous_grade_sha256"] is None for x in graded)


def test_source_correction_linked_and_latest_is_only_revision_scored(tmp_path):
    originals,grades,path=first_grade(tmp_path)
    updates=deepcopy(grades)
    for r in updates:
        r["official_stat_value"]=5
        r["target_over_given_official_stat_row"]=int(5>r["synthetic_line"])
        r["grade_source_captured_utc"]="2026-10-18T01:00:00Z"
    out=append_grades(updates,path,originals=originals)
    assert out["provider_corrections_appended"]==7
    full=read_jsonl(path)
    latest=validate_grade_journal(full,originals)
    assert len(full)==14 and len(latest)==7
    for first,second in zip(full[:7],full[7:]):
        assert second["previous_grade_sha256"]==first["grade_record_sha256"]
        assert second["record_type"]=="APPENDED_OFFICIAL_SOURCE_STAT_CORRECTION"
        assert second["previous_official_stat_value"]==4


def test_false_over_result_cannot_be_published_even_when_record_hash_is_valid(tmp_path):
    frozen=frozen_rows(tmp_path)
    grades,_=eligible_grades(frozen,schedule(),stats(),now="2026-10-17T01:00:00Z")
    bad=deepcopy(grades)
    bad[0]["target_over_given_official_stat_row"]=1-int(
        bad[0]["target_over_given_official_stat_row"])
    path=tmp_path/"bad.jsonl"
    with pytest.raises(ValueError,match="outcome conflicts"):
        append_grades(bad,path,originals=frozen)
    assert not path.exists()


def test_grade_cannot_change_original_probabilities_or_forecast_identity(tmp_path):
    frozen=frozen_rows(tmp_path)
    grades,_=eligible_grades(frozen,schedule(),stats(),now="2026-10-17T01:00:00Z")
    for field,bad in [
        ("p_original",0.999),("p_snap",0.999),("original_row_sha256","fake"),
        ("team","LA"),("kickoff_utc","2026-10-14T00:15:00Z")
    ]:
        tampered=deepcopy(grades)
        tampered[0][field]=bad
        with pytest.raises(ValueError,match="grade (changed|replaced)"):
            append_grades(tampered,tmp_path/("x"+field),originals=frozen)


def test_grade_journal_modified_after_freeze_must_fail_sha(tmp_path):
    originals,_,path=first_grade(tmp_path)
    altered=read_jsonl(path)
    altered[0]["official_stat_value"]=100.0
    with pytest.raises(ValueError,match="tampered or unsigned"):
        validate_grade_journal(altered,originals)


def test_signed_grade_cannot_rewrite_original_outcome(tmp_path):
    originals,_,path=first_grade(tmp_path)
    changed=read_jsonl(path)
    changed[0]["official_stat_value"]=100.0
    changed[0]["grade_record_sha256"]=record_hash(changed[0])
    # Changing a signed row is detectable by its mathematical/statistic proof,
    # although external append-only hosting is additionally needed to prove
    # a motivated adversary did not replace both content and the digest.
    with pytest.raises(ValueError,match="outcome conflicts"):
        validate_grade_journal(changed,originals)


def test_grade_before_72_hours_fails_even_when_sha_is_consistent(tmp_path):
    originals,_,path=first_grade(tmp_path)
    changed=read_jsonl(path)
    changed[0]["grade_source_captured_utc"]="2026-10-13T01:00:00Z"
    changed[0]["grade_record_sha256"]=record_hash(changed[0])
    with pytest.raises(ValueError,match="72-hour"):
        validate_grade_journal(changed,originals)


def test_duplicate_grade_revision_or_broken_correction_predecessor_fails(tmp_path):
    originals,_,path=first_grade(tmp_path)
    grades=read_jsonl(path)
    with pytest.raises(ValueError,match="sequential and unique"):
        validate_grade_journal(grades+[grades[0]],originals)
    replaced=deepcopy(grades[0])
    replaced.update(revision=2,
        record_type="APPENDED_OFFICIAL_SOURCE_STAT_CORRECTION",
        official_stat_value=5, target_over_given_official_stat_row=1,
        previous_official_stat_value=4,previous_grade_sha256="wrong",
        grade_source_captured_utc="2026-10-18T01:00:00Z")
    replaced["grade_record_sha256"]=record_hash(replaced)
    with pytest.raises(ValueError,match="predecessor chain"):
        validate_grade_journal(grades+[replaced],originals)


def test_provider_correction_cannot_change_frozen_odds_or_model(tmp_path):
    originals,grades,path=first_grade(tmp_path)
    old=path.read_text()
    changed=deepcopy(grades)
    changed[0]["p_original"]=0.03
    with pytest.raises(ValueError,match="original frozen forecast"):
        append_grades(changed,path,originals=originals)
    assert path.read_text()==old


def test_missing_grade_source_field_rejects_before_write(tmp_path):
    frozen=frozen_rows(tmp_path)
    grades,_=eligible_grades(frozen,schedule(),stats(),now="2026-10-17T01:00:00Z")
    del grades[0]["grade_source_captured_utc"]
    with pytest.raises(KeyError):
        append_grades(grades,tmp_path/"bad.jsonl",originals=frozen)


def test_no_source_rows_is_not_zero_or_under(tmp_path):
    frozen=frozen_rows(tmp_path)
    assert validate_grade_journal([],frozen)=={}
