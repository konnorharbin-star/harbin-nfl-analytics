"""Protect durable originals, independent game counts and append-only corrections."""
from copy import deepcopy
from datetime import timedelta
import json
import pytest
from nfl.coupled_research_journal import archive, grade_archive, read_journal, _archive_file
import gzip
from tests.test_coupled_forward_grading import orig, finals, players, CAP, KICK, ASOF


def inputs(tmp_path, snapshot=None):
    scan = snapshot or orig()
    scan["season"] = 2026
    replay = {"spec": "NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1",
              "capture_utc": scan["capture_utc"],
              "games": [{"game_id": g["game_id"], "forecast": g["frozen_football_score_input"]}
                        for g in scan["games"]]}
    source, replay_path = tmp_path / "source.json", tmp_path / "replay.json"
    source.write_text(json.dumps(scan))
    replay_path.write_text(json.dumps(replay))
    return source, replay_path


def freeze(tmp_path, **kwargs):
    source, replay = inputs(tmp_path, kwargs.pop("snapshot", None))
    return archive(tmp_path/"journal", source, replay,
                   at=kwargs.pop("at", CAP+timedelta(minutes=5)),
                   run_id="12345", code_sha="b"*40, **kwargs)


def test_preserve_original_bytes_and_do_not_replace_or_recount_repeated_games(tmp_path):
    first = freeze(tmp_path)
    root = tmp_path/"journal"
    before = (root/"primary.jsonl").read_bytes()
    snap = orig()
    snap["games"][0]["distribution"]["TOTAL"]["mean"] = 99
    again = freeze(tmp_path, snapshot=snap)
    assert first["new_primary_games"] == 1
    assert again["new_primary_games"] == 0
    assert again["repeated_games_not_recounted"] == 1
    assert (root/"primary.jsonl").read_bytes() == before
    assert len(read_journal(root/"capture_receipts.jsonl")) == 2
    receipt = read_journal(root/"primary.jsonl")[0]
    stored = root/"snapshots"/(receipt["snapshot_sha256"]+".json")
    assert json.loads(stored.read_bytes())["games"][0]["distribution"]["TOTAL"]["mean"] == 48


def test_no_postkickoff_archive_even_with_earlier_forecast_timestamp(tmp_path):
    result = freeze(tmp_path, at=KICK)
    assert result["new_primary_games"] == 0
    assert "POSTKICKOFF" in result["blocked"][0]["reason"]
    assert not (tmp_path/"journal"/"primary.jsonl").exists()


def test_replay_must_match_and_missing_context_is_visible(tmp_path):
    source, replay = inputs(tmp_path)
    data = json.loads(replay.read_text())
    data["games"][0]["forecast"]["independent_projected_total"] = 100
    replay.write_text(json.dumps(data))
    report = archive(tmp_path/"journal", source, replay, at=CAP,
                     run_id="123", code_sha="b"*40, sidecars=[tmp_path/"missing.json"])
    assert report["new_primary_games"] == 0
    assert report["blocked"][0]["reason"] == "REPLAY_NOT_IDENTICAL_TO_ORIGINAL"
    assert report["missing_supplemental_sources"] == [str(tmp_path/"missing.json")]


def test_automatic_grade_is_idempotent_and_corrections_append_not_overwrite(tmp_path):
    freeze(tmp_path)
    root = tmp_path/"journal"
    source = deepcopy((finals(), players()))
    first = grade_archive(root, *source, at=ASOF)
    assert first["distinct_primary_games_graded"] == 1
    original_bytes = (root/"grades.jsonl").read_bytes()
    second = grade_archive(root, *source, at=ASOF+timedelta(hours=1))
    assert second["new_or_corrected_grade_events"] == 0
    assert (root/"grades.jsonl").read_bytes() == original_bytes
    revised = players()
    revised[0]["receiving_yards"] = 91
    third = grade_archive(root, finals(), revised, at=ASOF+timedelta(days=1))
    assert third["new_or_corrected_grade_events"] == 1
    assert (root/"grades.jsonl").read_bytes().startswith(original_bytes)
    assert third["distinct_primary_games_graded"] == 1
    assert third["remaining_to_128_game_research_review"] == 127
    assert third["promotion_ready"] is False
    assert third["profitable_edge_validated"] is False
    assert grade_archive(root, finals(), revised, at=ASOF+timedelta(days=2))["new_or_corrected_grade_events"] == 0


def test_72h_maturity_and_source_integrity_fail_closed(tmp_path):
    freeze(tmp_path)
    root = tmp_path/"journal"
    report = grade_archive(root, finals(), players(), at=KICK+timedelta(hours=71))
    assert report["distinct_primary_games_graded"] == 0
    receipt = read_journal(root/"primary.jsonl")[0]
    (root/"snapshots"/(receipt["snapshot_sha256"]+".json")).write_text("{}")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        grade_archive(root, finals(), players(), at=ASOF)


def test_duplicate_source_and_journal_receipt_tampering_rejected(tmp_path):
    snap = orig()
    snap["games"].append(deepcopy(snap["games"][0]))
    with pytest.raises(ValueError, match="DUPLICATE"):
        freeze(tmp_path, snapshot=snap)
    freeze(tmp_path)
    root = tmp_path/"journal"
    path = root/"primary.jsonl"
    path.write_text(path.read_text().replace('"12345"', '"99999"'))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        grade_archive(root, finals(), players(), at=ASOF)


def test_model_revision_cannot_silently_pool_into_registered_cohort(tmp_path):
    freeze(tmp_path, model_sha="a"*64)
    with pytest.raises(ValueError, match="MODEL_CHANGED"):
        freeze(tmp_path, model_sha="c"*64)


def test_large_raw_json_source_is_losslessly_compressed_and_idempotent(tmp_path):
    data=json.dumps({"source_rows":["raw football source"]*250000}).encode()
    sha=_archive_file(tmp_path,data)
    path=tmp_path/"snapshots"/(sha+".json.gz")
    assert path.exists()
    assert gzip.decompress(path.read_bytes())==data
    first=path.read_bytes()
    assert _archive_file(tmp_path,data)==sha
    assert path.read_bytes()==first
