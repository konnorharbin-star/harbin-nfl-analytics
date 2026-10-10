"""Fail-closed source and paper settlement checks: no actual betting."""
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.forward_price_validation import (
    capture_audit,
    entry_reasons,
    evaluate_paper,
    pair_probability,
    source_reasons,
    utc,
)

NOW = datetime(2026, 10, 10, 17, tzinfo=UTC)
KICKOFF = NOW + timedelta(hours=6)


def stamp(dt):
    return dt.isoformat()


def complete_entry():
    return {
        "receipt_id": "fixed-nfl-01",
        "sport": "nfl",
        "game_id": "GAME01",
        "kickoff": stamp(KICKOFF),
        "market": "moneyline",
        "side": "home",
        "line": None,
        "book": "ExampleSportsbook",
        "american_odds": 120,
        "opposite_american_odds": -135,
        "source_url": "https://sportsbook.example.test/markets/GAME01",
        "source_quote_at": stamp(NOW - timedelta(minutes=2)),
        "observed_at": stamp(NOW),
        "frozen_at": stamp(NOW + timedelta(minutes=1)),
        "book_identity_verified": True,
        "source_quote_time_verified": True,
        "executable_price_verified": True,
        "book_access_verified": True,
        "settlement_rules_verified": True,
        "frozen_model_sha256": "a" * 64,
        "model_side_probability": 0.53,
    }


def complete_close(entry):
    return {
        **entry,
        "american_odds": -110,
        "opposite_american_odds": 100,
        "source_quote_at": stamp(KICKOFF - timedelta(minutes=12)),
        "observed_at": stamp(KICKOFF - timedelta(minutes=11)),
        "frozen_at": stamp(KICKOFF - timedelta(minutes=10)),
    }


def full_final(entry, home=27, away=20):
    return {
        "receipt_id": entry["receipt_id"],
        "sport": entry["sport"],
        "game_id": entry["game_id"],
        "status": "FINAL",
        "verified_result_source": True,
        "source_url": "https://scoreboard.example.test/fixture/GAME01",
        "observed_at": stamp(KICKOFF + timedelta(hours=4)),
        "home_score": home,
        "away_score": away,
    }


def write_case(root, entry=None, close=None, final=None):
    for folder, value in (("entries", entry), ("late_quotes", close), ("finals", final)):
        if value is None:
            continue
        destination = root / folder
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (value["receipt_id"] + ".json")).write_text(json.dumps(value))


def publisher(p):
    return KICKOFF - timedelta(minutes=5) if p.parent.name == "late_quotes" else NOW + timedelta(minutes=3)


def test_full_research_replay_has_correct_moneyline_profit_and_direction(tmp_path):
    entry = complete_entry()
    write_case(tmp_path, entry, complete_close(entry), full_final(entry))
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["settled"] == report["wins"] == 1
    assert report["units"] == pytest.approx(1.2)
    assert report["roi_per_entry"] == pytest.approx(1.2)
    assert report["late_market_samples"] == 1
    assert report["average_late_market_probability_change"] > 0
    assert report["late_market_metric_is_certified_closing_line_value"] is False
    assert report["betting_authorized"] is False
    assert report["actual_wagers"] == 0


def test_missing_closing_quote_is_not_imputed_to_zero(tmp_path):
    entry = complete_entry()
    write_case(tmp_path, entry, final=full_final(entry))
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["settled"] == 1
    assert report["late_market_samples"] == 0
    assert report["average_late_market_probability_change"] is None
    assert report["entries"][0]["late_market_status"] == "MISSING_LATE_QUOTE"


def test_late_quote_cannot_change_handicap_or_book(tmp_path):
    entry = complete_entry()
    entry.update(market="spread", side="away", line=4.5,
                 american_odds=-110, opposite_american_odds=-110)
    close = complete_close(entry)
    close["line"] = 5.0
    write_case(tmp_path, entry, close, full_final(entry))
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["settled"] == 1
    assert report["late_market_samples"] == 0
    assert report["entries"][0]["late_market_status"] == "CHANGED_MARKET_HANDICAP_OR_BOOK"


def test_close_after_kickoff_and_unpublished_are_both_blocked(tmp_path):
    entry = complete_entry()
    close = complete_close(entry)
    close["source_quote_at"] = stamp(KICKOFF)
    close["observed_at"] = stamp(KICKOFF + timedelta(seconds=1))
    write_case(tmp_path, entry, close, full_final(entry))
    a = evaluate_paper(tmp_path, commit_time=publisher)
    assert a["late_market_samples"] == 0
    close = complete_close(entry)
    write_case(tmp_path, close=close)
    b = evaluate_paper(
        tmp_path,
        commit_time=lambda p: None if p.parent.name == "late_quotes" else publisher(p),
    )
    assert b["entries"][0]["late_market_status"] == "LATE_QUOTE_NOT_PUBLISHED_PREGAME"


def test_no_publication_or_unverified_source_blocks_paper_entry(tmp_path):
    entry = complete_entry()
    write_case(tmp_path, entry, final=full_final(entry))
    without_commit = evaluate_paper(tmp_path, commit_time=lambda p: None)
    assert without_commit["blocked"] == 1
    assert "MISSING_VERIFIABLE_PREGAME_GIT_COMMIT" in without_commit["entries"][0]["blockers"]
    entry["executable_price_verified"] = False
    write_case(tmp_path, entry)
    with_commit = evaluate_paper(tmp_path, commit_time=publisher)
    assert with_commit["blocked"] == 1
    assert "UNVERIFIED_EXECUTABLE_PRICE_VERIFIED" in with_commit["entries"][0]["blockers"]


def test_invalid_price_and_source_timestamp_are_not_accepted():
    entry = complete_entry()
    assert not entry_reasons(entry, publisher(None))
    entry["american_odds"] = 99
    assert "INVALID_EXACT_PAIRED_PRICE" in entry_reasons(entry, publisher(None))
    entry = complete_entry()
    entry["source_quote_at"] = stamp(NOW - timedelta(hours=3))
    assert "STALE_ORIGIN_QUOTE" in entry_reasons(entry, publisher(None))
    with pytest.raises(ValueError, match="Naive"):
        utc("2026-10-10T17:00:00")


def test_source_quote_requires_both_sides_and_external_evidence():
    r = {
        "market": "moneyline", "line": None, "sides": ["home", "away"],
        "odds": [-380, 295],
        "kickoff": stamp(KICKOFF),
        "observed_at": stamp(NOW),
        "reported_source_time": None,
        "source_url": None,
        "book_identity_verified": False,
        "source_quote_time_verified": False,
        "executable_price_verified": False,
    }
    s = source_reasons(r)
    assert "UNKNOWN_OR_INVALID_ORIGIN_TIME" in s
    assert "UNVERIFIED_EXECUTABLE_PRICE_VERIFIED" in s
    assert "NO_DIRECT_SOURCE_URL" in s
    assert "INVALID_PAIRED_MARKET" not in s
    assert pair_probability(-380, 295) > 0.70


def test_existing_capture_report_yields_zero_when_aggregator_is_unverified(tmp_path):
    root = tmp_path / "captures"
    root.mkdir()
    entry = {
        "market": "moneyline", "line": None, "sides": ["home", "away"],
        "odds": [-380, 295], "kickoff": stamp(KICKOFF),
        "observed_at": stamp(NOW), "reported_source_time": None,
        "source_url": None, "book_identity_verified": False,
        "source_quote_time_verified": False,
        "executable_price_verified": False,
    }
    (root / "A.json").write_text(json.dumps({"source_records": [entry]}))
    (root / "B.json").write_text(json.dumps({"source_records": [deepcopy(entry)]}))
    report = capture_audit(root)
    assert report["raw_source_records"] == 2
    assert report["distinct_source_records"] == 1
    assert report["source_evidence_complete"] == 0
    assert report["betting_authorized"] is False


def test_paper_requires_unique_selection_and_valid_final(tmp_path):
    entry = complete_entry()
    clone = {**entry, "receipt_id": "another"}
    write_case(tmp_path, entry, final=full_final(entry))
    write_case(tmp_path, clone, final=full_final(clone))
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["blocked"] == 1
    assert report["settled"] == 1
    assert report["paper_entries"] == 2


def test_missing_or_unverified_final_stays_pending_not_loss(tmp_path):
    entry = complete_entry()
    final = full_final(entry)
    final["verified_result_source"] = False
    write_case(tmp_path, entry, final=final)
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["pending"] == 1
    assert report["losses"] == 0
    assert report["roi_per_entry"] is None


def test_push_settles_at_zero_units(tmp_path):
    entry = complete_entry()
    entry.update(market="total", side="under", line=47.0,
                 american_odds=-110, opposite_american_odds=-110)
    final = full_final(entry, home=27, away=20)
    write_case(tmp_path, entry, final=final)
    report = evaluate_paper(tmp_path, commit_time=publisher)
    assert report["pushes"] == 1
    assert report["units"] == 0
    assert report["roi_per_entry"] == 0
