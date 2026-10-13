"""Synthetic fixtures only: NOT evidence of actual U.S. sportsbook prices."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.nfl_prop_exact_public_market_shadow import (
    create_rows, append_first_seen, run, hash_object, VERSION,
)


def fixture():
    start = datetime(2026, 10, 12, 17, tzinfo=timezone.utc)
    iso = lambda dt: dt.isoformat()
    game_time = start + timedelta(hours=7)
    capture = start + timedelta(hours=2)
    vectors = [-1.0, 0.0, 1.0]*100
    catalog = {"calibration_version": "test-fixture",
               "training_year": 2024,
               "vectors": {"receiving_yards|WR": vectors}}
    catalog["sha256"] = hash_object(catalog)
    forecast = {
        "model": "NFL_PLAYER_PROP_FROZEN_FULL_DISTRIBUTION_V1",
        "game_id": "2026_05_BUF_LA", "player_id": "TEST_GSIS",
        "team": "BUF", "market": "receiving_yards", "position": "WR",
        "first_observed_at_utc": iso(start),
        "kickoff_utc": iso(game_time), "original_point_mean": 75,
        "original_scale": 18*.25+.45*75,
        "residual_catalog_sha256": catalog["sha256"],
        "residual_group_key": "receiving_yards|WR",
        "training_residual_count": len(vectors),
    }
    q = {
        "quote_id": "synthetic-observation-01",
        "game_id": forecast["game_id"], "player_id": forecast["player_id"],
        "team": "BUF", "market": forecast["market"],
        "kickoff_utc": iso(game_time), "period": "FULL_GAME",
        "bookmaker": "PUBLIC_MOCK_BOOK", "book_region": "US-ATTRIBUTED",
        "snapshot_id": "mock-snap-01",
        "source_url": "https://example.org/fictional-quote",
        "source_observed_utc": iso(capture),
        "source_quote_origin_utc": iso(capture-timedelta(minutes=5)),
        "source_origin_timestamp_claimed_verified": True,
        "line": 75.5, "sides": [
            {"selection": "OVER", "line": 75.5, "american_odds": -110},
            {"selection": "UNDER", "line": 75.5, "american_odds": -110},
        ]
    }
    for side in q["sides"]:
        for field in ("snapshot_id", "bookmaker", "source_url",
                      "source_observed_utc", "source_quote_origin_utc"):
            side[field] = q[field]
    return q, forecast, catalog, iso(capture+timedelta(minutes=1))


def test_exact_two_sided_same_player_snapshot_is_research_only():
    q, f, cat, now = fixture()
    row = create_rows([q], [f], cat, now=now)[0]
    assert row["origin_provenance"] == "CLAIMED_RECENT_ORIGIN_UNAUTHENTICATED"
    assert row["source_pair_over_no_vig_conditional_on_nonpush"] == 0.5
    assert row["p_model_push_given_official_stat_row"] == 0
    assert 0 < row["p_model_over_given_official_stat_row"] < 1
    assert row["estimated_ev"] is None and row["conservative_ev"] is None
    assert row["model_minus_book_comparable_probability_delta"] is None
    assert not row["account_accepted_price_verified"]
    assert row["decision"] == "PASS" and not row["real_bet"]


def test_missing_origin_stays_capture_only_not_current_quote():
    q, f, cat, now = fixture()
    q["source_quote_origin_utc"] = None
    q["source_origin_timestamp_claimed_verified"] = False
    for side in q["sides"]:
        side["source_quote_origin_utc"] = q["source_quote_origin_utc"]
    row = create_rows([q], [f], cat, now=now)[0]
    assert row["origin_provenance"] == "CAPTURE_ONLY_ORIGIN_UNKNOWN"
    assert not row["source_origin_independently_authenticated"]


def test_missing_origin_cannot_claim_verified_timestamp():
    q, f, cat, now = fixture()
    q["source_quote_origin_utc"] = None
    for side in q["sides"]:
        side["source_quote_origin_utc"] = q["source_quote_origin_utc"]
    with pytest.raises(ValueError, match="missing time"):
        create_rows([q], [f], cat, now=now)


def test_old_source_timestamp_is_blocked_from_freshness_claim():
    q, f, cat, now = fixture()
    q["source_quote_origin_utc"] = (
        datetime.fromisoformat(q["source_observed_utc"]) -
        timedelta(minutes=45)).isoformat()
    for side in q["sides"]:
        side["source_quote_origin_utc"] = q["source_quote_origin_utc"]
    row = create_rows([q], [f], cat, now=now)[0]
    assert row["origin_provenance"] == "SOURCE_ORIGIN_STALE_AT_CAPTURE"


def test_offer_issued_before_frozen_forecast_is_not_marked_fresh():
    q, f, cat, now = fixture()
    q["source_quote_origin_utc"] = (
        datetime.fromisoformat(f["first_observed_at_utc"]) -
        timedelta(minutes=1)).isoformat()
    for side in q["sides"]:
        side["source_quote_origin_utc"] = q["source_quote_origin_utc"]
    row = create_rows([q], [f], cat, now=now)[0]
    assert row["origin_provenance"] == "ORIGIN_PREDATES_FIRST_FROZEN_FORECAST"


@pytest.mark.parametrize("key,value", [
    ("game_id", "wrong-game"), ("player_id", "another-player"),
    ("market", "passing_yards"),
])
def test_cannot_match_different_player_game_or_stat(key, value):
    q, f, cat, now = fixture()
    q[key] = value
    with pytest.raises(ValueError, match="no exact frozen"):
        create_rows([q], [f], cat, now=now)


def test_team_and_kickoff_mismatch_block():
    q, f, cat, now = fixture()
    q["team"] = "LA"
    with pytest.raises(ValueError, match="identity disagrees"):
        create_rows([q], [f], cat, now=now)


def test_cannot_backfill_after_kickoff():
    q, f, cat, now = fixture()
    with pytest.raises(ValueError, match="after kickoff"):
        create_rows([q], [f], cat, now=q["kickoff_utc"])


def test_missing_other_side_or_different_line_fail_closed():
    q, f, cat, now = fixture()
    q["sides"] = q["sides"][:1]
    with pytest.raises(ValueError, match="two-sided"):
        create_rows([q], [f], cat, now=now)
    q, f, cat, now = fixture()
    q["sides"][1]["line"] = 76.5
    with pytest.raises(ValueError, match="same contract"):
        create_rows([q], [f], cat, now=now)


def test_future_book_quote_origin_rejected():
    q, f, cat, now = fixture()
    q["source_quote_origin_utc"] = now
    for side in q["sides"]:
        side["source_quote_origin_utc"] = q["source_quote_origin_utc"]
    with pytest.raises(ValueError, match="cannot follow observation"):
        create_rows([q], [f], cat, now=now)


def test_integer_line_keeps_nonzero_push_and_no_ev():
    q, f, cat, now = fixture()
    q["line"] = 75
    for side in q["sides"]:
        side["line"] = 75
    row = create_rows([q], [f], cat, now=now)[0]
    assert row["p_model_push_given_official_stat_row"] > 0
    assert row["estimated_ev"] is None


def test_book_account_execution_claim_forbidden():
    q, f, cat, now = fixture()
    q["account_price_accepted"] = True
    with pytest.raises(ValueError, match="must not assert execution"):
        create_rows([q], [f], cat, now=now)


def test_modified_catalog_rejected():
    q, f, cat, now = fixture()
    cat["vectors"]["receiving_yards|WR"][0] = -99.0
    with pytest.raises(ValueError, match="integrity failure"):
        create_rows([q], [f], cat, now=now)


def test_duplicate_quote_id_is_error():
    q, f, cat, now = fixture()
    with pytest.raises(ValueError, match="duplicate quote_id"):
        create_rows([q, q], [f], cat, now=now)


def test_append_only_original_checksum_and_rewrite(tmp_path):
    q, f, cat, now = fixture()
    first = create_rows([q], [f], cat, now=now)
    target = tmp_path / "original.jsonl"
    assert append_first_seen(first, target)["new"] == 1
    assert append_first_seen(first, target)["new"] == 0
    q["sides"][0]["american_odds"] = -120
    changed = create_rows([q], [f], cat, now=now)
    with pytest.raises(ValueError, match="refusing to rewrite"):
        append_first_seen(changed, target)
    old = json.loads(target.read_text().strip())
    old["line"] = 90.5
    target.write_text(json.dumps(old) + "\n")
    with pytest.raises(ValueError, match="integrity failure"):
        append_first_seen([], target)


def test_no_input_does_not_create_real_quotes(tmp_path):
    p = tmp_path / "input.json"
    p.write_text('{"records":[]}')
    report = run(quotes_path=p, journal_path=tmp_path/"journal.jsonl",
                 report_path=tmp_path/"report.json")
    assert report["candidate_public_quote_observations"] == 0
    assert report["first_seen_quote_model_joins"]["new"] == 0
    assert report["decision"] == "PASS" and not report["calibrated_edge_found"]
    assert report["true_ev"] is None
    assert not (tmp_path/"journal.jsonl").exists()

def test_mixed_source_snapshots_do_not_create_false_two_sided_hold():
    q, f, cat, now = fixture()
    q["sides"][1]["snapshot_id"] = "a-different-snapshot"
    with pytest.raises(ValueError, match="mismatched provenance"):
        create_rows([q], [f], cat, now=now)


def test_source_origin_field_must_be_explicitly_null_when_unknown():
    q, f, cat, now = fixture()
    del q["source_quote_origin_utc"]
    with pytest.raises(ValueError, match="incomplete"):
        create_rows([q], [f], cat, now=now)
