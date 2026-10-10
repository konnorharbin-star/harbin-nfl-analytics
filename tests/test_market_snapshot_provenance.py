import csv
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl.espn_market import ESPNTwoWayMarket
from nfl.line_history import SNAPSHOT_FIELDS, append_market_snapshots, load_market_snapshots


def inputs():
    capture = datetime(2026, 10, 10, 15, tzinfo=UTC)
    market = ESPNTwoWayMarket(
        game_id="g",
        market_type="moneyline",
        provider="espn",
        book="Fixture",
        source_event_id="123",
        captured_at=capture,
        first_side="home",
        first_line=None,
        first_american_odds=-110,
        second_side="away",
        second_line=None,
        second_american_odds=100,
        source_quote_at=capture - timedelta(minutes=1),
        source_quote_time_verified=True,
    )
    games = pl.DataFrame(
        [
            {
                "game_id": "g",
                "season": 2026,
                "week": 5,
                "home_team": "BUF",
                "away_team": "NE",
                "gameday": "2026-10-11",
                "gametime": "13:00",
            }
        ]
    )
    return market, games


def test_legacy_prefix_is_unchanged_and_moneyline_replays_are_idempotent(tmp_path):
    market, games = inputs()
    path = tmp_path / "markets.csv"
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS).writeheader()
    prefix = path.read_bytes()
    append_market_snapshots([market], games, path=path)
    original = path.read_bytes()
    assert original.startswith(prefix)
    assert append_market_snapshots([market], games, path=path)["appended_rows"] == 0
    assert path.read_bytes() == original
    row = load_market_snapshots(path).row(0, named=True)
    assert row["source_quote_time_verified"] is True
    assert row["source_quote_at"] == market.source_quote_at.isoformat()


def test_collector_time_never_becomes_book_origin(tmp_path):
    market, games = inputs()
    market = replace(market, source_quote_at=None, source_quote_time_verified=False)
    path = tmp_path / "markets.csv"
    append_market_snapshots([market], games, path=path)
    row = load_market_snapshots(path).row(0, named=True)
    assert row["source_quote_at"] is None and row["source_quote_time_verified"] is False
    event = json.loads(next(path.with_name(path.name + ".events").glob("*.json")).read_text())
    assert event["capture_is_book_update_time"] is False


def test_future_origin_fails_closed_and_corrupt_headers_are_not_overwritten(tmp_path):
    market, games = inputs()
    market = replace(market, source_quote_at=market.captured_at + timedelta(minutes=1))
    path = tmp_path / "markets.csv"
    append_market_snapshots([market], games, path=path)
    assert load_market_snapshots(path).row(0, named=True)["origin_time_usable"] is False
    broken = tmp_path / "broken.csv"
    broken.write_text("wrong,header\nkeep,me\n")
    before = broken.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        append_market_snapshots([market], games, path=broken)
    assert broken.read_bytes() == before


def test_late_archival_cannot_certify_an_old_pregame_quote(tmp_path):
    market, games = inputs()
    capture = datetime(2026, 9, 30, 15, tzinfo=UTC)
    market = replace(market, captured_at=capture, source_quote_at=capture - timedelta(minutes=1))
    games = games.with_columns(pl.lit("2026-10-01").alias("gameday"))
    path = tmp_path / "markets.csv"
    append_market_snapshots([market], games, path=path)
    assert load_market_snapshots(path).row(0, named=True)["origin_time_usable"] is False
