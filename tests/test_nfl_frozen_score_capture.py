import csv
import io
from datetime import UTC, datetime, timedelta

from scripts.nfl_frozen_score_capture import FIELDS, capture


def make_source(kickoff, margin="3", gid="game-1"):
    row = {
        "game_id": gid, "kickoff": kickoff.isoformat(), "model_margin_home": margin,
        "model_total": "45", "away_team": "AWAY", "home_team": "HOME",
        "calibrated_home_probability": "0.61",
    }
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=list(row))
    w.writeheader()
    w.writerow(row)
    return f.getvalue().encode()


def test_immutable_duplicate_source():
    now = datetime(2026, 10, 8, tzinfo=UTC)
    source = make_source(now + timedelta(days=2))
    one, report = capture(source, [], now)
    two, report2 = capture(source, one, now + timedelta(minutes=5))
    assert report["appended"] == 1
    assert report2["appended"] == 0
    assert len(two) == 1
    assert set(two[0]) == set(FIELDS)


def test_no_post_kickoff_backfill():
    now = datetime(2026, 10, 8, tzinfo=UTC)
    rows, info = capture(make_source(now - timedelta(seconds=1)), [], now)
    assert not rows
    assert info["rejected"]["already_started"] == 1


def test_reject_nonfinite():
    now = datetime(2026, 10, 8, tzinfo=UTC)
    rows, info = capture(make_source(now + timedelta(days=1), margin="NaN"), [], now)
    assert not rows
    assert info["rejected"]["invalid_forecast"] == 1
