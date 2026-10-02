from __future__ import annotations

import polars as pl

from nfl.forward_ledger_health import audit_candidate_ledger


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "early",
                "gameday": "2026-10-01",
                "gametime": "20:15",
                "game_type": "REG",
            },
            {
                "season": 2026,
                "week": 4,
                "game_id": "sunday-a",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "game_type": "REG",
            },
            {
                "season": 2026,
                "week": 4,
                "game_id": "sunday-b",
                "gameday": "2026-10-04",
                "gametime": "16:25",
                "game_type": "REG",
            },
        ]
    )


def _row(
    game_id: str,
    *,
    captured_at: str = "2026-10-02T00:52:00+00:00",
    kickoff: str = "2026-10-04T17:00:00+00:00",
) -> dict[str, object]:
    return {
        "season": 2026,
        "week": 4,
        "game_id": game_id,
        "captured_at": captured_at,
        "kickoff": kickoff,
    }


def test_inception_excludes_games_that_already_kicked_off() -> None:
    predictions = pl.DataFrame(
        [
            _row("sunday-a"),
            _row("sunday-b", kickoff="2026-10-04T20:25:00+00:00"),
        ]
    )

    report = audit_candidate_ledger(
        _schedules(),
        predictions,
        name="candidate",
        validator=lambda row: True,
    )

    assert report["status"] == "PASS"
    assert report["eligible_games"] == 2
    assert report["captured_eligible_games"] == 2
    assert report["capture_coverage"] == 1.0
    assert report["missing_game_ids"] == []


def test_opened_week_missing_snapshot_fails_health_gate() -> None:
    predictions = pl.DataFrame([_row("sunday-a")])

    report = audit_candidate_ledger(
        _schedules(),
        predictions,
        name="candidate",
        validator=lambda row: True,
    )

    assert report["status"] == "FAIL"
    assert report["eligible_games"] == 2
    assert report["captured_eligible_games"] == 1
    assert report["missing_game_ids"] == ["sunday-b"]
    assert report["promotion_sample_eligible"] is False


def test_duplicate_or_late_rows_fail_health_gate() -> None:
    predictions = pl.DataFrame(
        [
            _row("sunday-a"),
            _row(
                "sunday-a",
                captured_at="2026-10-03T00:00:00+00:00",
            ),
            _row(
                "sunday-b",
                captured_at="2026-10-04T21:00:00+00:00",
                kickoff="2026-10-04T20:25:00+00:00",
            ),
        ]
    )

    report = audit_candidate_ledger(
        _schedules(),
        predictions,
        name="candidate",
        validator=lambda row: True,
    )

    assert report["status"] == "FAIL"
    assert report["duplicate_rows"] == 1
    assert report["invalid_timing_rows"] == 1
    assert report["promotion_sample_eligible"] is False
