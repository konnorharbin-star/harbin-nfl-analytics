from datetime import UTC, datetime

from nfl.pregame_refresh import refresh_checkpoint


def _report(kickoff: str) -> dict[str, object]:
    return {
        "meta": {"season": 2026, "week": 5},
        "games": [
            {
                "game_id": "2026_05_AAA_BBB",
                "kickoff": kickoff,
            }
        ],
    }


def test_refresh_checkpoint_triggers_once_in_120_minute_band() -> None:
    now = datetime(2026, 10, 11, 15, 0, tzinfo=UTC)
    result = refresh_checkpoint(
        _report("2026-10-11T17:00:00+00:00"),
        now=now,
    )

    assert result["run"] is True
    assert result["season"] == 2026
    assert result["week"] == 5
    assert result["checkpoint_minutes"] == 120
    assert result["due_games"][0]["minutes_to_kickoff"] == 120.0


def test_refresh_checkpoint_triggers_75_and_final_20_minute_bands() -> None:
    kickoff = "2026-10-11T17:00:00+00:00"

    seventy = refresh_checkpoint(
        _report(kickoff),
        now=datetime(2026, 10, 11, 15, 50, tzinfo=UTC),
    )
    final = refresh_checkpoint(
        _report(kickoff),
        now=datetime(2026, 10, 11, 16, 50, tzinfo=UTC),
    )

    assert seventy["run"] is True
    assert seventy["checkpoint_minutes"] == 75
    assert final["run"] is True
    assert final["checkpoint_minutes"] == 20


def test_refresh_checkpoint_does_not_run_outside_target_windows() -> None:
    result = refresh_checkpoint(
        _report("2026-10-11T17:00:00+00:00"),
        now=datetime(2026, 10, 11, 14, 30, tzinfo=UTC),
    )

    assert result["run"] is False
    assert result["due_games"] == []


def test_refresh_checkpoint_ignores_games_after_kickoff() -> None:
    result = refresh_checkpoint(
        _report("2026-10-11T17:00:00+00:00"),
        now=datetime(2026, 10, 11, 17, 1, tzinfo=UTC),
    )

    assert result["run"] is False
