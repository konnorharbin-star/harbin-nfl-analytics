from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.recent_form_current import blocked_recent_form_shadow
from nfl.recent_form_forward import (
    append_recent_form_forward_predictions,
    grade_recent_form_forward_predictions,
    load_recent_form_forward_predictions,
)


def _projection() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
                "baseline_total": 44.0,
                "recent_form_shadow_total": 45.5,
                "recent_form_total_adjustment": 1.5,
                "recent_form_total_release_state": "SHADOW",
            }
        ]
    )


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": "2026_04_AAA_BBB",
                "gameday": "2026-10-04",
                "gametime": "13:00",
            }
        ]
    )


def test_forward_ledger_keeps_first_pregame_prediction(tmp_path) -> None:
    path = tmp_path / "recent.csv"
    captured = datetime(2026, 10, 1, 12, tzinfo=UTC)
    first = append_recent_form_forward_predictions(
        _projection(), _targets(), captured_at=captured, path=path
    )
    second = append_recent_form_forward_predictions(
        _projection(), _targets(), captured_at=datetime(2026, 10, 2, 12, tzinfo=UTC), path=path
    )

    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    assert second["skipped_duplicate"] == 1
    stored = load_recent_form_forward_predictions(path)
    assert stored.height == 1
    assert stored.get_column("captured_at")[0] == captured.isoformat()


def test_forward_ledger_rejects_post_kickoff_capture(tmp_path) -> None:
    result = append_recent_form_forward_predictions(
        _projection(),
        _targets(),
        captured_at=datetime(2026, 10, 4, 18, tzinfo=UTC),
        path=tmp_path / "recent.csv",
    )
    assert result["appended_rows"] == 0
    assert result["skipped_post_kickoff"] == 1


def test_forward_grading_is_insufficient_below_gate(tmp_path) -> None:
    path = tmp_path / "recent.csv"
    append_recent_form_forward_predictions(
        _projection(),
        _targets(),
        captured_at=datetime(2026, 10, 1, 12, tzinfo=UTC),
        path=path,
    )
    predictions = load_recent_form_forward_predictions(path)
    schedules = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "home_score": 24.0,
                "away_score": 21.0,
            }
        ]
    )
    graded, summary = grade_recent_form_forward_predictions(
        schedules,
        predictions,
        minimum_games=128,
        bootstrap_iterations=500,
    )

    assert graded.height == 1
    assert graded.get_column("actual_total")[0] == 45.0
    assert summary["status"] == "SHADOW_INSUFFICIENT_SAMPLE"
    assert summary["promotion_eligible"] is False


def test_blocked_shadow_preserves_canonical_total() -> None:
    canonical = pl.DataFrame(
        [{"game_id": "g1", "baseline_total": 47.25, "baseline_home_margin": 2.0}]
    )
    blocked = blocked_recent_form_shadow(canonical)

    assert blocked.get_column("baseline_total")[0] == 47.25
    assert blocked.get_column("recent_form_shadow_total")[0] is None
    assert blocked.get_column("recent_form_total_release_state")[0] == "BLOCKED"
