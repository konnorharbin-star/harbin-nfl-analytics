"""Catch silent stale NFL model/PNG publication before kickoff."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nfl.publication_watchdog import evaluate_publication_watchdog

NOW = datetime(2026, 10, 11, 19, 30, tzinfo=UTC)
MANIFEST = {"status": "PASS", "file_count": 24, "reason": "verified"}


def _model(
    *,
    age: int = 10,
    until: int = 100,
    games: int = 1,
    release: str = "RESEARCH",
    approved: float = 0.0,
):
    upcoming = (NOW + timedelta(minutes=until)).isoformat()
    return {
        "meta": {
            "generated_at": (NOW - timedelta(minutes=age)).isoformat(),
            "season": 2026,
            "week": 5,
        },
        "generated_at": (NOW - timedelta(minutes=age)).isoformat(),
        "release_state": release,
        "portfolio": {"approved_units": approved},
        "publication": {"latest_png": "outputs/latest.png"},
        "games": [
            {"game_id": "2026_05_BAL_ATL", "kickoff": upcoming}
            for _ in range(games)
        ],
    }


def _check(model, *, manifest=MANIFEST):
    return evaluate_publication_watchdog(
        model, now=NOW, manifest=manifest
    )


def test_valid_pregame_publication_passes_and_counts_unique_games():
    result = _check(_model(games=3))
    assert result["status"] == "PASS_PREGAME_FRESH"
    assert result["imminent_games"] == ["2026_05_BAL_ATL"]
    assert result["verified_file_count"] == 24
    assert result["max_model_age_minutes"] == 130
    assert result["watchdog_changes_bets_or_scores"] is False


def test_stale_pre_kickoff_model_is_a_failed_action_not_silent_old_png():
    result = _check(_model(age=131, until=100))
    assert result["status"] == "FAIL"
    assert "STALE_MODEL_NEAR_KICKOFF" in result["blocking_reasons"][0]


def test_final_kickoff_window_requires_recent_model_refresh():
    fail = _check(_model(age=40, until=12))
    assert fail["status"] == "FAIL"
    assert fail["max_model_age_minutes"] == 35
    passes = _check(_model(age=15, until=12))
    assert passes["status"] == "PASS_PREGAME_FRESH"


def test_old_report_outside_critical_window_does_not_spam_actions():
    result = _check(_model(age=1440, until=210))
    assert result["status"] == "PASS_OUTSIDE_PREGAME_WINDOW"
    assert result["max_model_age_minutes"] is None
    assert result["imminent_games"] == []


def test_corrupted_manifest_is_always_an_explicit_failure():
    result = _check(_model(until=210), manifest={
        "status": "FAIL", "reason": "stale page 2",
    })
    assert result["status"] == "FAIL"
    assert "PUBLICATION_MANIFEST_INVALID" in result["blocking_reasons"][0]


def test_missing_or_future_run_timestamp_cannot_be_fresh():
    model = _model()
    del model["meta"]["generated_at"]
    missing = _check(model)
    assert "MODEL_GENERATED_AT_UNKNOWN_OR_UNTZONED" in missing["blocking_reasons"]
    model = _model(age=-6)
    future = _check(model)
    assert "MODEL_GENERATED_IN_FUTURE" in future["blocking_reasons"]


def test_nonproduction_stake_mismatch_is_a_readonly_alarm():
    model = _model(approved=0.25)
    result = _check(model)
    assert result["status"] == "FAIL"
    assert "NONPRODUCTION_HAS_APPROVED_STAKE_OR_INVALID_PORTFOLIO" in (
        result["blocking_reasons"]
    )
    assert model["portfolio"]["approved_units"] == 0.25


def test_duplicate_market_kickoff_mismatch_is_detected():
    model = _model(games=2)
    model["games"][1]["kickoff"] = (
        NOW + timedelta(minutes=40)
    ).isoformat()
    result = _check(model)
    assert result["status"] == "FAIL"
    assert any(
        reason.startswith("GAME_KICKOFF_INCONSISTENT")
        for reason in result["blocking_reasons"]
    )


def test_watchdog_rejects_naive_reference_clock():
    with pytest.raises(ValueError, match="timezone"):
        evaluate_publication_watchdog(
            _model(), now=datetime(2026, 10, 11, 19, 30), manifest=MANIFEST
        )


def test_watchdog_workflow_is_read_only_and_does_not_cancel_main_model():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/nfl-publication-watchdog.yml"
    ).read_text(encoding="utf-8")
    assert "contents: read" in workflow
    assert "python -m scripts.check_publication_watchdog" in workflow
    assert "ref: main" in workflow
    assert "nfl-published-board-watchdog-" in workflow
    assert "uses: ./.github/workflows/nfl-model.yml" not in workflow
    assert "git push" not in workflow
