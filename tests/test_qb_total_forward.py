from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.qb_total_forward import (
    QB_TOTAL_SPEC_VERSION,
    append_qb_total_forward_predictions,
    grade_qb_total_forward_predictions,
    load_qb_total_forward_predictions,
)
from nfl.qb_validated import ValidatedQBAdjustment


def _projection(game_ids: tuple[str, ...] = ("g1",)) -> pl.DataFrame:
    rows = []
    for index, game_id in enumerate(game_ids):
        baseline = 50.0 + index
        shadow = 47.0 + index
        rows.append(
            {
                "season": 2026,
                "week": 5,
                "game_id": game_id,
                "home_team": f"H{index}",
                "away_team": f"A{index}",
                "baseline_total": baseline,
                "qb_total_shadow_total": shadow,
                "qb_total_shadow_correction": shadow - baseline,
                "qb_total_release_state": "SHADOW",
            }
        )
    return pl.DataFrame(rows)


def _targets(game_ids: tuple[str, ...] = ("g1",)) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": game_id,
                "gameday": "2026-10-04",
                "gametime": "13:00",
            }
            for game_id in game_ids
        ]
    )


def test_forward_capture_is_first_snapshot_only(tmp_path) -> None:
    path = tmp_path / "qb_forward.csv"
    captured = datetime(2026, 10, 1, 12, tzinfo=UTC)

    first = append_qb_total_forward_predictions(
        _projection(),
        _targets(),
        captured_at=captured,
        path=path,
    )
    second = append_qb_total_forward_predictions(
        _projection(),
        _targets(),
        captured_at=captured,
        path=path,
    )

    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    assert second["skipped_duplicate"] == 1
    stored = load_qb_total_forward_predictions(path)
    assert stored.height == 1
    assert stored.get_column("spec_version")[0] == QB_TOTAL_SPEC_VERSION
    assert stored.get_column("training_seasons")[0] == "2022;2023;2024;2025"


def test_forward_capture_rejects_post_kickoff_snapshot(tmp_path) -> None:
    result = append_qb_total_forward_predictions(
        _projection(),
        _targets(),
        captured_at=datetime(2026, 10, 5, 12, tzinfo=UTC),
        path=tmp_path / "qb_forward.csv",
    )

    assert result["appended_rows"] == 0
    assert result["skipped_post_kickoff"] == 1


def test_forward_grading_uses_only_persisted_pregame_rows(tmp_path) -> None:
    game_ids = ("g1", "g2", "g3")
    path = tmp_path / "qb_forward.csv"
    append_qb_total_forward_predictions(
        _projection(game_ids),
        _targets(game_ids),
        captured_at=datetime(2026, 10, 1, 12, tzinfo=UTC),
        path=path,
    )
    predictions = load_qb_total_forward_predictions(path)
    schedules = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "game_id": game_id,
                "home_score": 24.0 + index,
                "away_score": 23.0,
            }
            for index, game_id in enumerate(game_ids)
        ]
    )

    graded, summary = grade_qb_total_forward_predictions(
        schedules,
        predictions,
        minimum_games=1,
        bootstrap_iterations=500,
    )

    assert graded.height == 3
    assert summary["graded_games"] == 3
    assert summary["total_mae_improvement"] > 0
    assert summary["total_rmse_improvement"] > 0
    assert summary["status"] == "PROMOTION_EVIDENCE"
    assert summary["canonical_score_adjustment_enabled"] is False


def test_forward_grading_rejects_non_2026_rows() -> None:
    predictions = pl.DataFrame(
        [
            {
                "spec_version": QB_TOTAL_SPEC_VERSION,
                "captured_at": "2025-10-01T12:00:00+00:00",
                "kickoff": "2025-10-04T17:00:00+00:00",
                "season": 2025,
                "week": 5,
                "game_id": "g1",
                "baseline_total": 50.0,
                "qb_total_shadow_total": 48.0,
                "qb_total_feature_set": "quality",
                "qb_total_ridge_alpha": 0.1,
                "qb_prior_dropbacks": 75.0,
                "training_seasons": "2022;2023;2024;2025",
                "qb_total_release_state": "SHADOW",
            }
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 5,
                "game_id": "g1",
                "home_score": 24.0,
                "away_score": 24.0,
            }
        ]
    )

    graded, summary = grade_qb_total_forward_predictions(schedules, predictions)

    assert graded.is_empty()
    assert summary["status"] == "NO_VALID_PREGAME_PREDICTIONS"
    assert summary["promotion_eligible"] is False


def test_validated_qb_adjustment_forces_margin_to_zero() -> None:
    rows = []
    for index in range(120):
        margin_signal = ((index % 9) - 4) / 10.0
        epa_total = ((index % 7) - 3) / 10.0
        cpoe_total = float((index % 5) - 2)
        rows.append(
            {
                "margin_residual": 2.0 * margin_signal,
                "total_residual": (1.5 * epa_total) + (0.1 * cpoe_total),
                "qb_epa_margin_signal": margin_signal,
                "qb_epa_total_signal": epa_total,
                "qb_cpoe_total_signal": cpoe_total,
            }
        )
    adjustment = ValidatedQBAdjustment().fit(pl.DataFrame(rows))
    frame = pl.DataFrame(
        [
            {
                "baseline_home_margin": 3.5,
                "baseline_total": 44.0,
                "qb_epa_margin_signal": 0.8,
                "qb_epa_total_signal": 0.4,
                "qb_cpoe_total_signal": 2.0,
            }
        ]
    )

    adjusted = adjustment.apply(frame)

    assert adjusted.get_column("qb_margin_correction")[0] == 0.0
    assert adjusted.get_column("qb_adjusted_home_margin")[0] == 3.5
    assert adjusted.get_column("qb_total_shadow_total")[0] == adjusted.get_column(
        "qb_adjusted_total"
    )[0]
