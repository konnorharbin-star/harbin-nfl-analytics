from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.probability_forward import (
    FROZEN_LOGISTIC_ALPHA,
    FROZEN_MARGIN_SCALE,
    FROZEN_TOTAL_SCALE,
    PROBABILITY_SPEC_VERSION,
    PROBABILITY_TRAINING_SEASONS,
    append_probability_forward_predictions,
    attach_probability_shadow,
    grade_probability_forward_predictions,
)


def _historical() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    game = 0
    for season in PROBABILITY_TRAINING_SEASONS:
        for week in range(1, 5):
            for index in range(4):
                margin = float((index - 1.5) * 3.0 + (season - 2021) * 0.2)
                total = 43.0 + float(index) + 0.3 * (season - 2021)
                actual_margin = margin + (-1.5 if game % 3 == 0 else 1.0)
                if game % 11 == 0:
                    actual_margin = 0.0
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"h-{season}-{week}-{index}",
                        "projected_home_margin": margin,
                        "projected_total": total,
                        "actual_home_margin": actual_margin,
                        "actual_total": total + (-3.0 if game % 2 else 2.0),
                    }
                )
                game += 1
    return pl.DataFrame(rows)


def _projection() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026],
            "week": [4],
            "game_id": ["2026_04_A_B"],
            "home_team": ["B"],
            "away_team": ["A"],
            "baseline_home_margin": [2.5],
            "baseline_total": [46.0],
        }
    )


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2026_04_A_B"],
            "gameday": ["2026-10-04"],
            "gametime": ["13:00"],
        }
    )


def test_probability_shadow_attaches_without_changing_canonical_scores() -> None:
    projection = _projection()
    shadowed, meta = attach_probability_shadow(projection, _historical())

    assert shadowed.get_column("baseline_home_margin").to_list() == [2.5]
    assert shadowed.get_column("baseline_total").to_list() == [46.0]
    assert shadowed.get_column("probability_release_state").to_list() == ["SHADOW"]
    assert 0.0 < shadowed["baseline_gaussian_home_win_probability"][0] < 1.0
    assert 0.0 < shadowed["shadow_logistic_home_win_probability"][0] < 1.0
    assert shadowed["shadow_total_sigma"][0] < shadowed["baseline_total_sigma"][0]
    assert meta["margin_scale"] == FROZEN_MARGIN_SCALE
    assert meta["total_scale"] == FROZEN_TOTAL_SCALE
    assert meta["logistic_alpha"] == FROZEN_LOGISTIC_ALPHA
    assert meta["canonical_probability_change_enabled"] is False


def test_probability_forward_capture_is_prekickoff_and_deduplicated(tmp_path) -> None:
    shadowed, _ = attach_probability_shadow(_projection(), _historical())
    path = tmp_path / "probability.csv"
    captured = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)

    first = append_probability_forward_predictions(
        shadowed,
        _targets(),
        captured_at=captured,
        path=path,
    )
    second = append_probability_forward_predictions(
        shadowed,
        _targets(),
        captured_at=captured,
        path=path,
    )

    assert first["appended_rows"] == 1
    assert first["total_rows"] == 1
    assert second["appended_rows"] == 0
    assert second["skipped_duplicate"] == 1

    late = append_probability_forward_predictions(
        shadowed,
        _targets(),
        captured_at=datetime(2026, 10, 5, 12, 0, tzinfo=UTC),
        path=tmp_path / "late.csv",
    )
    assert late["appended_rows"] == 0
    assert late["skipped_post_kickoff"] == 1


def test_probability_forward_capture_rejects_non_2026_rows(tmp_path) -> None:
    shadowed, _ = attach_probability_shadow(_projection(), _historical())
    wrong = shadowed.with_columns(pl.lit(2025).alias("season"))
    result = append_probability_forward_predictions(
        wrong,
        _targets(),
        captured_at=datetime(2026, 10, 1, 20, 0, tzinfo=UTC),
        path=tmp_path / "wrong.csv",
    )
    assert result["appended_rows"] == 0
    assert result["skipped_wrong_season"] == 1


def test_probability_forward_grading_keeps_home_and_total_gates_separate() -> None:
    predictions: list[dict[str, object]] = []
    schedules: list[dict[str, object]] = []
    for index in range(8):
        game_id = f"2026_04_{index}"
        predictions.append(
            {
                "ledger_version": 1,
                "spec_version": PROBABILITY_SPEC_VERSION,
                "captured_at": "2026-10-01T18:00:00+00:00",
                "kickoff": "2026-10-04T17:00:00+00:00",
                "season": 2026,
                "week": 4,
                "game_id": game_id,
                "home_team": f"H{index}",
                "away_team": f"A{index}",
                "baseline_home_margin": 1.0,
                "baseline_total": 45.0,
                "baseline_gaussian_home_win_probability": 0.50,
                "shadow_logistic_home_win_probability": 0.80,
                "baseline_total_mean": 45.0,
                "baseline_total_sigma": 10.0,
                "shadow_total_mean": 50.0,
                "shadow_total_sigma": 9.0,
                "margin_scale": FROZEN_MARGIN_SCALE,
                "total_scale": FROZEN_TOTAL_SCALE,
                "logistic_alpha": FROZEN_LOGISTIC_ALPHA,
                "training_seasons": ";".join(
                    str(value) for value in PROBABILITY_TRAINING_SEASONS
                ),
                "release_state": "SHADOW",
            }
        )
        schedules.append(
            {
                "season": 2026,
                "week": 4,
                "game_id": game_id,
                "home_score": 30,
                "away_score": 20,
            }
        )

    graded, summary = grade_probability_forward_predictions(
        pl.DataFrame(schedules),
        pl.DataFrame(predictions),
        minimum_games=4,
        bootstrap_iterations=500,
    )

    assert graded.height == 8
    assert summary["home_win"]["status"] == "PROMOTION_EVIDENCE"
    assert summary["total_distribution"]["status"] == "PROMOTION_EVIDENCE"
    assert summary["promotion_eligible"] is True
    assert summary["canonical_probability_change_enabled"] is False
