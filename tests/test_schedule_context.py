from datetime import date, timedelta

import polars as pl

from nfl.schedule_context import (
    MARGIN_CONTEXT_FEATURES,
    TOTAL_CONTEXT_FEATURES,
    build_schedule_context_week_snapshot,
    schedule_context_features,
)
from nfl.schedule_context_eval import evaluate_schedule_context


def _schedules() -> pl.DataFrame:
    games = [
        (2024, 16, "p1", "SEA", "MIA", 20, 24),
        (2024, 16, "p2", "DEN", "BUF", 17, 21),
        (2024, 17, "p3", "MIA", "DEN", 21, 20),
        (2024, 17, "p4", "BUF", "SEA", 14, 28),
        (2025, 1, "g1", "SEA", "BUF", 24, 27),
        (2025, 1, "g2", "MIA", "DEN", 17, 20),
        (2025, 2, "g3", "BUF", "MIA", 21, 23),
        (2025, 2, "g4", "DEN", "SEA", 20, 30),
        (2025, 3, "g5", "MIA", "SEA", 17, 31),
        (2025, 3, "g6", "DEN", "BUF", 19, 22),
    ]
    start = date(2024, 12, 1)
    return pl.DataFrame(
        {
            "season": [game[0] for game in games],
            "week": [game[1] for game in games],
            "game_id": [game[2] for game in games],
            "game_type": ["REG"] * len(games),
            "gameday": [start + timedelta(days=7 * index) for index in range(len(games))],
            "away_team": [game[3] for game in games],
            "home_team": [game[4] for game in games],
            "away_score": [game[5] for game in games],
            "home_score": [game[6] for game in games],
            "away_rest": [7.0] * len(games),
            "home_rest": [7.0] * len(games),
            "location": ["Home"] * len(games),
        }
    )


def test_schedule_context_builds_rest_and_travel_signals() -> None:
    features = schedule_context_features(
        {
            "away_team": "SEA",
            "home_team": "MIA",
            "gameday": date(2025, 10, 5),
            "away_rest": 4.0,
            "home_rest": 7.0,
            "location": "Home",
        }
    )

    assert features["ctx_rest_diff_days"] == 3.0
    assert features["ctx_short_rest_edge"] == 1.0
    assert features["ctx_away_travel_1000"] > 2.0
    assert features["ctx_timezone_shift_hours"] == 3.0
    assert features["ctx_neutral_site"] == 0.0
    assert features["ctx_travel_available"] == 1.0


def test_target_week_scores_cannot_change_context_predictors() -> None:
    schedules = _schedules()
    base = build_schedule_context_week_snapshot(schedules, 2025, 3)
    mutated_schedules = schedules.with_columns(
        pl.when((pl.col("season") == 2025) & (pl.col("week") == 3))
        .then(pl.lit(99))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when((pl.col("season") == 2025) & (pl.col("week") == 3))
        .then(pl.lit(1))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )
    mutated = build_schedule_context_week_snapshot(mutated_schedules, 2025, 3)

    context_columns = list(dict.fromkeys(MARGIN_CONTEXT_FEATURES + TOTAL_CONTEXT_FEATURES))
    predictor_columns = [
        "game_id",
        "baseline_home_margin",
        "baseline_total",
        *context_columns,
    ]
    assert base.select(predictor_columns).equals(mutated.select(predictor_columns))


def test_schedule_context_gate_falls_back_to_zero_adjustment() -> None:
    feature_columns = list(dict.fromkeys(MARGIN_CONTEXT_FEATURES + TOTAL_CONTEXT_FEATURES))
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(40):
            margin_actual = 1.0 if index % 2 == 0 else -1.0
            total_actual = 41.0 if index % 2 == 0 else 39.0
            row: dict[str, object] = {
                "season": season,
                "week": 5 + (index % 10),
                "game_id": f"{season}-{index}",
                "actual_home_margin": margin_actual,
                "baseline_home_margin": 0.0,
                "margin_residual": margin_actual,
                "actual_total": total_actual,
                "baseline_total": 40.0,
                "total_residual": total_actual - 40.0,
            }
            row.update({column: 0.0 for column in feature_columns})
            rows.append(row)
    dataset = pl.DataFrame(rows)

    evaluation = evaluate_schedule_context(
        dataset,
        test_seasons=(2023, 2024, 2025),
        alpha_grid=(1.0, 10.0),
    )

    assert evaluation.margin.selected_alpha is None
    assert evaluation.total.selected_alpha is None
    assert evaluation.margin.shadow_candidate is False
    assert evaluation.total.shadow_candidate is False
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False
