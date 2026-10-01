from datetime import date, timedelta

import polars as pl

from nfl.online_state import (
    ONLINE_STATE_CONFIGS,
    ONLINE_STATE_FEATURES,
    build_online_state_features,
)
from nfl.online_state_fixed import evaluate_online_state_rolling


def _schedules(week3_home_score: int = 27, week2_home_score: int = 24) -> pl.DataFrame:
    games = [
        (2022, 1, "a1", "A", "B", 17, 20),
        (2022, 1, "a2", "C", "D", 21, 24),
        (2022, 2, "a3", "B", "C", 20, 23),
        (2022, 2, "a4", "D", "A", 17, 28),
        (2023, 1, "b1", "A", "C", 20, 23),
        (2023, 1, "b2", "B", "D", 17, 21),
        (2023, 2, "b3", "C", "B", 20, week2_home_score),
        (2023, 2, "b4", "D", "A", 17, 30),
        (2023, 3, "b5", "B", "A", 20, week3_home_score),
        (2023, 3, "b6", "D", "C", 19, 22),
    ]
    start = date(2022, 9, 1)
    return pl.DataFrame(
        {
            "season": [row[0] for row in games],
            "week": [row[1] for row in games],
            "game_id": [row[2] for row in games],
            "game_type": ["REG"] * len(games),
            "gameday": [start + timedelta(days=7 * i) for i in range(len(games))],
            "away_team": [row[3] for row in games],
            "home_team": [row[4] for row in games],
            "away_score": [row[5] for row in games],
            "home_score": [row[6] for row in games],
        }
    )


def test_same_week_scores_cannot_change_same_week_state_features() -> None:
    config = ONLINE_STATE_CONFIGS[1]
    base = build_online_state_features(_schedules(week3_home_score=27), config)
    mutated = build_online_state_features(_schedules(week3_home_score=60), config)

    base_week = base.filter((pl.col("season") == 2023) & (pl.col("week") == 3)).select(
        ["game_id", *ONLINE_STATE_FEATURES]
    )
    mutated_week = mutated.filter(
        (pl.col("season") == 2023) & (pl.col("week") == 3)
    ).select(["game_id", *ONLINE_STATE_FEATURES])
    assert base_week.equals(mutated_week)


def test_prior_week_scores_change_later_state() -> None:
    config = ONLINE_STATE_CONFIGS[1]
    base = build_online_state_features(_schedules(week2_home_score=24), config)
    mutated = build_online_state_features(_schedules(week2_home_score=50), config)

    base_week = base.filter((pl.col("season") == 2023) & (pl.col("week") == 3)).select(
        list(ONLINE_STATE_FEATURES)
    )
    mutated_week = mutated.filter(
        (pl.col("season") == 2023) & (pl.col("week") == 3)
    ).select(list(ONLINE_STATE_FEATURES))
    assert not base_week.equals(mutated_week)


def _zero_residual_dataset(config_name: str) -> pl.DataFrame:
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for i in range(180):
            baseline_margin = ((i % 11) - 5) * 0.5
            baseline_total = 42.0 + (i % 9)
            row = {
                "season": season,
                "margin_residual": 0.0,
                "total_residual": 0.0,
                "actual_home_margin": baseline_margin,
                "baseline_home_margin": baseline_margin,
                "actual_total": baseline_total,
                "baseline_total": baseline_total,
                "state_config": config_name,
            }
            for j, name in enumerate(ONLINE_STATE_FEATURES):
                row[name] = float((i + j + season) % 17)
            rows.append(row)
    return pl.DataFrame(rows)


def test_online_state_gate_keeps_zero_when_candidate_is_not_strictly_better() -> None:
    datasets = {
        config.name: _zero_residual_dataset(config.name)
        for config in ONLINE_STATE_CONFIGS
    }
    evaluation = evaluate_online_state_rolling(datasets)

    assert evaluation.margin.state_config == "disabled"
    assert evaluation.margin.blend_weight == 0.0
    assert evaluation.margin.shadow_candidate is False
    assert evaluation.total.state_config == "disabled"
    assert evaluation.total.blend_weight == 0.0
    assert evaluation.total.shadow_candidate is False
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False
