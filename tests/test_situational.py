from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from nfl.situational import team_situational_features
from nfl.situational_dataset import build_situational_week_snapshot
from nfl.situational_fixed import evaluate_situational_rolling


def _schedules() -> pl.DataFrame:
    games = [
        (1, "g1", "A", "B", 20, 24),
        (1, "g2", "C", "D", 17, 21),
        (2, "g3", "B", "C", 21, 20),
        (2, "g4", "D", "A", 14, 28),
        (3, "g5", "C", "A", 17, 30),
        (3, "g6", "B", "D", 20, 23),
        (4, "g7", "A", "D", 27, 20),
        (4, "g8", "C", "B", 24, 21),
        (5, "g9", "B", "A", 17, 31),
        (5, "g10", "D", "C", 19, 22),
    ]
    start = date(2025, 9, 7)
    return pl.DataFrame(
        {
            "season": [2025] * len(games),
            "week": [game[0] for game in games],
            "game_id": [game[1] for game in games],
            "game_type": ["REG"] * len(games),
            "gameday": [start + timedelta(days=7 * (game[0] - 1)) for game in games],
            "away_team": [game[2] for game in games],
            "home_team": [game[3] for game in games],
            "away_score": [game[4] for game in games],
            "home_score": [game[5] for game in games],
        }
    )


def _pbp(target_week_shift: float = 0.0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    play_id = 0
    situations = (
        (1, 10.0, 50.0, 1, 0, 1, 0, 0.20),
        (2, 6.0, 15.0, 0, 1, 0, 0, 0.35),
        (3, 7.0, 42.0, 1, 0, 1, 0, 0.15),
        (3, 1.0, 8.0, 0, 1, 0, 0, 0.30),
        (1, 10.0, 60.0, 1, 0, 1, 1, -0.90),
    )
    for game in _schedules().iter_rows(named=True):
        week = int(game["week"])
        away = str(game["away_team"])
        home = str(game["home_team"])
        for offense, defense, team_bias in (
            (away, home, -0.05),
            (home, away, 0.08),
        ):
            for down, ydstogo, yardline, pass_attempt, rush_attempt, dropback, sack, epa in situations:
                play_id += 1
                value = epa + team_bias
                if week == 3:
                    value += target_week_shift
                rows.append(
                    {
                        "season": 2025,
                        "week": week,
                        "game_id": game["game_id"],
                        "play_id": play_id,
                        "posteam": offense,
                        "defteam": defense,
                        "epa": value,
                        "success": int(value > 0),
                        "pass_attempt": pass_attempt,
                        "rush_attempt": rush_attempt,
                        "qb_dropback": dropback,
                        "sack": sack,
                        "down": down,
                        "ydstogo": ydstogo,
                        "yardline_100": yardline,
                    }
                )
    return pl.DataFrame(rows)


def test_team_situational_features_are_finite_and_complete() -> None:
    features = team_situational_features(_pbp().filter(pl.col("week") <= 2))

    assert features.height == 4
    assert set(features.get_column("team")) == {"A", "B", "C", "D"}
    expected = {
        "off_red_zone_epa",
        "off_third_down_success",
        "off_short_yardage_success",
        "off_sack_rate",
        "off_early_down_pass_rate",
        "off_plays_per_game",
        "def_red_zone_epa_allowed",
        "def_third_down_success_allowed",
        "def_short_yardage_success_allowed",
        "def_sack_rate_generated",
        "def_early_down_pass_rate_allowed",
        "def_plays_per_game_allowed",
    }
    assert expected.issubset(features.columns)
    assert features.select(pl.exclude("team").is_finite().all()).row(0) == tuple(
        True for _ in range(len(features.columns) - 1)
    )


def test_target_week_pbp_cannot_change_situational_snapshot() -> None:
    low = build_situational_week_snapshot(
        _schedules(),
        _pbp(target_week_shift=-50.0),
        2025,
        3,
        score_ridge=4.0,
    )
    high = build_situational_week_snapshot(
        _schedules(),
        _pbp(target_week_shift=50.0),
        2025,
        3,
        score_ridge=4.0,
    )

    assert low.equals(high)
    assert "sit_red_zone_epa_margin_signal" in low.columns
    assert "sit_sack_rate_total_signal" in low.columns


def _rolling_dataset(*, signal: bool) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(30):
            x = ((index % 11) - 5) / 5.0
            z = ((index % 7) - 3) / 4.0
            margin_residual = (3.0 * x) + (0.5 * z) if signal else 0.0
            total_residual = (2.0 * x) - (0.4 * z) if signal else 0.0
            rows.append(
                {
                    "season": season,
                    "margin_residual": margin_residual,
                    "total_residual": total_residual,
                    "actual_home_margin": margin_residual,
                    "baseline_home_margin": 0.0,
                    "actual_total": 44.0 + total_residual,
                    "baseline_total": 44.0,
                    "sit_red_zone_epa_margin_signal": x,
                    "sit_sack_rate_margin_signal": z,
                    "sit_red_zone_epa_total_signal": x,
                    "sit_sack_rate_total_signal": z,
                }
            )
    return pl.DataFrame(rows)


def test_fixed_situational_selector_can_find_stable_signal() -> None:
    evaluation = evaluate_situational_rolling(
        _rolling_dataset(signal=True),
        feature_sets=("redzone_pressure",),
        ridge_grid=(0.1,),
        min_training_games=24,
    )

    assert evaluation.margin.shadow_candidate is True
    assert evaluation.total.shadow_candidate is True
    assert evaluation.margin.feature_set == "redzone_pressure"
    assert evaluation.total.feature_set == "redzone_pressure"
    assert evaluation.margin.positive_folds == 3
    assert evaluation.total.positive_folds == 3
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False


def test_fixed_situational_selector_falls_back_to_zero() -> None:
    evaluation = evaluate_situational_rolling(
        _rolling_dataset(signal=False),
        feature_sets=("redzone_pressure",),
        ridge_grid=(0.1,),
        min_training_games=24,
    )

    assert evaluation.margin.feature_set == "disabled"
    assert evaluation.total.feature_set == "disabled"
    assert evaluation.margin.shadow_candidate is False
    assert evaluation.total.shadow_candidate is False
