from datetime import date, timedelta

import polars as pl

from nfl.drive_dataset import build_drive_week_snapshot
from nfl.drive_features import DRIVE_METRICS, team_drive_features
from nfl.drive_fixed import evaluate_drive_rolling


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


def _pbp(target_week_score_shift: float = 0.0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    play_id = 0
    for game in _schedules().iter_rows(named=True):
        week = int(game["week"])
        for offense, defense in (
            (str(game["away_team"]), str(game["home_team"])),
            (str(game["home_team"]), str(game["away_team"])),
        ):
            score = 0.0
            for drive, scored, turnover in ((1, 3.0, 0), (2, 0.0, 1)):
                for index, (play_type, yardline, first_down) in enumerate(
                    (("pass", 75.0, 1), ("run", 45.0, 1), ("pass", 18.0, 0)),
                    start=1,
                ):
                    play_id += 1
                    score_before = score
                    score_after = score
                    if index == 3:
                        score_after += scored
                        if week == 3:
                            score_after += target_week_score_shift
                    rows.append(
                        {
                            "season": 2025,
                            "week": week,
                            "game_id": game["game_id"],
                            "play_id": play_id,
                            "drive": drive,
                            "posteam": offense,
                            "defteam": defense,
                            "play_type": play_type,
                            "yardline_100": yardline if drive == 1 else yardline + 20.0,
                            "first_down": first_down,
                            "interception": turnover if index == 3 else 0,
                            "fumble_lost": 0,
                            "posteam_score": score_before,
                            "posteam_score_post": score_after,
                        }
                    )
                    score = score_after
    return pl.DataFrame(rows)


def test_team_drive_features_are_finite() -> None:
    features = team_drive_features(_pbp().filter(pl.col("week") <= 2))

    assert features.height == 4
    assert set(features.get_column("team")) == {"A", "B", "C", "D"}
    for metric in DRIVE_METRICS:
        assert f"off_{metric}" in features.columns
        assert f"def_{metric}_allowed" in features.columns
    assert features.null_count().select(pl.all().sum()).row(0)[0] == 0


def test_target_week_drive_data_cannot_change_snapshot() -> None:
    low = build_drive_week_snapshot(
        _schedules(),
        _pbp(target_week_score_shift=-50.0),
        2025,
        3,
        score_ridge=4.0,
    )
    high = build_drive_week_snapshot(
        _schedules(),
        _pbp(target_week_score_shift=50.0),
        2025,
        3,
        score_ridge=4.0,
    )

    assert low.equals(high)
    assert low.height == 2
    assert "drive_points_per_drive_margin_signal" in low.columns
    assert "drive_points_per_drive_total_signal" in low.columns


def test_drive_rolling_falls_back_to_zero_when_baseline_is_exact() -> None:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(160):
            row: dict[str, object] = {
                "season": season,
                "baseline_home_margin": float((index % 9) - 4),
                "actual_home_margin": float((index % 9) - 4),
                "margin_residual": 0.0,
                "baseline_total": float(40 + (index % 7)),
                "actual_total": float(40 + (index % 7)),
                "total_residual": 0.0,
            }
            for metric in DRIVE_METRICS:
                row[f"drive_{metric}_margin_signal"] = 0.0
                row[f"drive_{metric}_total_signal"] = 0.0
            rows.append(row)

    evaluation = evaluate_drive_rolling(pl.DataFrame(rows))

    assert evaluation.margin.feature_set == "disabled"
    assert evaluation.total.feature_set == "disabled"
    assert evaluation.margin.shadow_candidate is False
    assert evaluation.total.shadow_candidate is False
    assert evaluation.canonical_score_adjustment_enabled is False
    assert evaluation.promotion_eligible is False
