from datetime import date, timedelta

import polars as pl

from nfl.factorized_eval import evaluate_factorized_architecture
from nfl.factorized_score import (
    build_factorized_week_predictions,
    factorized_training_tables,
)


def _schedules() -> pl.DataFrame:
    games = [
        (2024, 16, "p1", "A", "B", 20, 24),
        (2024, 16, "p2", "C", "D", 17, 21),
        (2024, 17, "p3", "B", "C", 21, 20),
        (2024, 17, "p4", "D", "A", 14, 28),
        (2025, 1, "g1", "A", "C", 24, 27),
        (2025, 1, "g2", "B", "D", 17, 20),
        (2025, 2, "g3", "C", "B", 21, 23),
        (2025, 2, "g4", "D", "A", 20, 30),
        (2025, 3, "g5", "B", "A", 17, 31),
        (2025, 3, "g6", "D", "C", 19, 22),
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
        }
    )


def _pbp(target_extra_drives: int = 0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for game in _schedules().iter_rows(named=True):
        base_drives = 10
        extra = target_extra_drives if int(game["season"]) == 2025 and int(game["week"]) == 3 else 0
        for offense in (str(game["away_team"]), str(game["home_team"])):
            for drive in range(1, base_drives + extra + 1):
                rows.append(
                    {
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "game_id": str(game["game_id"]),
                        "drive": drive,
                        "posteam": offense,
                    }
                )
    return pl.DataFrame(rows)


def test_factorized_training_tables_use_positive_drive_counts() -> None:
    efficiency, possessions = factorized_training_tables(_schedules(), _pbp(), 2025, 3)

    assert efficiency.height == 16
    assert possessions.height == 8
    assert efficiency.get_column("points_for").min() > 0
    assert possessions.get_column("shared_drives").min() == 10.0


def test_target_week_pbp_cannot_change_factorized_projection() -> None:
    base = build_factorized_week_predictions(
        _schedules(),
        _pbp(target_extra_drives=0),
        2025,
        3,
        factorized_ridge=8.0,
    )
    mutated = build_factorized_week_predictions(
        _schedules(),
        _pbp(target_extra_drives=50),
        2025,
        3,
        factorized_ridge=8.0,
    )

    assert base.equals(mutated)
    assert base.height == 2
    assert "factorized_shared_drives" in base.columns
    assert "factorized_home_ppd" in base.columns


def test_architecture_gate_falls_back_to_canonical_when_not_strictly_better(
    monkeypatch,
) -> None:
    def fake_walkforward(*args, season: int, **kwargs) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "season": [season, season],
                "week": [5, 5],
                "game_id": [f"{season}-a", f"{season}-b"],
                "actual_home_margin": [3.0, -4.0],
                "actual_total": [44.0, 47.0],
                "baseline_home_margin": [2.0, -3.0],
                "baseline_total": [43.0, 48.0],
                "factorized_home_margin": [2.0, -3.0],
                "factorized_total": [43.0, 48.0],
            }
        )

    monkeypatch.setattr("nfl.factorized_eval.build_factorized_walkforward", fake_walkforward)
    evaluation = evaluate_factorized_architecture(
        pl.DataFrame(),
        pl.DataFrame(),
        test_seasons=(2023, 2024, 2025),
        ridge_grid=(2.0, 8.0),
    )

    assert evaluation.selected_ridge is None
    assert evaluation.shadow_candidate is False
    assert evaluation.selected_margin_mae == evaluation.baseline_margin_mae
    assert evaluation.selected_total_rmse == evaluation.baseline_total_rmse
    assert evaluation.canonical_score_change_enabled is False
    assert evaluation.promotion_eligible is False
