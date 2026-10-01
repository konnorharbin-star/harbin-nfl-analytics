from datetime import date, timedelta

import polars as pl

from nfl.decomposition_eval import evaluate_scoring_decomposition
from nfl.scoring_decomposition import (
    build_decomposed_week_predictions,
    decomposition_training_tables,
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


def _pbp(target_score_shift: float = 0.0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for game in _schedules().iter_rows(named=True):
        target = int(game["season"]) == 2025 and int(game["week"]) == 3
        for offense in (str(game["away_team"]), str(game["home_team"])):
            score = 0.0
            for delta in (7.0, 3.0):
                score_post = score + delta + (target_score_shift if target else 0.0)
                rows.append(
                    {
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "game_id": str(game["game_id"]),
                        "posteam": offense,
                        "posteam_score": score,
                        "posteam_score_post": score_post,
                    }
                )
                score = score_post
    return pl.DataFrame(rows)


def test_decomposition_training_tables_split_final_points() -> None:
    possession, remainder = decomposition_training_tables(
        _schedules(),
        _pbp(),
        2025,
        3,
    )

    assert possession.height == 16
    assert remainder.height == 16
    assert possession.get_column("points_for").min() == 10.0
    assert possession.get_column("points_for").max() == 10.0
    assert remainder.get_column("points_for").min() >= 4.0


def test_target_week_score_deltas_cannot_change_pregame_projection() -> None:
    base = build_decomposed_week_predictions(
        _schedules(),
        _pbp(target_score_shift=0.0),
        2025,
        3,
        remainder_ridge=None,
    )
    mutated = build_decomposed_week_predictions(
        _schedules(),
        _pbp(target_score_shift=50.0),
        2025,
        3,
        remainder_ridge=None,
    )

    assert base.equals(mutated)
    assert base.height == 2
    assert "candidate_home_possession_points" in base.columns
    assert "candidate_home_remainder_points" in base.columns


def test_decomposition_gate_falls_back_when_candidate_ties_baseline(monkeypatch) -> None:
    def fake_walkforward(*args, **kwargs) -> pl.DataFrame:
        season = int(args[2])
        return pl.DataFrame(
            {
                "season": [season, season],
                "week": [5, 5],
                "game_id": [f"{season}-a", f"{season}-b"],
                "actual_home_margin": [3.0, -4.0],
                "actual_total": [44.0, 47.0],
                "baseline_home_margin": [2.0, -3.0],
                "baseline_total": [43.0, 48.0],
                "candidate_home_margin": [2.0, -3.0],
                "candidate_total": [43.0, 48.0],
            }
        )

    monkeypatch.setattr(
        "nfl.decomposition_eval.build_decomposed_walkforward",
        fake_walkforward,
    )
    evaluation = evaluate_scoring_decomposition(
        pl.DataFrame(),
        pl.DataFrame(),
        test_seasons=(2023, 2024, 2025),
    )

    assert evaluation.selected_spec == "canonical"
    assert evaluation.shadow_candidate is False
    assert evaluation.selected_margin_mae == evaluation.baseline_margin_mae
    assert evaluation.selected_total_rmse == evaluation.baseline_total_rmse
    assert evaluation.canonical_score_change_enabled is False
    assert evaluation.promotion_eligible is False
