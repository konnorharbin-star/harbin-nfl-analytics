from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from nfl.recent_form import team_recent_pbp_features
from nfl.recent_form_dataset import build_recent_form_week_snapshot
from nfl.recent_form_residuals import evaluate_recent_form_rolling


def _schedules() -> pl.DataFrame:
    games = [
        (1, "g1", "A", "B", 20, 24),
        (1, "g2", "C", "D", 17, 21),
        (2, "g3", "B", "C", 21, 20),
        (2, "g4", "D", "A", 14, 28),
        (3, "g5", "C", "A", 17, 30),
        (3, "g6", "B", "D", 20, 23),
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


def _pbp(target_week_epa: float = 0.0) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    play_id = 0
    for game in _schedules().iter_rows(named=True):
        week = int(game["week"])
        away = str(game["away_team"])
        home = str(game["home_team"])
        for offense, defense, base in ((away, home, -0.2), (home, away, 0.4)):
            for is_pass, delta, yards in ((1, 0.3, 22), (0, -0.1, 11)):
                play_id += 1
                epa = base + delta
                if week == 3:
                    epa += target_week_epa
                rows.append(
                    {
                        "season": 2025,
                        "week": week,
                        "game_id": game["game_id"],
                        "play_id": play_id,
                        "posteam": offense,
                        "defteam": defense,
                        "epa": epa,
                        "success": int(epa > 0),
                        "pass_attempt": is_pass,
                        "rush_attempt": 1 - is_pass,
                        "qb_dropback": is_pass,
                        "yards_gained": yards,
                        "down": 1 if is_pass else 2,
                    }
                )
    return pl.DataFrame(rows)


def test_recent_form_uses_game_level_ewma() -> None:
    history = _pbp().filter(pl.col("week") < 3)
    features = team_recent_pbp_features(history, alpha=0.5)
    a = features.filter(pl.col("team") == "A").row(0, named=True)

    # A's offense appears in g1 as home and g4 as home. Each game averages two plays.
    first = (0.4 + 0.3 + 0.4 - 0.1) / 2
    second = (0.4 + 0.3 + 0.4 - 0.1) / 2
    assert a["recent_off_epa_per_play"] == pytest.approx((0.5 * first) + (0.5 * second))
    assert a["recent_off_games"] == 2
    assert a["recent_def_games"] == 2


def test_target_week_pbp_cannot_change_recent_form_snapshot() -> None:
    low = build_recent_form_week_snapshot(
        _schedules(), _pbp(target_week_epa=-50.0), 2025, 3, recent_alpha=0.35, score_ridge=4.0
    )
    high = build_recent_form_week_snapshot(
        _schedules(), _pbp(target_week_epa=50.0), 2025, 3, recent_alpha=0.35, score_ridge=4.0
    )

    assert low.equals(high)
    assert "recent_epa_per_play_margin_signal" in low.columns
    assert "recent_epa_per_play_total_signal" in low.columns


def _rolling_dataset(*, useful: bool, alpha: float) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2022, 2023, 2024, 2025):
        for index in range(60):
            signal = ((index % 9) - 4) / 4.0
            signed = signal if index % 2 == 0 else -signal
            feature = signed if useful and alpha == 0.35 else 0.0
            margin = 2.5 * signed
            total_residual = -1.8 * signed
            rows.append(
                {
                    "season": season,
                    "week": 5 + (index % 12),
                    "game_id": f"{season}-{index}",
                    "baseline_home_margin": 0.0,
                    "actual_home_margin": margin,
                    "margin_residual": margin,
                    "baseline_total": 44.0,
                    "actual_total": 44.0 + total_residual,
                    "total_residual": total_residual,
                    "recent_epa_per_play_margin_signal": feature,
                    "recent_epa_per_play_total_signal": feature,
                }
            )
    return pl.DataFrame(rows).sort(["season", "week", "game_id"])


def test_rolling_selection_finds_nonzero_shadow_candidate() -> None:
    evaluation = evaluate_recent_form_rolling(
        {
            0.20: _rolling_dataset(useful=True, alpha=0.20),
            0.35: _rolling_dataset(useful=True, alpha=0.35),
        },
        feature_sets=("epa",),
        ridge_grid=(1.0, 10.0),
        blend_grid=(0.5, 1.0),
        min_training_games=40,
        min_positive_folds=2,
    )

    assert evaluation.margin.shadow_candidate
    assert evaluation.total.shadow_candidate
    assert evaluation.margin.recent_alpha == pytest.approx(0.35)
    assert evaluation.total.recent_alpha == pytest.approx(0.35)
    assert not evaluation.promotion_eligible


def test_rolling_selection_can_fail_closed_to_zero_weight() -> None:
    evaluation = evaluate_recent_form_rolling(
        {
            0.20: _rolling_dataset(useful=False, alpha=0.20),
            0.35: _rolling_dataset(useful=False, alpha=0.35),
        },
        feature_sets=("epa",),
        ridge_grid=(10.0,),
        blend_grid=(0.5, 1.0),
        min_training_games=40,
        min_positive_folds=2,
    )

    assert evaluation.margin.blend_weight == 0.0
    assert evaluation.total.blend_weight == 0.0
    assert not evaluation.margin.shadow_candidate
    assert not evaluation.total.shadow_candidate
