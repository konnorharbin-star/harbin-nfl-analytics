from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from nfl.recent_form_current import build_current_recent_form_signals


def _schedules() -> pl.DataFrame:
    games = [
        (1, "g1", "A", "B", 20.0, 24.0),
        (1, "g2", "C", "D", 17.0, 21.0),
        (2, "g3", "B", "C", 21.0, 20.0),
        (2, "g4", "D", "A", 14.0, 28.0),
        (3, "g5", "C", "A", None, None),
        (3, "g6", "B", "D", None, None),
    ]
    start = date(2026, 9, 6)
    return pl.DataFrame(
        {
            "season": [2026] * len(games),
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


def _pbp(target_week_epa: float) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    play_id = 0
    for game in _schedules().iter_rows(named=True):
        week = int(game["week"])
        away = str(game["away_team"])
        home = str(game["home_team"])
        for offense, defense, base in ((away, home, -0.1), (home, away, 0.3)):
            for is_pass, delta, yards in ((1, 0.2, 21), (0, -0.05, 12)):
                play_id += 1
                epa = base + delta + (target_week_epa if week == 3 else 0.0)
                rows.append(
                    {
                        "season": 2026,
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


def test_current_recent_form_ignores_target_week_pbp() -> None:
    low = build_current_recent_form_signals(_schedules(), _pbp(-100.0), 2026, 3)
    high = build_current_recent_form_signals(_schedules(), _pbp(100.0), 2026, 3)

    assert low.equals(high)
    assert low.height == 2
    assert "recent_epa_per_play_total_signal" in low.columns
    assert "recent_success_rate_total_signal" in low.columns
