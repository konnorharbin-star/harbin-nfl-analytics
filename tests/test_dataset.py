from datetime import date, timedelta

import polars as pl

from nfl.dataset import build_walkforward_dataset, build_week_snapshot


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


def _pbp(target_week_epa: float = 0.5) -> pl.DataFrame:
    schedules = _schedules()
    rows: list[dict[str, object]] = []
    play_id = 0
    for game in schedules.iter_rows(named=True):
        week = int(game["week"])
        away = str(game["away_team"])
        home = str(game["home_team"])
        for offense, defense, base_epa in (
            (away, home, -0.1),
            (home, away, 0.2),
        ):
            for is_pass, yards, delta in ((1, 22, 0.3), (0, 11, -0.2)):
                play_id += 1
                epa = base_epa + delta
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
                        "play_type": "pass" if is_pass else "run",
                        "epa": epa,
                        "success": 1 if epa > 0 else 0,
                        "pass_attempt": is_pass,
                        "rush_attempt": 1 - is_pass,
                        "qb_dropback": is_pass,
                        "yards_gained": yards,
                        "down": 1 if is_pass else 2,
                    }
                )
    return pl.DataFrame(rows)


def test_week_snapshot_contains_only_pregame_predictors() -> None:
    snapshot = build_week_snapshot(_schedules(), _pbp(), 2025, 3, ridge=4.0)

    assert snapshot.height == 2
    assert set(snapshot.get_column("game_id")) == {"g5", "g6"}
    assert "epa_per_play_matchup_advantage" in snapshot.columns
    assert "baseline_home_margin" in snapshot.columns
    assert "margin_residual" in snapshot.columns


def test_target_week_pbp_cannot_change_snapshot() -> None:
    low = build_week_snapshot(_schedules(), _pbp(target_week_epa=-50.0), 2025, 3, ridge=4.0)
    high = build_week_snapshot(_schedules(), _pbp(target_week_epa=50.0), 2025, 3, ridge=4.0)

    assert low.equals(high)


def test_walkforward_dataset_reconstructs_each_week() -> None:
    dataset = build_walkforward_dataset(
        _schedules(),
        _pbp(),
        2025,
        start_week=3,
        end_week=5,
        ridge=4.0,
    )

    assert dataset.height == 6
    assert dataset.get_column("week").to_list() == [3, 3, 4, 4, 5, 5]
    assert dataset.get_column("game_id").n_unique() == 6
