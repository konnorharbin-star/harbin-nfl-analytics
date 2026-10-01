from datetime import date

import polars as pl

from nfl.oa_dataset import build_oa_week_snapshot
from nfl.opponent_adjusted import OpponentAdjustedPBPModel, team_game_pbp_metrics


def _play_rows(game_id: str, week: int, a_epa: float, b_epa: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    play_id = 0
    for offense, defense, epa in (("A", "B", a_epa), ("B", "A", b_epa)):
        rows.extend(
            [
                {
                    "season": 2025,
                    "week": week,
                    "game_id": game_id,
                    "play_id": play_id,
                    "posteam": offense,
                    "defteam": defense,
                    "epa": epa,
                    "success": 1 if epa > 0 else 0,
                    "pass_attempt": 1,
                    "rush_attempt": 0,
                    "qb_dropback": 1,
                    "yards_gained": 22 if epa > 0 else 2,
                    "down": 1,
                },
                {
                    "season": 2025,
                    "week": week,
                    "game_id": game_id,
                    "play_id": play_id + 1,
                    "posteam": offense,
                    "defteam": defense,
                    "epa": epa / 2.0,
                    "success": 1 if epa > 0 else 0,
                    "pass_attempt": 0,
                    "rush_attempt": 1,
                    "qb_dropback": 0,
                    "yards_gained": 12 if epa > 0 else 2,
                    "down": 2,
                },
            ]
        )
        play_id += 2
    return rows


def _pbp(target_a_epa: float = 0.0) -> pl.DataFrame:
    rows = []
    rows.extend(_play_rows("g1", 1, 0.8, -0.4))
    rows.extend(_play_rows("g2", 2, 0.6, -0.2))
    rows.extend(_play_rows("g3", 3, target_a_epa, -target_a_epa))
    return pl.DataFrame(rows)


def _schedule(target_home_score: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_id": ["g1", "g2", "g3"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [date(2025, 9, 7), date(2025, 9, 14), date(2025, 9, 21)],
            "away_team": ["A", "B", "B"],
            "home_team": ["B", "A", "A"],
            "away_score": [20, 14, 17],
            "home_score": [24, 31, target_home_score],
        }
    )


def test_team_game_metrics_create_two_observations_per_game() -> None:
    metrics = team_game_pbp_metrics(_pbp().filter(pl.col("week") <= 2))

    assert metrics.height == 4
    assert set(metrics.get_column("offense").to_list()) == {"A", "B"}
    assert metrics.get_column("pass_plays").min() == 1
    assert metrics.get_column("rush_plays").min() == 1


def test_opponent_adjusted_state_favors_stronger_offense() -> None:
    model = OpponentAdjustedPBPModel(ridge=4.0).fit(
        _pbp().filter(pl.col("week") <= 2)
    )
    signals = model.matchup_signals("A", "B")

    assert signals["oa_epa_per_play_margin_signal"] > 0
    assert signals["oa_success_rate_margin_signal"] > 0


def test_target_week_pbp_cannot_change_oa_pregame_features() -> None:
    first = build_oa_week_snapshot(
        _schedule(30),
        _pbp(target_a_epa=-5.0),
        2025,
        3,
        score_ridge=4.0,
        pbp_ridge=4.0,
    )
    second = build_oa_week_snapshot(
        _schedule(70),
        _pbp(target_a_epa=5.0),
        2025,
        3,
        score_ridge=4.0,
        pbp_ridge=4.0,
    )

    predictor_columns = [
        name
        for name in first.columns
        if name.startswith("baseline_") or name.startswith("oa_")
    ]
    assert first.select(predictor_columns).to_dicts() == second.select(
        predictor_columns
    ).to_dicts()
