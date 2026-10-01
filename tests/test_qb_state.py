from datetime import date

import polars as pl

from nfl.qb_dataset import build_qb_week_snapshot
from nfl.qb_state import primary_qb_games, team_qb_state


def _qb_row(
    *,
    week: int,
    game_id: str,
    team: str,
    player_id: str,
    attempts: int,
    sacks: int,
    epa: float,
    cpoe: float,
    interceptions: int = 0,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "player_name": player_id,
        "position": "QB",
        "season": 2025,
        "week": week,
        "season_type": "REG",
        "game_id": game_id,
        "team": team,
        "attempts": attempts,
        "passing_interceptions": interceptions,
        "sacks_suffered": sacks,
        "passing_epa": epa,
        "passing_cpoe": cpoe,
    }


def _player_stats(target_epa: float = 0.0) -> pl.DataFrame:
    rows = [
        _qb_row(
            week=1,
            game_id="g1",
            team="A",
            player_id="A1",
            attempts=30,
            sacks=2,
            epa=4.0,
            cpoe=2.0,
        ),
        _qb_row(
            week=1,
            game_id="g1",
            team="B",
            player_id="B1",
            attempts=28,
            sacks=3,
            epa=1.0,
            cpoe=-1.0,
            interceptions=1,
        ),
        _qb_row(
            week=2,
            game_id="g2",
            team="A",
            player_id="A1",
            attempts=4,
            sacks=0,
            epa=-1.0,
            cpoe=-4.0,
        ),
        _qb_row(
            week=2,
            game_id="g2",
            team="A",
            player_id="A2",
            attempts=25,
            sacks=1,
            epa=6.0,
            cpoe=5.0,
        ),
        _qb_row(
            week=2,
            game_id="g2",
            team="B",
            player_id="B1",
            attempts=31,
            sacks=2,
            epa=2.0,
            cpoe=1.0,
        ),
        _qb_row(
            week=3,
            game_id="g3",
            team="A",
            player_id="A2",
            attempts=35,
            sacks=0,
            epa=target_epa,
            cpoe=20.0,
        ),
        _qb_row(
            week=3,
            game_id="g3",
            team="B",
            player_id="B1",
            attempts=35,
            sacks=0,
            epa=-target_epa,
            cpoe=-20.0,
        ),
    ]
    return pl.DataFrame(rows)


def _schedule(target_home_score: int = 28) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_id": ["g1", "g2", "g3"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [date(2025, 9, 7), date(2025, 9, 14), date(2025, 9, 21)],
            "away_team": ["A", "B", "B"],
            "home_team": ["B", "A", "A"],
            "away_score": [20, 17, 17],
            "home_score": [24, 27, target_home_score],
        }
    )


def test_primary_qb_uses_largest_dropback_role() -> None:
    history = _player_stats().filter(pl.col("week") < 3)
    primaries = primary_qb_games(history)
    a_week_two = primaries.filter(
        (pl.col("team") == "A") & (pl.col("week") == 2)
    )

    assert a_week_two.get_column("player_id").to_list() == ["A2"]


def test_team_qb_state_detects_recent_change_without_target_game() -> None:
    state = team_qb_state(_player_stats(target_epa=100.0), 2025, 3, prior_dropbacks=40.0)
    a = state.filter(pl.col("team") == "A").row(0, named=True)
    b = state.filter(pl.col("team") == "B").row(0, named=True)

    assert a["qb_proxy_id"] == "A2"
    assert a["qb_changed_last_observation"] == 1
    assert a["qb_consecutive_starts"] == 1
    assert b["qb_proxy_id"] == "B1"
    assert b["qb_changed_last_observation"] == 0
    assert b["qb_consecutive_starts"] == 2


def test_target_week_qb_stats_and_score_cannot_change_pregame_predictors() -> None:
    first = build_qb_week_snapshot(
        _schedule(10),
        _player_stats(target_epa=-100.0),
        2025,
        3,
        score_ridge=4.0,
        qb_prior_dropbacks=40.0,
    )
    second = build_qb_week_snapshot(
        _schedule(70),
        _player_stats(target_epa=100.0),
        2025,
        3,
        score_ridge=4.0,
        qb_prior_dropbacks=40.0,
    )

    predictor_columns = [
        name
        for name in first.columns
        if name.startswith("baseline_")
        or name.startswith("qb_")
        or name.endswith("_qb_proxy_id")
        or name.endswith("_qb_proxy_last_week")
    ]
    assert first.select(predictor_columns).to_dicts() == second.select(
        predictor_columns
    ).to_dicts()
