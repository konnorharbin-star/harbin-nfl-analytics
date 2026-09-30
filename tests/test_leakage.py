from datetime import date

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.data import (
    assert_strictly_pregame,
    completed_games,
    pregame_history,
    schedule_to_team_games,
)


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_id": ["2025_01_A_B", "2025_02_C_D", "2025_03_E_F"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [date(2025, 9, 7), date(2025, 9, 14), date(2025, 9, 21)],
            "away_team": ["A", "C", "E"],
            "home_team": ["B", "D", "F"],
            "away_score": [20, 17, None],
            "home_score": [24, 14, None],
        }
    )


def test_completed_games_excludes_unplayed_rows() -> None:
    out = completed_games(_schedule())
    assert out.get_column("week").to_list() == [1, 2]


def test_completed_games_respects_as_of_date() -> None:
    out = completed_games(_schedule(), as_of="2025-09-10")
    assert out.get_column("week").to_list() == [1]


def test_pregame_history_excludes_target_week() -> None:
    history = pregame_history(_schedule(), 2025, 2)
    assert history.get_column("week").to_list() == [1]
    assert_strictly_pregame(history, 2025, 2)


def test_leakage_assertion_fails_on_target_week() -> None:
    with pytest.raises(DataContractError, match="target/future"):
        assert_strictly_pregame(completed_games(_schedule()), 2025, 2)


def test_team_game_expansion_is_two_rows_per_completed_game() -> None:
    team_games = schedule_to_team_games(_schedule())
    assert team_games.height == 4
    assert set(team_games.get_column("team")) == {"A", "B", "C", "D"}
    margins = dict(zip(team_games.get_column("team"), team_games.get_column("point_margin")))
    assert margins["A"] == -4.0
    assert margins["B"] == 4.0
