import polars as pl
import pytest

from nfl.advanced import (
    assert_pbp_strictly_pregame,
    pregame_pbp,
    pregame_team_pbp_features,
    team_pbp_features,
)
from nfl.contracts import DataContractError


def _pbp(target_week_epa: float = 9.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025] * 7,
            "week": [1, 1, 1, 1, 2, 2, 3],
            "game_id": ["g1", "g1", "g1", "g1", "g2", "g2", "g3"],
            "posteam": ["A", "A", "B", "B", "A", "B", "A"],
            "defteam": ["B", "B", "A", "A", "B", "A", "B"],
            "epa": [1.0, -0.5, 0.2, -0.2, 2.0, -1.0, target_week_epa],
            "success": [1, 0, 1, 0, 1, 0, 1],
            "pass_attempt": [1, 0, 1, 0, 1, 0, 1],
            "rush_attempt": [0, 1, 0, 1, 0, 1, 0],
            "qb_dropback": [1, 0, 1, 0, 1, 0, 1],
            "yards_gained": [25, 4, 8, 12, 21, 3, 80],
            "down": [1, 2, 1, 2, 3, 1, 1],
        }
    )


def test_pregame_pbp_excludes_target_week() -> None:
    history = pregame_pbp(_pbp(), 2025, 3)
    assert history.get_column("week").max() == 2
    assert_pbp_strictly_pregame(history, 2025, 3)


def test_pbp_leakage_assertion_fails_closed() -> None:
    with pytest.raises(DataContractError, match="target/future"):
        assert_pbp_strictly_pregame(_pbp(), 2025, 3)


def test_team_features_calculate_offense_and_defense() -> None:
    features = team_pbp_features(pregame_pbp(_pbp(), 2025, 3))
    a = features.filter(pl.col("team") == "A").row(0, named=True)

    assert a["off_plays"] == 3
    assert a["def_plays"] == 3
    assert a["off_epa_per_play"] == pytest.approx((1.0 - 0.5 + 2.0) / 3)
    assert a["off_success_rate"] == pytest.approx(2 / 3)
    assert a["off_explosive_rate"] == pytest.approx(2 / 3)
    assert a["off_pass_epa_per_dropback"] == pytest.approx(1.5)
    assert a["off_rush_epa_per_attempt"] == pytest.approx(-0.5)


def test_target_week_plays_cannot_change_pregame_features() -> None:
    low = pregame_team_pbp_features(_pbp(target_week_epa=-20.0), 2025, 3)
    high = pregame_team_pbp_features(_pbp(target_week_epa=20.0), 2025, 3)

    assert low.equals(high)
