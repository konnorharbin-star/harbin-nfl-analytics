from datetime import date

import numpy as np
import polars as pl

from nfl.data import schedule_to_team_games
from nfl.ratings import FairScoreModel, pregame_sample_weights
from nfl.recency import build_week_score_predictions


def test_recency_weights_anchor_latest_eligible_week() -> None:
    team_games = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
        }
    )
    weights = pregame_sample_weights(
        team_games,
        2025,
        4,
        current_season_half_life=2.0,
    )

    expected = np.asarray([0.5, np.sqrt(0.5), 1.0])
    assert np.allclose(weights, expected)


def test_prior_season_rows_keep_fixed_low_weight() -> None:
    team_games = pl.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "week": [17, 18, 1, 2],
        }
    )
    weights = pregame_sample_weights(
        team_games,
        2025,
        3,
        prior_season_weight=0.1,
        current_season_half_life=2.0,
    )

    assert np.allclose(weights[:2], [0.1, 0.1])
    assert np.allclose(weights[2:], [np.sqrt(0.5), 1.0])


def test_weighted_fit_moves_toward_recent_scoring_state() -> None:
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "points_for": [10, 30, 13, 27, 35, 14, 38, 17],
            "is_home": [True, False, False, True, True, False, False, True],
        }
    )
    unweighted = FairScoreModel(ridge=4.0).fit(games)
    weighted = FairScoreModel(ridge=4.0).fit(
        games,
        sample_weight=[0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0],
    )

    assert weighted.offense["A"] > unweighted.offense["A"]
    assert weighted.offense["B"] < unweighted.offense["B"]


def _schedule(target_home_score: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025, 2025],
            "week": [17, 18, 1, 2, 3],
            "game_id": ["p1", "p2", "g1", "g2", "g3"],
            "game_type": ["REG", "REG", "REG", "REG", "REG"],
            "gameday": [
                date(2024, 12, 29),
                date(2025, 1, 5),
                date(2025, 9, 7),
                date(2025, 9, 14),
                date(2025, 9, 21),
            ],
            "away_team": ["A", "B", "A", "B", "A"],
            "home_team": ["B", "A", "B", "A", "B"],
            "away_score": [17, 20, 20, 17, 10],
            "home_score": [24, 27, 24, 21, target_home_score],
        }
    )


def test_target_week_result_cannot_change_recency_projection() -> None:
    first = build_week_score_predictions(
        _schedule(7),
        2025,
        3,
        ridge=4.0,
        half_life_weeks=2.0,
    )
    second = build_week_score_predictions(
        _schedule(70),
        2025,
        3,
        ridge=4.0,
        half_life_weeks=2.0,
    )

    assert first.get_column("projected_home_margin").to_list() == second.get_column(
        "projected_home_margin"
    ).to_list()
    assert first.get_column("projected_total").to_list() == second.get_column(
        "projected_total"
    ).to_list()


def test_target_week_rows_are_rejected_by_weight_builder() -> None:
    schedules = _schedule(7)
    team_games = schedule_to_team_games(schedules.filter(pl.col("season") == 2025))

    with np.testing.assert_raises_regex(Exception, "outside the canonical pregame history"):
        pregame_sample_weights(
            team_games,
            2025,
            3,
            current_season_half_life=2.0,
        )
