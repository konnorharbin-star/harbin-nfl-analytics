from datetime import date

import numpy as np
import polars as pl

from nfl.ratings import FairScoreModel
from nfl.recency import build_week_score_predictions, exponential_week_weights


def test_exponential_weights_anchor_latest_eligible_week() -> None:
    team_games = pl.DataFrame({"week": [1, 2, 3]})
    weights = exponential_week_weights(
        team_games,
        target_week=4,
        half_life_weeks=2.0,
    )

    expected = np.asarray([0.5, np.sqrt(0.5), 1.0])
    assert np.allclose(weights, expected)


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
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "game_id": ["g1", "g2", "g3"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [date(2025, 9, 7), date(2025, 9, 14), date(2025, 9, 21)],
            "away_team": ["A", "B", "A"],
            "home_team": ["B", "A", "B"],
            "away_score": [20, 17, 10],
            "home_score": [24, 21, target_home_score],
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
