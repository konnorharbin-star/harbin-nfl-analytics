from datetime import date

import numpy as np
import polars as pl

from nfl.priors import build_week_prior_predictions, prior_season_weights


def test_prior_season_weights_keep_current_games_full_strength() -> None:
    team_games = pl.DataFrame({"season": [2024, 2024, 2025, 2025]})
    weights = prior_season_weights(team_games, season=2025, prior_weight=0.2)

    assert np.allclose(weights, [0.2, 0.2, 1.0, 1.0])


def _schedule(target_home_score: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "week": [17, 18, 1, 2],
            "game_id": ["p1", "p2", "c1", "target"],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gameday": [
                date(2024, 12, 29),
                date(2025, 1, 5),
                date(2025, 9, 7),
                date(2025, 9, 14),
            ],
            "away_team": ["A", "B", "A", "B"],
            "home_team": ["B", "A", "B", "A"],
            "away_score": [17, 20, 21, 10],
            "home_score": [24, 27, 24, target_home_score],
        }
    )


def test_target_week_result_cannot_change_prior_projection() -> None:
    first = build_week_prior_predictions(
        _schedule(7),
        2025,
        2,
        prior_weight=0.2,
        ridge=4.0,
    )
    second = build_week_prior_predictions(
        _schedule(70),
        2025,
        2,
        prior_weight=0.2,
        ridge=4.0,
    )

    assert first.get_column("projected_home_margin").to_list() == second.get_column(
        "projected_home_margin"
    ).to_list()
    assert first.get_column("projected_total").to_list() == second.get_column(
        "projected_total"
    ).to_list()
