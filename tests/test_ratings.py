from datetime import date

import polars as pl

from nfl.ratings import (
    VALIDATED_PRIOR_SEASON_WEIGHT,
    FairScoreModel,
    fair_score_history,
    fit_pregame_fair_score,
)


def _balanced_team_games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "points_for": [24, 21, 21, 24, 24, 21, 21, 24],
            "is_home": [True, False, False, True, True, False, False, True],
        }
    )


def test_balanced_teams_learn_home_field() -> None:
    model = FairScoreModel(ridge=4.0).fit(_balanced_team_games())
    projection = model.project("A", "B")

    assert model.home_field is not None
    assert 2.0 < model.home_field < 4.0
    assert projection.home_points > projection.away_points
    assert 2.0 < projection.home_margin < 4.0


def test_stronger_offense_projects_more_points() -> None:
    games = pl.DataFrame(
        {
            "team": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "points_for": [35, 14, 31, 17, 38, 13, 34, 16],
            "is_home": [True, False, False, True, True, False, False, True],
        }
    )
    model = FairScoreModel(ridge=4.0).fit(games)
    projection = model.project("A", "B")

    assert model.offense["A"] > model.offense["B"]
    assert projection.home_points > projection.away_points


def _schedule(week_three_home_score: int) -> pl.DataFrame:
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
            "home_score": [24, 21, week_three_home_score],
        }
    )


def test_target_week_result_cannot_change_pregame_model() -> None:
    model_one = fit_pregame_fair_score(_schedule(7), 2025, 3, ridge=4.0)
    model_two = fit_pregame_fair_score(_schedule(70), 2025, 3, ridge=4.0)

    projection_one = model_one.project("B", "A")
    projection_two = model_two.project("B", "A")

    assert projection_one.home_points == projection_two.home_points
    assert projection_one.away_points == projection_two.away_points


def test_canonical_history_uses_only_immediate_prior_regular_season() -> None:
    schedules = pl.DataFrame(
        {
            "season": [2023, 2024, 2024, 2025, 2025],
            "week": [18, 18, 19, 1, 2],
            "game_id": ["old", "prior", "prior_post", "current", "target"],
            "game_type": ["REG", "REG", "POST", "REG", "REG"],
            "gameday": [
                date(2024, 1, 7),
                date(2025, 1, 5),
                date(2025, 1, 12),
                date(2025, 9, 7),
                date(2025, 9, 14),
            ],
            "away_team": ["A", "A", "A", "A", "A"],
            "home_team": ["B", "B", "B", "B", "B"],
            "away_score": [17, 20, 24, 21, 10],
            "home_score": [24, 27, 28, 24, 35],
        }
    )

    history = fair_score_history(schedules, 2025, 2)

    assert history.get_column("game_id").to_list() == ["prior", "current"]


def test_default_prior_weight_is_applied_to_prior_team_rows() -> None:
    schedules = pl.DataFrame(
        {
            "season": [2024, 2024, 2025],
            "week": [17, 18, 1],
            "game_id": ["p1", "p2", "c1"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [date(2024, 12, 29), date(2025, 1, 5), date(2025, 9, 7)],
            "away_team": ["A", "B", "A"],
            "home_team": ["B", "A", "B"],
            "away_score": [17, 20, 21],
            "home_score": [24, 27, 24],
        }
    )

    model = fit_pregame_fair_score(schedules, 2025, 2, ridge=4.0)
    expected_weight = 2.0 + (4.0 * VALIDATED_PRIOR_SEASON_WEIGHT)

    assert model.training_rows == 6
    assert model.training_weight == expected_weight
