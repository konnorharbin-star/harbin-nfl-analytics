from datetime import date

import polars as pl

from nfl.ratings import FairScoreModel, fit_pregame_fair_score


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
