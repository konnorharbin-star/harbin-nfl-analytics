from datetime import date

import polars as pl

from nfl.data import canonical_team_code
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
    assert abs(model.training_weight - expected_weight) < 1e-12



def test_franchise_relocation_aliases_are_canonical() -> None:
    assert canonical_team_code("OAK") == "LV"
    assert canonical_team_code("SD") == "LAC"
    assert canonical_team_code("STL") == "LA"
    assert canonical_team_code("KC") == "KC"


def test_raiders_relocation_keeps_prior_season_rating_history() -> None:
    schedules = pl.DataFrame(
        {
            "season": [2019, 2019, 2020],
            "week": [16, 17, 1],
            "game_id": ["2019-a", "2019-b", "2020-target"],
            "game_type": ["REG", "REG", "REG"],
            "gameday": [
                date(2019, 12, 22),
                date(2019, 12, 29),
                date(2020, 9, 13),
            ],
            "away_team": ["KC", "OAK", "KC"],
            "home_team": ["OAK", "KC", "LV"],
            "away_score": [24, 21, None],
            "home_score": [27, 28, None],
        }
    )

    model = fit_pregame_fair_score(schedules, 2020, 1, ridge=4.0)
    projection = model.project("LV", "KC")

    assert "LV" in model.teams
    assert "OAK" not in model.teams
    assert projection.home_points >= 0
    assert projection.away_points >= 0
