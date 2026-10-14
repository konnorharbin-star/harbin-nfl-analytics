from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from nfl.overtime_research import Possession as P
from nfl.overtime_research import append_overtime, resolve_overtime
from nfl.overtime_source import regulation_history, score_challenger


@pytest.mark.parametrize(
    "events,expected",
    [
        ([P("touchdown", 180, 1), P("empty", 120)], (7, 0)),
        ([P("touchdown", 180, 1), P("touchdown", 120, 2)], (7, 8)),
        ([P("touchdown", 180, 1), P("touchdown", 120, 1), P("field_goal", 60)], (10, 7)),
        ([P("field_goal", 180), P("touchdown", 120, 1)], (3, 6)),
        ([P("empty", 180), P("touchdown", 120, 1)], (0, 6)),
        ([P("empty", 180), P("empty", 120), P("touchdown", 60, 2)], (6, 0)),
        ([P("field_goal", 400), P("touchdown", 201, 1)], (3, 0)),
        ([P("empty", 600)], (0, 0)),
        ([P("touchdown", 600, 1)], (6, 0)),
        ([P("touchdown", 300, 1), P("touchdown", 300, 1)], (7, 7)),
        ([P("safety", 50)], (0, 2)),
        ([P("defensive_td", 50)], (0, 6)),
        ([P("field_goal", 200), P("defensive_td", 100)], (9, 0)),
    ],
)
def test_regular_season_resolution(events, expected):
    result = resolve_overtime(events)
    assert (result["home_points"], result["away_points"]) == expected
    reversed_result = resolve_overtime(events, first_team=1)
    assert (reversed_result["away_points"], reversed_result["home_points"]) == expected


def test_opening_td_is_not_sudden_death_and_missing_response_is_not_tie():
    with pytest.raises(ValueError, match="incomplete"):
        resolve_overtime([P("touchdown", 100, 1)])


@pytest.mark.parametrize(
    "args", [("invalid", 10), ("empty", 0), ("empty", 10, 1), ("touchdown", 10, 3), ("empty", 1.5)]
)
def test_event_validation(args):
    with pytest.raises(ValueError):
        P(*args)


def test_overtime_only_changes_tied_draws_and_mass_reconciles():
    scores = np.array([[21, 20], [10, 10], [0, 0], [18, 22]])
    original = scores.copy()
    result = append_overtime(scores, [P("empty", 600)], score_phase="VERIFIED_REGULATION_ONLY")
    np.testing.assert_array_equal(scores, original)
    assert result["regulation"] == result["with_overtime"]
    assert result["overtime_draws"] == 2
    result = append_overtime(scores, [P("field_goal", 200)], score_phase="VERIFIED_REGULATION_ONLY")
    assert result["with_overtime"]["tie"] == 0
    assert sum(result["with_overtime"][k] for k in ("home_win", "away_win", "tie")) == 1
    assert result["with_overtime"]["total"] > result["regulation"]["total"]
    assert result["used_for_bet_ranking"] is False
    assert result["player_props_updated"] is False
    assert result == append_overtime(
        scores, [P("field_goal", 200)], score_phase="VERIFIED_REGULATION_ONLY"
    )


def test_full_game_input_cannot_receive_overtime_twice():
    with pytest.raises(ValueError, match="regulation-only"):
        append_overtime([[21, 21]], [P("field_goal", 100)], score_phase="FULL_GAME")


@pytest.mark.parametrize("scores", [[], [[1]], [[-1, 2]], [[1.5, 2]], [[float("nan"), 2]]])
def test_bad_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        append_overtime(scores, [P("empty", 600)], score_phase="VERIFIED_REGULATION_ONLY")


def history_fixture():
    schedules, plays = [], []
    for index in range(45):
        gid = f"2025_{index:02}_A_B"
        schedules.append(
            {
                "game_id": gid,
                "season": 2025,
                "week": index % 18 + 1,
                "game_type": "REG",
                "gameday": "2025-09-01",
                "gametime": "12:00",
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 0,
            }
        )
        for drive in range(3):
            plays.append(
                {
                    "game_id": gid,
                    "play_id": drive,
                    "qtr": 4,
                    "game_seconds_remaining": 0 if drive == 2 else 60,
                    "fixed_drive": drive,
                    "posteam": "A",
                    "posteam_score": 0 if drive == 0 else 3,
                    "defteam_score": 0,
                    "posteam_score_post": 3,
                    "defteam_score_post": 0,
                    "total_home_score": 3,
                    "total_away_score": 0,
                    "drive_time_of_possession": "1:00",
                    "drive_quarter_end": 4,
                }
            )
    return schedules, plays


def test_source_excludes_ot_future_and_immature_games_without_zero_fallback():
    schedules, plays = history_fixture()
    schedules[0].update(home_score=6, away_score=3)
    for row in plays:
        if row["game_id"] == schedules[0]["game_id"]:
            row.update(total_home_score=3, total_away_score=3)
    plays.append(
        {**plays[0], "play_id": 999, "qtr": 5, "total_home_score": 6, "total_away_score": 3}
    )
    at = datetime(2026, 10, 14, tzinfo=UTC)
    schedules.append({**schedules[1], "game_id": "future", "gameday": "2026-10-15"})
    schedules.append({**schedules[1], "game_id": "immature", "gameday": "2026-10-13"})
    plays.extend([{**plays[3], "game_id": gid} for gid in ("future", "immature")])
    games, kernel, audit = regulation_history(
        pl.DataFrame(schedules), pl.DataFrame(plays), at=at, season=2026
    )
    assert len(games) == 45
    assert games[0]["home_score"] == games[0]["away_score"] == 3
    assert audit["overtime_rows_excluded"] == 1
    assert audit["regulation_tied_games"] == 1
    assert len(kernel) == 135
    assert all(g["game_id"] not in ("future", "immature") for g in games)
    assert len(audit["input_sha256"]) == 64


def test_missing_end_is_excluded_and_challenger_keeps_market_gate_closed():
    schedules, plays = history_fixture()
    plays = [p for p in plays if p["game_id"] != schedules[0]["game_id"]]
    at = datetime(2026, 10, 14, tzinfo=UTC)
    games, kernel, audit = regulation_history(
        pl.DataFrame(schedules), pl.DataFrame(plays), at=at, season=2026
    )
    assert len(games) == 44
    assert audit["missing_regulation_end_game"] == 1
    target = {**schedules[0], "game_id": "target", "gameday": "2026-10-15"}
    report = score_challenger(games, kernel, target, draws=1000)
    assert report["promotion_earned"] is False
    assert report["market_accuracy_proven"] is False
    assert report["with_overtime"]["total"] >= report["regulation"]["total"]
    with pytest.raises(ValueError, match="mature"):
        score_challenger([{**games[0], "gameday": "2026-10-14"}], kernel, target)
