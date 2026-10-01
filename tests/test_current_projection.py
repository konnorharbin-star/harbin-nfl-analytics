from datetime import date

import polars as pl

from nfl.current import (
    build_current_qb_projection,
    next_unplayed_regular_week,
    unplayed_regular_games,
)
from nfl.qb_validated import ValidatedQBAdjustment


def _schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 4],
            "game_id": ["g1", "g2", "g3", "g4"],
            "game_type": ["REG", "REG", "REG", "REG"],
            "gameday": [
                date(2025, 9, 7),
                date(2025, 9, 14),
                date(2025, 9, 21),
                date(2025, 9, 28),
            ],
            "away_team": ["A", "B", "B", "A"],
            "home_team": ["B", "A", "A", "B"],
            "away_score": [20, 17, None, None],
            "home_score": [24, 27, None, None],
        }
    )


def _player_stats() -> pl.DataFrame:
    rows = []
    for week, game_id in ((1, "g1"), (2, "g2")):
        rows.extend(
            [
                {
                    "player_id": "A1",
                    "player_name": "A One",
                    "position": "QB",
                    "season": 2025,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "team": "A",
                    "attempts": 30,
                    "passing_interceptions": 0,
                    "sacks_suffered": 2,
                    "passing_epa": 4.0,
                    "passing_cpoe": 2.0,
                },
                {
                    "player_id": "B1",
                    "player_name": "B One",
                    "position": "QB",
                    "season": 2025,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game_id,
                    "team": "B",
                    "attempts": 30,
                    "passing_interceptions": 1,
                    "sacks_suffered": 3,
                    "passing_epa": 1.0,
                    "passing_cpoe": -1.0,
                },
            ]
        )
    return pl.DataFrame(rows)


def _adjustment() -> ValidatedQBAdjustment:
    rows = []
    for index in range(120):
        epa_margin = ((index % 9) - 4) / 10.0
        epa_total = ((index % 7) - 3) / 10.0
        cpoe_total = float((index % 5) - 2)
        rows.append(
            {
                "margin_residual": 2.0 * epa_margin,
                "total_residual": (1.5 * epa_total) + (0.1 * cpoe_total),
                "qb_epa_margin_signal": epa_margin,
                "qb_epa_total_signal": epa_total,
                "qb_cpoe_total_signal": cpoe_total,
            }
        )
    return ValidatedQBAdjustment().fit(pl.DataFrame(rows))


def test_next_unplayed_week_and_target_filter() -> None:
    schedules = _schedule()

    assert next_unplayed_regular_week(schedules, 2025) == 3
    assert unplayed_regular_games(schedules, 2025, 3).get_column("game_id").to_list() == [
        "g3"
    ]


def test_current_projection_emits_canonical_and_shadow_states() -> None:
    projection = build_current_qb_projection(
        _schedule(),
        _player_stats(),
        2025,
        3,
        _adjustment(),
        score_ridge=4.0,
        qb_prior_dropbacks=40.0,
    )
    row = projection.row(0, named=True)

    assert row["game_id"] == "g3"
    assert row["baseline_release_state"] == "CANONICAL"
    assert row["qb_margin_release_state"] == "RESEARCH"
    assert row["qb_total_release_state"] == "SHADOW"
    assert row["home_qb_proxy_last_week"] == 2
    assert row["away_qb_proxy_last_week"] == 2
    assert "qb_adjusted_home_margin" in projection.columns
    assert "qb_adjusted_total" in projection.columns
