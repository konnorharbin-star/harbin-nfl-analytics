from datetime import UTC, datetime

import polars as pl
import pytest

from nfl.clean_efficiency_research import aggregate, examples, fit, predict, study


def play(**updates):
    return dict(
        game_id="g",
        play_id=1,
        posteam="A",
        defteam="B",
        epa=-1.0,
        qb_dropback=1.0,
        rush_attempt=0.0,
        qb_kneel=0.0,
        qb_spike=0.0,
        play_type="pass",
        down=1.0,
        qtr=1.0,
        half_seconds_remaining=600.0,
        score_differential=0.0,
        **updates,
    )


def fixture():
    return pl.DataFrame([dict(game_id="g", game_type="REG", home_team="A", away_team="B")])


def test_sacks_and_scrambles_count_as_dropbacks_not_designed_runs():
    sack = play()
    scramble = play()
    scramble.update(play_id=2, rush_attempt=1, play_type="run", epa=1.0)
    totals, _ = aggregate(fixture(), pl.DataFrame([sack, scramble]))
    assert totals[("g", "A", "off")]["pass_epa"] == [0.0, 2]
    assert totals[("g", "A", "off")]["rush_epa"] == [0.0, 0]
    assert totals[("g", "A", "off")]["success_rate"] == [1.0, 2]
    assert totals[("g", "B", "def")]["success_rate"] == [1.0, 2]


@pytest.mark.parametrize(
    "updates",
    [
        dict(qb_kneel=1),
        dict(qb_spike=1),
        dict(qtr=5),
        dict(play_type="no_play"),
        dict(half_seconds_remaining=120),
        dict(score_differential=15),
        dict(epa=None),
        dict(epa=float("nan")),
        dict(qb_kneel=None),
        dict(posteam="X"),
    ],
)
def test_ineligible_and_missing_plays_do_not_become_zero_features(updates):
    row = play()
    row.update(updates)
    totals, excluded = aggregate(fixture(), pl.DataFrame([row]))
    assert totals == {}
    assert sum(excluded.values()) == 1


def test_duplicate_play_fails_instead_of_double_weighting():
    with pytest.raises(ValueError, match="DUPLICATE"):
        aggregate(fixture(), pl.DataFrame([play(), play()]))


def synthetic(year, n=110):
    return [
        dict(
            season=year,
            game_id=f"{year}_{i}",
            week=5 + i % 14,
            baseline_margin=0.0,
            baseline_total=40.0,
            actual_margin=0.0,
            actual_total=40.0,
            signals={
                m + "_" + t: (i % 11 - 5) / 10
                for m in ("pass_epa", "rush_epa", "success_rate")
                for t in ("margin", "total")
            },
        )
        for i in range(n)
    ]


def test_evaluation_outcomes_cannot_select_weights_or_fit_coefficients():
    rows = synthetic(2023) + synthetic(2024) + synthetic(2025)
    _, frozen = study(rows)
    for row in rows:
        if row["season"] == 2025:
            row.update(actual_margin=99.0, actual_total=101.0)
    report, after = study(rows)
    assert frozen == after
    assert all(m["blend"] == 0 for m in frozen.values())
    assert report["validated_edges"] == 0
    assert report["used_for_bet_ranking"] is False


def test_zero_blend_preserves_exact_baseline():
    rows = synthetic(2023)
    model = fit(rows, "total", 10.0)
    assert predict(rows[0], model, 0.0) == 40.0


def test_target_and_future_play_outcomes_cannot_change_pregame_features():
    games = []
    for week in range(1, 7):
        games.append(
            dict(
                game_id=f"g{week}",
                season=2026,
                week=week,
                game_type="REG",
                home_team="A",
                away_team="B",
                home_score=20.0,
                away_score=17.0,
                gameday=f"2026-09-{week * 4:02d}",
                gametime="13:00",
            )
        )
    schedule = pl.DataFrame(games)
    totals = {
        (g["game_id"], team, role): {
            m: [10.0, 50] for m in ("pass_epa", "rush_epa", "success_rate")
        }
        for g in games
        for team in ("A", "B")
        for role in ("off", "def")
    }
    at = datetime(2026, 9, 20, 12, tzinfo=UTC)
    before, _ = examples(schedule, totals, seasons=(2026,), at=at)
    assert before
    for key in totals:
        if key[0] in ("g5", "g6"):
            totals[key] = {m: [99999.0, 50] for m in totals[key]}
    after, _ = examples(schedule, totals, seasons=(2026,), at=at)
    assert before == after
    assert before[0]["actual_margin"] is None
