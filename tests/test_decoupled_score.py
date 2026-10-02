from __future__ import annotations

import polars as pl

from nfl.decoupled_score import (
    build_week_decoupled_predictions,
    decoupled_sample_weights,
    evaluate_decoupled_architecture,
)


def _schedules() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    game = 0
    for season in (2022, 2023):
        for week in range(1, 6):
            game += 1
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week}_A_B",
                    "game_type": "REG",
                    "gameday": f"{season}-09-{week + 1:02d}",
                    "away_team": "B",
                    "home_team": "A",
                    "away_score": 17 + (week % 2),
                    "home_score": 24 + (week % 3),
                }
            )
            game += 1
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week}_D_C",
                    "game_type": "REG",
                    "gameday": f"{season}-09-{week + 1:02d}",
                    "away_team": "D",
                    "home_team": "C",
                    "away_score": 20 + (week % 3),
                    "home_score": 21 + (week % 2),
                }
            )
    return pl.DataFrame(rows)


def test_prior_season_games_receive_validated_low_weight() -> None:
    history = _schedules().filter(
        (pl.col("season") == 2022)
        | ((pl.col("season") == 2023) & (pl.col("week") < 5))
    )
    weights = decoupled_sample_weights(history, season=2023)
    seasons = history.get_column("season").to_list()

    assert all(
        weight == 0.10 if season == 2022 else weight == 1.0
        for season, weight in zip(seasons, weights, strict=True)
    )


def test_target_week_score_mutation_cannot_change_pregame_projection() -> None:
    schedules = _schedules()
    before = build_week_decoupled_predictions(
        schedules,
        2023,
        5,
        margin_ridge=8.0,
        total_ridge=8.0,
    ).select(
        "game_id",
        "baseline_home_margin",
        "baseline_total",
        "decoupled_home_margin",
        "decoupled_total",
    )

    mutated = schedules.with_columns(
        pl.when((pl.col("season") == 2023) & (pl.col("week") == 5))
        .then(pl.lit(99))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when((pl.col("season") == 2023) & (pl.col("week") == 5))
        .then(pl.lit(1))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )
    after = build_week_decoupled_predictions(
        mutated,
        2023,
        5,
        margin_ridge=8.0,
        total_ridge=8.0,
    ).select(before.columns)

    assert before.equals(after)


def _fake_frame(
    season: int,
    *,
    margin_better: bool,
    total_better: bool,
) -> pl.DataFrame:
    actual_margin = [4.0, -6.0]
    actual_total = [44.0, 50.0]
    baseline_margin = [2.0, -4.0]
    baseline_total = [42.0, 52.0]
    candidate_margin = [3.0, -5.0] if margin_better else baseline_margin
    candidate_total = [43.0, 51.0] if total_better else baseline_total
    return pl.DataFrame(
        {
            "season": [season, season],
            "week": [5, 5],
            "game_id": [f"{season}-a", f"{season}-b"],
            "actual_home_margin": actual_margin,
            "actual_total": actual_total,
            "baseline_home_margin": baseline_margin,
            "baseline_total": baseline_total,
            "decoupled_home_margin": candidate_margin,
            "decoupled_total": candidate_total,
        }
    )


def test_architecture_gate_falls_back_to_zero_when_no_ridge_improves(
    monkeypatch,
) -> None:
    def fake_walkforward(_schedules, season: int, **_kwargs) -> pl.DataFrame:
        return _fake_frame(season, margin_better=False, total_better=False)

    monkeypatch.setattr(
        "nfl.decoupled_score.build_decoupled_walkforward",
        fake_walkforward,
    )
    result = evaluate_decoupled_architecture(
        pl.DataFrame(),
        ridge_grid=(2.0, 8.0, 32.0),
    )

    assert result.selected_margin_ridge is None
    assert result.selected_total_ridge is None
    assert result.margin_shadow_candidate is False
    assert result.total_shadow_candidate is False
    assert result.margin_selected_mae == result.margin_baseline_mae
    assert result.total_selected_rmse == result.total_baseline_rmse


def test_margin_and_total_can_select_different_fixed_ridges(monkeypatch) -> None:
    def fake_walkforward(
        _schedules,
        season: int,
        *,
        margin_ridge: float,
        total_ridge: float,
        **_kwargs,
    ) -> pl.DataFrame:
        return _fake_frame(
            season,
            margin_better=margin_ridge == 2.0,
            total_better=total_ridge == 32.0,
        )

    monkeypatch.setattr(
        "nfl.decoupled_score.build_decoupled_walkforward",
        fake_walkforward,
    )
    result = evaluate_decoupled_architecture(
        pl.DataFrame(),
        ridge_grid=(2.0, 8.0, 32.0),
    )

    assert result.selected_margin_ridge == 2.0
    assert result.selected_total_ridge == 32.0
    assert result.margin_shadow_candidate is True
    assert result.total_shadow_candidate is True
    assert result.canonical_score_change_enabled is False
    assert result.promotion_eligible is False
