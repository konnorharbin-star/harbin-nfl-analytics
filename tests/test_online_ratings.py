from __future__ import annotations

import polars as pl

from nfl.online_ratings import (
    OnlineConfig,
    build_online_predictions,
    evaluate_online_ratings,
)


def _schedules() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2021, 2022, 2023):
        for week in range(1, 7):
            rows.extend(
                [
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
                    },
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
                    },
                ]
            )
    return pl.DataFrame(rows)


def test_target_week_results_do_not_change_online_pregame_predictions() -> None:
    schedules = _schedules()
    config = OnlineConfig(0.08, 0.60, 1.75)
    before = build_online_predictions(
        schedules,
        config,
        test_seasons=(2023,),
        warmup_start=2021,
        start_week=5,
        end_week=5,
    )

    mutated = schedules.with_columns(
        pl.when((pl.col("season") == 2023) & (pl.col("week") == 5))
        .then(pl.lit(99))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when((pl.col("season") == 2023) & (pl.col("week") == 5))
        .then(pl.lit(0))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )
    after = build_online_predictions(
        mutated,
        config,
        test_seasons=(2023,),
        warmup_start=2021,
        start_week=5,
        end_week=5,
    )

    assert before.equals(after)


def _baseline() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2023, 2024, 2025):
        rows.extend(
            [
                {
                    "season": season,
                    "week": 5,
                    "game_id": f"{season}-a",
                    "baseline_home_margin": 2.0,
                    "baseline_total": 42.0,
                    "actual_home_margin": 4.0,
                    "actual_total": 44.0,
                },
                {
                    "season": season,
                    "week": 5,
                    "game_id": f"{season}-b",
                    "baseline_home_margin": -4.0,
                    "baseline_total": 52.0,
                    "actual_home_margin": -6.0,
                    "actual_total": 50.0,
                },
            ]
        )
    return pl.DataFrame(rows)


def _candidate(
    config: OnlineConfig,
    *,
    margin_better: bool,
    total_better: bool,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2023, 2024, 2025):
        rows.extend(
            [
                {
                    "season": season,
                    "week": 5,
                    "game_id": f"{season}-a",
                    "online_home_margin": 3.0 if margin_better else 2.0,
                    "online_total": 43.0 if total_better else 42.0,
                },
                {
                    "season": season,
                    "week": 5,
                    "game_id": f"{season}-b",
                    "online_home_margin": -5.0 if margin_better else -4.0,
                    "online_total": 51.0 if total_better else 52.0,
                },
            ]
        )
    return pl.DataFrame(rows)


def test_online_gate_falls_back_to_canonical_when_no_config_improves(
    monkeypatch,
) -> None:
    monkeypatch.setattr("nfl.online_ratings._baseline_frame", lambda *a, **k: _baseline())
    monkeypatch.setattr(
        "nfl.online_ratings.build_online_predictions",
        lambda schedules, config, **kwargs: _candidate(
            config,
            margin_better=False,
            total_better=False,
        ),
    )

    result = evaluate_online_ratings(
        pl.DataFrame(),
        update_grid=(0.08,),
        carry_grid=(0.60,),
        home_field_grid=(1.75,),
    )

    assert result.selected_margin_config is None
    assert result.selected_total_config is None
    assert result.margin_shadow_candidate is False
    assert result.total_shadow_candidate is False


def test_margin_and_total_can_select_different_online_configs(monkeypatch) -> None:
    monkeypatch.setattr("nfl.online_ratings._baseline_frame", lambda *a, **k: _baseline())

    def fake_predictions(_schedules, config: OnlineConfig, **_kwargs) -> pl.DataFrame:
        return _candidate(
            config,
            margin_better=config.update_alpha == 0.04,
            total_better=config.update_alpha == 0.08,
        )

    monkeypatch.setattr(
        "nfl.online_ratings.build_online_predictions",
        fake_predictions,
    )
    result = evaluate_online_ratings(
        pl.DataFrame(),
        update_grid=(0.04, 0.08),
        carry_grid=(0.60,),
        home_field_grid=(1.75,),
    )

    assert result.selected_margin_config is not None
    assert result.selected_margin_config.update_alpha == 0.04
    assert result.selected_total_config is not None
    assert result.selected_total_config.update_alpha == 0.08
    assert result.canonical_score_change_enabled is False
    assert result.promotion_eligible is False
