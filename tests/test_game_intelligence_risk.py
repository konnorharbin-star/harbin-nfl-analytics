"""QB/OL forensics: cutoff, provenance, missingness and outcome isolation."""
from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_risk import (
    assemble_pregame_risk,
    attach_postgame_turnovers,
    summarize_risk_attribution,
)


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    base_rows = []
    qb_rows = []
    ol_rows = []
    for number in range(80):
        season = 2024 if number < 40 else 2025
        key = {"season": season, "week": 6, "game_id": f"g{number}"}
        base_rows.append({
            **key, "home_team": "A", "away_team": "B",
            "baseline_home_margin": 3.0, "baseline_total": 45.0,
            "margin_residual": 16.0 if number % 2 == 0 else -2.0,
            "total_residual": 19.0 if number % 3 == 0 else 1.0,
        })
        qb_rows.append({
            **key, "home_qb_proxy_last_week": 5,
            "away_qb_proxy_last_week": 4,
            "qb_change_total_signal": -1.0 if number % 2 == 0 else 0.0,
            "qb_sack_rate_total_signal": -0.03 if number % 2 == 0 else 0.0,
            "actual_home_margin": 999.0,
        })
        ol_rows.append({
            **key, "home_ol_injury_risk": 0.4 if number % 2 == 0 else 0.0,
            "away_ol_injury_risk": 0.0,
            "home_depth_temporal_mode": "weekly",
            "away_depth_temporal_mode": "weekly",
        })
    return (
        pl.DataFrame(base_rows), pl.DataFrame(qb_rows),
        pl.DataFrame(ol_rows),
    )


def test_provenance_qualified_risks_never_copy_qb_postgame_outcomes():
    base, qb, ol = _inputs()
    frames = assemble_pregame_risk(base, qb, ol)
    assert frames.height == 80
    assert "actual_home_margin" not in frames.columns
    assert frames["last_observed_qb_switch"].sum() == 40
    assert frames["historical_qb_sack_exposure"].sum() == 40
    assert frames["pregame_ol_injury_stress"].sum() == 40
    assert frames["ol_provenance"].unique().to_list() == [
        "WEEKLY_UNTIMESTAMPED"
    ]
    frames, status = attach_postgame_turnovers(frames, pl.DataFrame())
    assert status == "MISSING_TURNOVER_PLAY_FIELDS"
    report = summarize_risk_attribution(frames, turnover_status=status)
    qb_rates = report["targets"]["margin"]["risk_cohorts"][
        "qb_last_observed_switch"
    ]
    assert qb_rates["risk_present"]["extreme_error_rate"] == 1.0
    assert qb_rates["risk_absent"]["extreme_error_rate"] == 0.0
    assert report["staking_authorized"] is False


def test_missing_ol_does_not_mean_healthy_and_season_only_is_invalid():
    base, qb, ol = _inputs()
    missing = assemble_pregame_risk(base, qb, None)
    assert missing["pregame_ol_injury_stress"].null_count() == 80
    old = ol.with_columns(
        pl.lit("season_only").alias("home_depth_temporal_mode")
    )
    invalid = assemble_pregame_risk(base, qb, old)
    assert invalid["pregame_ol_injury_stress"].null_count() == 80


def test_target_week_qb_observation_and_duplicate_keys_fail():
    base, qb, ol = _inputs()
    future = qb.with_columns(
        pl.when(pl.col("game_id") == "g0").then(6)
        .otherwise(pl.col("home_qb_proxy_last_week"))
        .alias("home_qb_proxy_last_week")
    )
    with pytest.raises(DataContractError):
        assemble_pregame_risk(base, future, ol)
    with pytest.raises(DataContractError):
        assemble_pregame_risk(base, pl.concat([qb, qb.head(1)]), ol)


def test_turnovers_are_postgame_descriptions_only():
    base, qb, ol = _inputs()
    games = assemble_pregame_risk(base, qb, ol)
    pbp = pl.DataFrame({
        "season": [2024, 2024, 2024],
        "week": [6, 6, 6],
        "game_id": ["g0", "g0", "g0"],
        "posteam": ["A", "B", "B"],
        "interception": [1, 1, 0],
        "fumble_lost": [0, 0, 1],
    })
    enhanced, source = attach_postgame_turnovers(games, pbp)
    assert source == "POSTGAME_EXPLANATION_ONLY"
    assert enhanced.filter(
        pl.col("game_id") == "g0"
    )["turnover_net_home_advantage"][0] == 1
    assert enhanced["turnover_net_home_advantage"].null_count() == 79
    report = summarize_risk_attribution(enhanced, turnover_status=source)
    assert report["postgame_turnovers_as_features"] is False
    assert report["targets"]["total"]["risk_cohorts"][
        "postgame_turnover_swing"
    ]["covered_games"] == 1
