import math

import polars as pl

from nfl.probability_runtime import (
    apply_probability_reliability_veto,
    build_operational_probability_distribution,
)


def _history() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    for season in (2022, 2023, 2024, 2025):
        for index in range(160):
            counter += 1
            margin = float((index % 19) - 9)
            total = 41.0 + float(index % 11)
            margin_error = float(((counter * 7) % 23) - 11)
            total_error = float(((counter * 9) % 27) - 13)
            rows.append(
                {
                    "season": season,
                    "week": 5 + index % 14,
                    "game_id": f"{season}-{index}",
                    "projected_home_margin": margin,
                    "projected_total": total,
                    "actual_home_margin": margin + margin_error,
                    "actual_total": total + total_error,
                }
            )
    return pl.DataFrame(rows)


def test_operational_probability_builder_returns_validated_metadata() -> None:
    distribution, meta = build_operational_probability_distribution(
        _history(),
        current_season=2026,
    )

    probability = distribution.home_win_probability(3.0, 46.0)
    assert meta["status"] == "READY"
    assert meta["validation_season"] == 2024
    assert meta["holdout_season"] == 2025
    assert isinstance(meta["reliability_ready"], bool)
    assert isinstance(meta["model_family"], str)
    assert math.isfinite(probability)
    assert 0.0 < probability < 1.0
    assert distribution.margin_sigma_for(3.0, 46.0) > 0
    assert distribution.total_sigma_for(46.0, 3.0) > 0


def test_operational_probability_builder_falls_back_without_nested_history() -> None:
    distribution, meta = build_operational_probability_distribution(
        _history().filter(pl.col("season") == 2025),
        current_season=2026,
    )

    assert meta["status"] == "FALLBACK"
    assert meta["reliability_ready"] is False
    assert 0.0 < distribution.home_win_probability(0.0, 45.0) < 1.0


def test_probability_reliability_veto_blocks_all_markets_when_not_ready() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_signal": "BET",
                "production_signal": "BET",
                "research_signal": "STRONG",
                "portfolio_signal": "BET",
                "stake_units": 0.5,
                "research_stake_units": 0.5,
            }
        ]
    )

    row = apply_probability_reliability_veto(
        candidates,
        {
            "reliability_ready": False,
            "model_family": "gaussian_unconditional",
        },
    ).row(0, named=True)

    assert row["probability_reliability_veto"] is True
    assert row["quant_signal"] == "PASS"
    assert row["research_signal"] == "PASS"
    assert row["portfolio_signal"] == "PASS"
    assert row["stake_units"] == 0.0
    assert row["research_stake_units"] == 0.0


def test_probability_reliability_veto_preserves_signals_when_ready() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_signal": "BET",
                "research_signal": "BET",
                "stake_units": 0.25,
                "research_stake_units": 0.25,
            }
        ]
    )

    row = apply_probability_reliability_veto(
        candidates,
        {
            "reliability_ready": True,
            "model_family": "conditional_student_t",
        },
    ).row(0, named=True)

    assert row["probability_reliability_veto"] is False
    assert row["quant_signal"] == "BET"
    assert row["stake_units"] == 0.25
