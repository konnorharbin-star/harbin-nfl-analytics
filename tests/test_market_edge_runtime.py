from __future__ import annotations

import polars as pl

from nfl.market_edge_runtime import (
    assess_market_edge_candidate,
    build_verified_market_edge_registry,
)


def _history(
    *,
    model_probability: float,
    market_probability: float,
    tune_rate: float,
    holdout_rate: float,
    verified: bool = True,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season, rate in ((2024, tune_rate), (2025, holdout_rate)):
        wins_per_100 = int(round(rate * 100))
        for market in ("moneyline", "spread", "total"):
            for index in range(200):
                won = (index % 100) < wins_per_100
                rows.append(
                    {
                        "season": season,
                        "week": 1 + index % 18,
                        "game_id": f"{season}-{market}-{index}",
                        "market_type": market,
                        "side": (
                            "home"
                            if market != "total"
                            else "over"
                        ),
                        "model_probability": model_probability,
                        "no_vig_probability": market_probability,
                        "probability_edge": (
                            model_probability - market_probability
                        ),
                        "decimal_odds": 1.91,
                        "result": "win" if won else "loss",
                        "net_units": 0.91 if won else -1.0,
                        "entry_price_verified": verified,
                        "entry_quote_verified": verified,
                    }
                )
    return pl.DataFrame(rows)


def test_verified_market_prefers_market_when_model_is_overconfident() -> None:
    registry = build_verified_market_edge_registry(
        _history(
            model_probability=0.75,
            market_probability=0.55,
            tune_rate=0.55,
            holdout_rate=0.55,
        )
    )

    assert registry["operational_ready"] is False
    assert registry["market_only_markets"] == [
        "moneyline",
        "spread",
        "total",
    ]
    for market in ("moneyline", "spread", "total"):
        entry = registry["markets"][market]
        assert entry["status"] == "MARKET_ONLY_PREFERRED"
        assert entry["selected_alpha"] == 0.0
        assert entry["operational_alpha"] == 0.0
        assert entry["decision_enabled"] is False


def test_verified_incremental_model_value_can_enable_market_blend() -> None:
    registry = build_verified_market_edge_registry(
        _history(
            model_probability=0.65,
            market_probability=0.55,
            tune_rate=0.65,
            holdout_rate=0.65,
        )
    )

    assert registry["operational_ready"] is True
    assert set(registry["validated_incremental_markets"]) == {
        "moneyline",
        "spread",
        "total",
    }
    for market in ("moneyline", "spread", "total"):
        entry = registry["markets"][market]
        assert entry["status"] == "VALIDATED_INCREMENTAL"
        assert entry["selected_alpha"] > 0.0
        assert entry["operational_alpha"] > 0.0
        assert entry["decision_enabled"] is True


def test_holdout_outcomes_never_select_alpha() -> None:
    base = _history(
        model_probability=0.68,
        market_probability=0.55,
        tune_rate=0.62,
        holdout_rate=0.62,
    )
    first = build_verified_market_edge_registry(base)

    mutated = base.with_columns(
        pl.when(pl.col("season") == 2025)
        .then(pl.lit("loss"))
        .otherwise(pl.col("result"))
        .alias("result")
    )
    second = build_verified_market_edge_registry(mutated)

    for market in ("moneyline", "spread", "total"):
        assert first["markets"][market]["selected_alpha"] == (
            second["markets"][market]["selected_alpha"]
        )


def test_market_only_candidate_has_zero_executable_edge() -> None:
    registry = build_verified_market_edge_registry(
        _history(
            model_probability=0.75,
            market_probability=0.55,
            tune_rate=0.55,
            holdout_rate=0.55,
        )
    )
    decision = assess_market_edge_candidate(
        registry,
        market_type="moneyline",
        model_probability=0.70,
        no_vig_probability=0.55,
        decimal_odds=1.91,
    )

    assert decision.ready is False
    assert decision.status == "MARKET_ONLY_PREFERRED"
    assert decision.operational_alpha == 0.0
    assert decision.decision_probability == 0.55
    assert abs(decision.decision_probability_edge) < 1e-12
    assert decision.raw_probability_edge > 0.0


def test_missing_registry_fails_closed_to_market_probability() -> None:
    decision = assess_market_edge_candidate(
        None,
        market_type="spread",
        model_probability=0.62,
        no_vig_probability=0.50,
        decimal_odds=1.91,
    )

    assert decision.ready is False
    assert decision.status == "BLOCKED"
    assert decision.operational_alpha == 0.0
    assert decision.decision_probability == 0.50
    assert abs(decision.decision_probability_edge) < 1e-12


def test_unverified_rows_cannot_enable_market_edge_registry() -> None:
    registry = build_verified_market_edge_registry(
        _history(
            model_probability=0.65,
            market_probability=0.55,
            tune_rate=0.65,
            holdout_rate=0.65,
            verified=False,
        )
    )

    assert registry["status"] == "INSUFFICIENT_DATA"
    assert registry["operational_ready"] is False
