from __future__ import annotations

import polars as pl

from nfl.market_shrinkage import (
    calibration_buckets,
    evaluate_market_edge_shrinkage,
    fit_alpha,
    shrink_probability,
    signed_residual_probability,
    signed_residual_research,
)


def _synthetic_bets(
    *,
    holdout_win_rate: float = 0.60,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for market in ("moneyline", "spread", "total"):
        for season in (2022, 2023, 2024, 2025):
            win_rate = 0.60 if season < 2025 else holdout_win_rate
            for index in range(200):
                win = (index % 100) < int(round(win_rate * 100))
                rows.append(
                    {
                        "season": season,
                        "week": 1 + index % 18,
                        "game_id": f"{season}-{market}-{index}",
                        "market_type": market,
                        "model_probability": 0.80,
                        "no_vig_probability": 0.60,
                        "decimal_odds": 1.91,
                        "result": "win" if win else "loss",
                        "net_units": 0.91 if win else -1.0,
                    }
                )
    return pl.DataFrame(rows)


def test_shrink_probability_endpoints_match_market_and_model() -> None:
    assert abs(shrink_probability(0.75, 0.55, 0.0) - 0.55) < 1e-12
    assert abs(shrink_probability(0.75, 0.55, 1.0) - 0.75) < 1e-12


def test_overconfident_model_shrinks_toward_market() -> None:
    bets = _synthetic_bets().filter(pl.col("market_type") == "spread")
    development = bets.filter(pl.col("season") < 2024)

    alpha, _ = fit_alpha(development)

    assert alpha is not None
    assert alpha < 1.0


def test_holdout_outcomes_do_not_select_alpha() -> None:
    first = evaluate_market_edge_shrinkage(_synthetic_bets(holdout_win_rate=0.60))
    second = evaluate_market_edge_shrinkage(_synthetic_bets(holdout_win_rate=0.20))

    for market in ("moneyline", "spread", "total"):
        assert first["markets"][market]["alpha"] == second["markets"][market]["alpha"]


def test_research_report_never_changes_canonical_policy() -> None:
    report = evaluate_market_edge_shrinkage(_synthetic_bets())

    assert report["status"] == "RESEARCH_ONLY"
    assert report["research_conclusion"] == "NO_INCREMENTAL_MODEL_VALUE"
    assert report["forward_shrinkage_shadow_recommended"] is False
    assert report["canonical_market_probability_change_enabled"] is False
    assert report["betting_policy_change_enabled"] is False
    for market in ("moneyline", "spread", "total"):
        assert report["markets"][market]["canonical_change_enabled"] is False


def test_shrinkage_improves_overconfident_holdout_probabilities() -> None:
    report = evaluate_market_edge_shrinkage(_synthetic_bets())

    for market in ("moneyline", "spread", "total"):
        candidate = report["markets"][market]
        raw = candidate["holdout_probability"]["raw_model"]
        shrunk = candidate["holdout_probability"]["shrunk_model"]
        assert shrunk["brier"] < raw["brier"]
        assert shrunk["log_loss"] < raw["log_loss"]
        assert candidate["probability_validated"] is True
        assert candidate["status"] == "MARKET_ONLY_PREFERRED"


def test_fixed_calibration_buckets_report_historical_overconfidence() -> None:
    frame = pl.DataFrame(
        {
            "result": ["win", "loss", "win", "push"],
            "model_probability": [0.90, 0.90, 0.90, 0.90],
            "no_vig_probability": [0.60, 0.60, 0.60, 0.60],
        }
    )
    buckets = calibration_buckets(frame)
    bucket = buckets["0.6-0.7"]
    assert bucket["rows"] == 3
    assert abs(bucket["observed_win_rate"] - 2 / 3) < 1e-12
    assert bucket["raw_brier"] > bucket["market_brier"]
    assert buckets["0.4-0.5"]["status"] == "EMPTY"


def test_calibration_report_keeps_holdout_descriptive_only() -> None:
    report = evaluate_market_edge_shrinkage(_synthetic_bets())
    for market in ("moneyline", "spread", "total"):
        diagnostics = report["markets"][market]["calibration_diagnostics"]
        assert diagnostics["holdout_2025"]["0.6-0.7"]["status"] == "DESCRIPTIVE_ONLY"
        assert diagnostics["holdout_2025"]["0.6-0.7"]["rows"] == 200
        assert report["markets"][market]["canonical_change_enabled"] is False


def test_signed_residual_research_does_not_use_holdout_for_fitting() -> None:
    bets = _synthetic_bets()
    first = signed_residual_research(
        bets.filter(pl.col("season") < 2024),
        bets.filter(pl.col("season") == 2024),
        bets.filter(pl.col("season") == 2025),
    )
    changed = _synthetic_bets(holdout_win_rate=0.20)
    second = signed_residual_research(
        changed.filter(pl.col("season") < 2024),
        changed.filter(pl.col("season") == 2024),
        changed.filter(pl.col("season") == 2025),
    )
    assert first["selected_alpha"] == second["selected_alpha"]
    assert first["canonical_market_probability_change_enabled"] is False
    assert first["betting_policy_change_enabled"] is False


def test_signed_residual_probability_can_reverse_direction() -> None:
    assert signed_residual_probability(0.8, 0.5, -0.25) < 0.5
    assert abs(signed_residual_probability(0.8, 0.5, 0) - 0.5) < 1e-12
    assert abs(signed_residual_probability(0.8, 0.5, 1) - 0.8) < 1e-12
