from __future__ import annotations

import polars as pl

from nfl.regime_reliability import (
    assess_candidate_regime_reliability,
    build_regime_reliability_report,
    candidate_segment_keys,
)


def _history(holdout_rate: float = 0.60) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2024, 2025):
        for market in ("moneyline", "spread", "total"):
            rate = 0.60 if season == 2024 else holdout_rate
            for index in range(240):
                side = (
                    ("home" if index % 2 == 0 else "away")
                    if market != "total"
                    else ("over" if index % 2 == 0 else "under")
                )
                market_probability = (
                    0.52 if market == "total" and side == "over"
                    else 0.48 if market == "total"
                    else 0.55 if index % 2 == 0
                    else 0.45
                )
                probability = 0.60
                line = (
                    (-3.0 if index % 2 == 0 else 3.0)
                    if market == "spread"
                    else None
                )
                won = (index % 20) < int(round(rate * 20))
                rows.append(
                    {
                        "season": season,
                        "week": 1 + index % 18,
                        "game_id": f"{season}-{market}-{index}",
                        "market_type": market,
                        "side": side,
                        "line": line,
                        "model_probability": probability,
                        "no_vig_probability": market_probability,
                        "probability_edge": probability - market_probability,
                        "projected_home_margin": (2.0, 5.0, 9.0)[index % 3],
                        "projected_total": (40.0, 45.0, 51.0)[index % 3],
                        "result": "win" if won else "loss",
                        "net_units": 0.91 if won else -1.0,
                        "clv_proxy": 0.02,
                        "entry_price_verified": True,
                        "entry_quote_verified": True,
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def test_segment_keys_are_fixed() -> None:
    keys = candidate_segment_keys(
        market_type="spread",
        side="home",
        model_probability=0.60,
        no_vig_probability=0.48,
        probability_edge=0.12,
        projected_home_margin=6.0,
        projected_total=45.0,
        line=-3.0,
        week=8,
    )
    assert keys["market"] == "market:spread"
    assert keys["role"] == "role:spread:favorite"
    assert keys["confidence"] == "confidence:spread:58_62"
    assert keys["edge"] == "edge:spread:10_plus"
    assert keys["margin_environment"] == (
        "margin_environment:spread:moderate"
    )
    assert keys["season_phase"] == "season_phase:spread:middle"


def test_confidence_and_edge_segments_do_not_mix_markets() -> None:
    moneyline = candidate_segment_keys(
        market_type="moneyline",
        side="home",
        model_probability=0.60,
        no_vig_probability=0.55,
        probability_edge=0.05,
        projected_home_margin=5.0,
        projected_total=45.0,
        week=8,
    )
    total = candidate_segment_keys(
        market_type="total",
        side="over",
        model_probability=0.60,
        no_vig_probability=0.55,
        probability_edge=0.05,
        projected_home_margin=5.0,
        projected_total=45.0,
        line=44.5,
        week=8,
    )

    assert moneyline["confidence"] != total["confidence"]
    assert moneyline["edge"] != total["edge"]
    assert moneyline["season_phase"] != total["season_phase"]


def test_reliable_market_requires_two_seasons() -> None:
    report = build_regime_reliability_report(_history())
    statuses = report["summary"]["market_status"]
    assert statuses == {
        "moneyline": "RELIABLE",
        "spread": "RELIABLE",
        "total": "RELIABLE",
    }
    assert report["summary"]["operational_ready"] is True


def test_bad_holdout_marks_markets_unreliable() -> None:
    report = build_regime_reliability_report(_history(0.45))
    assert report["segments"]["market:moneyline"]["status"] == "UNRELIABLE"
    assert report["segments"]["market:spread"]["status"] == "UNRELIABLE"
    assert report["segments"]["market:total"]["status"] == "UNRELIABLE"
    assert report["summary"]["operational_ready"] is False


def test_unverified_rows_cannot_create_registry() -> None:
    frame = _history().with_columns(
        pl.lit(False).alias("entry_quote_verified")
    )
    report = build_regime_reliability_report(frame)
    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["operational_registry"]["operational_ready"] is False


def test_candidate_gate_blocks_unreliable_segment() -> None:
    report = build_regime_reliability_report(_history())
    registry = report["operational_registry"]
    segments = registry["segments"]
    segments["confidence:moneyline:58_62"] = {
        **segments["confidence:moneyline:58_62"],
        "status": "UNRELIABLE",
    }
    assessment = assess_candidate_regime_reliability(
        registry,
        market_type="moneyline",
        side="home",
        model_probability=0.60,
        no_vig_probability=0.55,
        probability_edge=0.05,
        projected_home_margin=5.0,
        projected_total=45.0,
        week=8,
    )
    assert assessment["ready"] is False
    assert "confidence:moneyline:58_62" in assessment["blocked_segments"]


def test_candidate_gate_missing_registry_fails_closed() -> None:
    assessment = assess_candidate_regime_reliability(
        None,
        market_type="total",
        side="under",
        model_probability=0.60,
        no_vig_probability=0.48,
        probability_edge=0.12,
        projected_home_margin=3.0,
        projected_total=41.0,
        line=41.5,
        week=3,
    )
    assert assessment["ready"] is False
    assert assessment["status"] == "BLOCKED"
