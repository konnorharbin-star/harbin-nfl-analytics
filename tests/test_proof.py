from __future__ import annotations

import json

import polars as pl

from nfl.proof import build_evidence_report


def _bet(
    game_id: str,
    *,
    verified: bool,
    quote_verified: bool,
    clv: float | None,
) -> dict[str, object]:
    return {
        "season": 2025,
        "week": 5,
        "game_id": game_id,
        "market_type": "spread",
        "side": "home",
        "probability_edge": 0.08,
        "expected_value_per_unit": 0.10,
        "result": "win",
        "net_units": 0.9,
        "clv_proxy": clv,
        "entry_line_observed": verified,
        "entry_price_verified": verified,
        "entry_quote_verified": quote_verified,
    }


def test_proof_uses_verified_provider_rows_without_promoting_archive_fallbacks(
    tmp_path,
) -> None:
    backtest = tmp_path / "free_market_backtest.json"
    backtest.write_text(
        json.dumps(
            {
                "overall": {"bets": 2, "roi_per_unit_staked": -0.1},
                "by_market": {},
                "by_season": {},
                "source": {"historical_market": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    free_bets = tmp_path / "free_market_bets.csv"
    pl.DataFrame(
        [
            _bet(
                "free-a",
                verified=False,
                quote_verified=False,
                clv=None,
            ),
            _bet(
                "free-b",
                verified=False,
                quote_verified=False,
                clv=None,
            ),
        ]
    ).write_csv(free_bets)

    provider_bets = tmp_path / "verified_market_bets.csv"
    pl.DataFrame(
        [
            _bet(
                "provider-a",
                verified=True,
                quote_verified=True,
                clv=0.5,
            ),
            _bet(
                "provider-b",
                verified=True,
                quote_verified=True,
                clv=0.25,
            ),
            _bet(
                "provider-unverified",
                verified=True,
                quote_verified=False,
                clv=1.0,
            ),
        ]
    ).write_csv(provider_bets)

    report = build_evidence_report(
        backtest_path=backtest,
        bets_path=free_bets,
        verified_bets_path=provider_bets,
    )

    promotion = report["promotion_sample"]
    assert promotion["verified_archive_bets"] == 0
    assert promotion["verified_provider_bets"] == 2
    assert promotion["verified_bets"] == 2
    assert promotion["verified_clv_samples"] == 2
    assert promotion["verified_clv_coverage"] == 1.0
    assert promotion["entry_quote_verified"] is True
    assert report["source"]["verified_provider_rows"] == 2
