"""Regression checks for market-outlier quarantine; never generate wagers."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.price_comparison_scan import normalize, scan

NOW = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)


def paired_quotes(*, books=7):
    game = {
        "game_id": "2026_05_PHI_JAX",
        "home_team": "JAX",
        "away_team": "PHI",
        "kickoff": (NOW + timedelta(hours=18)).isoformat(),
    }
    quotes = []
    for i in range(books):
        odds = (-380, 295) if i % 2 else (-400, 315)
        quotes.extend(
            normalize(
                game,
                {
                    "provider": f"Reference{i}",
                    "source": "test_fixture",
                    "home_ml": odds[0],
                    "away_ml": odds[1],
                },
                NOW.isoformat(),
            )
        )
    quotes.extend(
        normalize(
            game,
            {
                "provider": "FanDuel",
                "source": "test_fixture",
                "home_ml": 110,
                "away_ml": -130,
            },
            NOW.isoformat(),
        )
    )
    return quotes


def test_realistic_jaguars_anomaly_is_not_ranked_as_a_market_edge():
    result = scan(paired_quotes(), NOW)
    quarantined = result["quarantined_quotes"]
    assert len(quarantined) == 1
    assert quarantined[0]["book_label"] == "FanDuel"
    assert quarantined[0]["american_odds_pair"] == [110.0, -130.0]
    assert quarantined[0]["absolute_divergence_pp"] > 25.0
    assert quarantined[0]["status"] == "UNVERIFIED_EXTREME_SOURCE_OUTLIER"
    assert not any(x["book_label"] == "FanDuel" for x in result["candidates"])
    assert result["counts"]["extreme_unverified_quotes_quarantined"] == 1
    assert result["betting_authorized"] is False
    assert result["qualified_bets"] == 0


def test_source_confirmed_large_differences_remain_diagnostic_not_bets():
    rows = paired_quotes()
    for r in rows:
        if r["book_label"] == "FanDuel":
            r.update(
                book_identity_verified=True,
                source_quote_time_verified=True,
                executable_price_verified=True,
            )
    result = scan(rows, NOW)
    assert result["quarantined_quotes"] == []
    assert result["candidates"][0]["book_label"] == "FanDuel"
    assert result["candidates"][0]["betting_authorized"] is False
    assert result["candidates"][0]["stake_units"] == 0


def test_quarantining_fourth_book_drops_below_minimum_reference_coverage():
    result = scan(paired_quotes(books=3), NOW)
    assert result["counts"]["insufficient_after_outlier_quarantine"] == 1
    assert result["candidates"] == []


def test_clean_quotes_without_large_dislocation_pass_coherence_check():
    rows = paired_quotes()
    for row in rows:
        if row["book_label"] == "FanDuel":
            row["odds"] = [-380, 295]
    result = scan(rows, NOW)
    assert result["quarantined_quotes"] == []
    assert result["counts"]["extreme_unverified_quotes_quarantined"] == 0


def test_distant_line_groups_and_stale_quotes_never_create_proof():
    rows = deepcopy(paired_quotes())
    rows[0]["line"] = 9.5
    rows[0]["market"] = "total"
    rows[0]["sides"] = ["over", "under"]
    rows[0]["odds"] = [-110, -110]
    rows[0]["observed_at"] = (NOW - timedelta(minutes=16)).isoformat()
    report = scan(rows, NOW)
    assert report["counts"]["stale_future_or_started"] >= 1
    assert all(x["betting_authorized"] is False for x in report["candidates"])


@pytest.mark.parametrize("bad", [None, "", 0, 90])
def test_unusable_prices_do_not_create_quarantine(bad):
    rows = paired_quotes()
    rows[-1]["odds"][0] = bad
    report = scan(rows, NOW)
    assert report["counts"]["invalid_pair"] >= 1
    assert report["quarantined_quotes"] == []


def test_archived_october_10_jaguars_dislocation_is_quarantined():
    """Exercise the raw unaltered capture, not only a synthetic analogue."""
    archive = Path(
        "history/price_scan_v1/captures/6f49d529fec7adf50373711b.json"
    )
    assert archive.exists(), "Expected exact immutable capture for regression"
    payload = json.loads(archive.read_text(encoding="utf-8"))
    observed = datetime.fromisoformat(payload["report"]["observed_at"])
    scan_result = scan(payload["source_records"], observed)
    assert any(
        r["game_id"] == "2026_05_PHI_JAX"
        and r["book_label"] == "FanDuel"
        and r["american_odds_pair"] == [110.0, -130.0]
        for r in scan_result["quarantined_quotes"]
    )
    assert not any(
        r["game_id"] == "2026_05_PHI_JAX"
        and r["book_label"] == "FanDuel"
        for r in scan_result["candidates"]
    )
