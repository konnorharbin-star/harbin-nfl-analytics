from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from nfl.decision_intelligence import (
    attach_decision_intelligence,
    consensus_diagnostics,
)
from nfl.espn_market import ESPNTwoWayMarket
from nfl.policy import DEFAULT_POLICY


def _spread_market(
    *,
    book: str,
    home_line: float,
    captured_at: datetime,
) -> ESPNTwoWayMarket:
    return ESPNTwoWayMarket(
        game_id="2026_05_AAA_BBB",
        market_type="spread",
        provider="fixture",
        book=book,
        source_event_id=f"event-{book}",
        captured_at=captured_at,
        first_side="home",
        first_line=home_line,
        first_american_odds=-110,
        second_side="away",
        second_line=-home_line,
        second_american_odds=-110,
    )


def _candidate(now: datetime, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2026,
        "week": 5,
        "game_id": "2026_05_AAA_BBB",
        "date": "2026-10-11",
        "kickoff": (now + timedelta(hours=6)).isoformat(),
        "away_team": "AAA",
        "home_team": "BBB",
        "quant_market": "spread",
        "quant_side": "home",
        "quant_book": "Book A",
        "quant_price": -3.0,
        "quant_odds": -110,
        "quant_quote_at": now.isoformat(),
        "quant_signal": "BET",
        "research_signal": "BET",
        "market_execution_verified": True,
        "market_quote_timestamp_verified": True,
        "market_book_count": 2,
        "market_disagreement_severity": "LOW",
        "market_dispersion_high": False,
    }
    row.update(overrides)
    return row


def test_spread_consensus_is_cross_book_home_margin() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    markets = [
        _spread_market(book="Book A", home_line=-3.0, captured_at=now),
        _spread_market(book="Book B", home_line=-3.5, captured_at=now),
    ]

    result = consensus_diagnostics(
        "spread",
        markets,
        model_home_margin=6.0,
        model_total=45.0,
        model_home_probability=0.65,
        config=DEFAULT_POLICY["decision_intelligence"],
    )

    assert result["market_consensus_verified"] is True
    assert result["market_consensus_books"] == 2
    assert result["market_consensus_metric"] == "market_home_margin"
    assert abs(float(result["market_consensus_value"]) - 3.25) < 1e-12
    assert abs(float(result["market_dispersion"]) - 0.25) < 1e-12
    assert abs(float(result["market_disagreement"]) - 2.75) < 1e-12
    assert result["market_disagreement_severity"] == "MEDIUM"


def test_clean_bet_is_bet_now_in_shadow_timing() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    frame, meta = attach_decision_intelligence(
        pl.DataFrame([_candidate(now)]),
        policy=DEFAULT_POLICY,
        snapshots=pl.DataFrame(),
        now=now,
    )

    row = frame.row(0, named=True)
    assert row["execution_action"] == "BET_NOW"
    assert row["research_execution_action"] == "BET_NOW"
    assert row["timing_market_move"] == "NO_HISTORY"
    assert meta["enforced"] is False


def test_high_disagreement_waits_for_review() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    frame, _ = attach_decision_intelligence(
        pl.DataFrame(
            [
                _candidate(
                    now,
                    market_disagreement_severity="HIGH",
                )
            ]
        ),
        policy=DEFAULT_POLICY,
        snapshots=pl.DataFrame(),
        now=now,
    )

    row = frame.row(0, named=True)
    assert row["execution_action"] == "WAIT"
    assert "disagreement" in row["execution_action_reason"]


def test_bettor_improving_line_waits_for_non_strong_signal() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    prior = now - timedelta(minutes=20)
    snapshots = pl.DataFrame(
        [
            {
                "game_id": "2026_05_AAA_BBB",
                "market_type": "spread",
                "book": "Book A",
                "captured_at": prior.isoformat(),
                "first_side": "home",
                "first_line": -3.5,
                "first_american_odds": -110,
                "second_side": "away",
                "second_line": 3.5,
                "second_american_odds": -110,
            }
        ]
    )

    frame, meta = attach_decision_intelligence(
        pl.DataFrame([_candidate(now, quant_price=-3.0)]),
        policy=DEFAULT_POLICY,
        snapshots=snapshots,
        now=now,
    )

    row = frame.row(0, named=True)
    assert row["timing_market_move"] == "BETTOR_IMPROVED"
    assert abs(float(row["timing_line_value_move"]) - 0.5) < 1e-12
    assert row["execution_action"] == "WAIT"
    assert meta["movement_coverage"] == 1.0


def test_research_only_quote_fails_closed() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    frame, _ = attach_decision_intelligence(
        pl.DataFrame(
            [
                _candidate(
                    now,
                    market_execution_verified=False,
                    market_quote_timestamp_verified=False,
                )
            ]
        ),
        policy=DEFAULT_POLICY,
        snapshots=pl.DataFrame(),
        now=now,
    )

    row = frame.row(0, named=True)
    assert row["execution_action"] == "PASS"
    assert row["research_execution_action"] == "PASS"
    assert "research-only" in row["execution_action_reason"]
