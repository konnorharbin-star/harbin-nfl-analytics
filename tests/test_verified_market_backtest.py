from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.verified_market_backtest import (
    build_market_decisions,
    build_verified_market_bets,
    eligible_projection_game_ids,
)


def _projections() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(64):
        projected_margin = float((index % 9) - 4)
        projected_total = 42.0 + float(index % 7)
        rows.append(
            {
                "season": 2024,
                "week": 1 + (index % 16),
                "game_id": f"2024_{index:03d}",
                "home_team": "AAA",
                "away_team": "BBB",
                "projected_home_margin": projected_margin,
                "projected_total": projected_total,
                "actual_home_margin": projected_margin + float((index % 5) - 2),
                "actual_total": projected_total + float((index % 7) - 3),
            }
        )
    rows.append(
        {
            "season": 2025,
            "week": 5,
            "game_id": "2025_05_NE_BUF",
            "home_team": "BUF",
            "away_team": "NE",
            "projected_home_margin": 7.0,
            "projected_total": 45.0,
            "actual_home_margin": 7.0,
            "actual_total": 47.0,
        }
    )
    return pl.DataFrame(rows)


def _quote(
    *,
    book: str,
    snapshot_id: str,
    captured_at: datetime,
    decision_time: datetime,
    side: str,
    line: float,
    odds: int,
) -> dict[str, object]:
    return {
        "game_id": "2025_05_NE_BUF",
        "market_type": "spread",
        "side": side,
        "line": line,
        "american_odds": odds,
        "provider": "fixture",
        "book": book,
        "captured_at": captured_at,
        "snapshot_id": snapshot_id,
        "source_event_id": "provider-event-1",
        "decision_time": decision_time,
    }


def test_market_decisions_use_explicit_utc_kickoff_offset() -> None:
    schedules = pl.DataFrame(
        {
            "game_id": ["2025_05_NE_BUF"],
            "gameday": ["2025-10-05"],
            "gametime": ["13:00"],
        }
    )

    decisions = build_market_decisions(
        schedules,
        ["2025_05_NE_BUF"],
        minutes_before_kickoff=60,
    )

    assert decisions.height == 1
    assert decisions.row(0, named=True)["decision_time"] == datetime(
        2025,
        10,
        5,
        16,
        0,
        tzinfo=UTC,
    )


def test_eligible_games_require_chronological_probability_history() -> None:
    projections = _projections()

    eligible = eligible_projection_game_ids(projections)

    assert eligible == ["2025_05_NE_BUF"]


def test_verified_backtest_line_shops_and_attaches_same_book_clv() -> None:
    projections = _projections()
    entry_at = datetime(2025, 10, 5, 15, 55, tzinfo=UTC)
    decision_at = datetime(2025, 10, 5, 16, 0, tzinfo=UTC)
    close_at = datetime(2025, 10, 5, 16, 50, tzinfo=UTC)
    close_decision = datetime(2025, 10, 5, 16, 55, tzinfo=UTC)

    entries = pl.DataFrame(
        [
            _quote(
                book="book-a",
                snapshot_id="entry-a",
                captured_at=entry_at,
                decision_time=decision_at,
                side="home",
                line=-3.5,
                odds=-110,
            ),
            _quote(
                book="book-a",
                snapshot_id="entry-a",
                captured_at=entry_at,
                decision_time=decision_at,
                side="away",
                line=3.5,
                odds=-110,
            ),
            _quote(
                book="book-b",
                snapshot_id="entry-b",
                captured_at=entry_at,
                decision_time=decision_at,
                side="home",
                line=-3.5,
                odds=105,
            ),
            _quote(
                book="book-b",
                snapshot_id="entry-b",
                captured_at=entry_at,
                decision_time=decision_at,
                side="away",
                line=3.5,
                odds=-125,
            ),
        ]
    )
    closes = pl.DataFrame(
        [
            _quote(
                book="book-b",
                snapshot_id="close-b",
                captured_at=close_at,
                decision_time=close_decision,
                side="home",
                line=-4.5,
                odds=-110,
            ),
            _quote(
                book="book-b",
                snapshot_id="close-b",
                captured_at=close_at,
                decision_time=close_decision,
                side="away",
                line=4.5,
                odds=-110,
            ),
        ]
    )

    bets = build_verified_market_bets(projections, entries, closes)

    assert bets.height == 1
    row = bets.row(0, named=True)
    assert row["market_type"] == "spread"
    assert row["side"] == "home"
    assert row["book"] == "book-b"
    assert row["market_book_count"] == 2
    assert row["result"] == "win"
    assert row["entry_price_verified"] is True
    assert row["entry_quote_verified"] is True
    assert row["entry_price_stage"] == "timestamped_provider_entry"
    assert row["closing_quote_verified"] is True
    assert row["closing_snapshot_at"] == close_at
    assert row["clv_proxy"] == 1.0
