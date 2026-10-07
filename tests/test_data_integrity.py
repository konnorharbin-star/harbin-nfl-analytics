from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.data_integrity import assess_data_integrity
from nfl.espn_market import ESPNTwoWayMarket


def _projection(game_id: str = "2026_05_AAA_BBB") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "game_id": game_id,
                "home_team": "BBB",
                "away_team": "AAA",
                "baseline_home_margin": 3.0,
                "baseline_total": 45.0,
            }
        ]
    )


def _targets(game_id: str = "2026_05_AAA_BBB") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "game_id": game_id,
                "home_team": "BBB",
                "away_team": "AAA",
                "gameday": "2026-10-11",
                "gametime": "13:00",
            }
        ]
    )


def _market(captured_at: datetime) -> ESPNTwoWayMarket:
    return ESPNTwoWayMarket(
        game_id="2026_05_AAA_BBB",
        market_type="spread",
        provider="fixture",
        book="Book A",
        source_event_id="event-a",
        captured_at=captured_at,
        first_side="home",
        first_line=-3.0,
        first_american_odds=-110,
        second_side="away",
        second_line=3.0,
        second_american_odds=-110,
    )


def test_current_run_integrity_is_ok_for_consistent_inputs() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    result = assess_data_integrity(
        _projection(),
        _targets(),
        [_market(now)],
        pl.DataFrame([{"game_id": "2026_05_AAA_BBB"}]),
        now=now,
    )

    assert result["status"] == "OK"
    assert result["missing_data_count"] == 0
    assert result["duplicate_game_count"] == 0
    assert result["verified_quote_rows"] == 1
    assert result["errors"] == []


def test_projection_target_mismatch_is_hard_failure() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    result = assess_data_integrity(
        _projection("2026_05_CCC_DDD"),
        _targets(),
        [_market(now)],
        pl.DataFrame([{"game_id": "2026_05_AAA_BBB"}]),
        now=now,
    )

    assert result["status"] == "FAIL"
    assert result["missing_projection_games"] == ["2026_05_AAA_BBB"]
    assert result["extra_projection_games"] == ["2026_05_CCC_DDD"]


def test_future_verified_quote_is_hard_failure() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    result = assess_data_integrity(
        _projection(),
        _targets(),
        [_market(datetime(2026, 10, 6, 19, tzinfo=UTC))],
        pl.DataFrame([{"game_id": "2026_05_AAA_BBB"}]),
        now=now,
    )

    assert result["status"] == "FAIL"
    assert result["future_verified_quote_rows"] == 1


def test_source_failure_is_visible_warning_without_invented_data() -> None:
    now = datetime(2026, 10, 6, 18, tzinfo=UTC)
    result = assess_data_integrity(
        _projection(),
        _targets(),
        [_market(now)],
        pl.DataFrame([{"game_id": "2026_05_AAA_BBB"}]),
        source_errors=["injuries: source unavailable"],
        now=now,
    )

    assert result["status"] == "WARN"
    assert result["source_health"]["status"] == "WARN"
    assert "injuries: source unavailable" in result["warnings"]
