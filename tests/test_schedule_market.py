from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from nfl.contracts import DataContractError
from nfl.execution_market import validate_execution_row
from nfl.line_history import append_market_snapshots, load_market_snapshots
from nfl.market_intel import build_market_intelligence
from nfl.pro_market import collect_current_markets
from nfl.schedule_market import (
    NFLVERSE_SCHEDULE_PROVIDER,
    schedule_snapshot_markets,
)


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_ARI_NYG",
                "home_team": "NYG",
                "away_team": "ARI",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "away_moneyline": -135,
                "home_moneyline": 114,
                "spread_line": -2.5,
                "away_spread_odds": -112,
                "home_spread_odds": -108,
                "total_line": 44.5,
                "under_odds": -112,
                "over_odds": -108,
            }
        ]
    )


def _projection() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_ARI_NYG",
                "gameday": "2026-10-04",
                "home_team": "NYG",
                "away_team": "ARI",
                "baseline_home_margin": -1.0,
                "baseline_total": 45.0,
            }
        ]
    )


def _historical() -> pl.DataFrame:
    rows = []
    for index in range(80):
        projected_margin = float((index % 7) - 3)
        projected_total = 43.0 + float(index % 6)
        rows.append(
            {
                "projected_home_margin": projected_margin,
                "projected_total": projected_total,
                "actual_home_margin": projected_margin + float((index % 9) - 4),
                "actual_total": projected_total + float((index % 11) - 5),
            }
        )
    return pl.DataFrame(rows)


def test_schedule_snapshot_normalizes_three_markets_and_spread_sign() -> None:
    stamp = datetime(2026, 10, 1, 15, tzinfo=UTC)
    markets = schedule_snapshot_markets(_targets(), captured_at=stamp)

    assert len(markets) == 3
    assert {market.market_type for market in markets} == {
        "moneyline",
        "spread",
        "total",
    }
    assert {market.provider for market in markets} == {NFLVERSE_SCHEDULE_PROVIDER}
    spread = next(market for market in markets if market.market_type == "spread")
    assert spread.first_side == "home"
    assert spread.first_line == 2.5
    assert spread.second_line == -2.5
    total = next(market for market in markets if market.market_type == "total")
    assert total.first_line == 44.5
    assert total.second_line == 44.5


def test_aggregator_uses_schedule_snapshot_when_verified_source_fails() -> None:
    class BlockedESPN:
        def current_markets(self, targets: pl.DataFrame, *, week: int):
            raise DataContractError("HTTP Error 403: Forbidden")

    class NoOptionalSource:
        configured = False

        def current_markets(self, targets: pl.DataFrame):
            return []

    markets, meta = collect_current_markets(
        _targets(),
        week=4,
        espn_client=BlockedESPN(),  # type: ignore[arg-type]
        pro_client=NoOptionalSource(),  # type: ignore[arg-type]
    )

    assert len(markets) == 3
    assert meta["status"] == "RESEARCH_FALLBACK"
    assert meta["research_fallback_rows"] == 3
    assert meta["verified_rows"] == 0
    assert meta["books"] == 0
    assert meta["multi_book_coverage"] == 0.0
    assert any("403" in str(value) for value in meta["source_errors"])


def test_research_fallback_comparison_cannot_become_executable_or_verified_coverage() -> None:
    markets = schedule_snapshot_markets(
        _targets(),
        captured_at=datetime(2026, 10, 1, 15, tzinfo=UTC),
    )
    frame, meta = build_market_intelligence(
        _projection(),
        _targets(),
        markets,
        _historical(),
    )

    assert frame.height == 3
    assert set(frame.get_column("quant_signal").to_list()) == {"PASS"}
    assert set(frame.get_column("market_execution_verified").to_list()) == {False}
    assert set(frame.get_column("market_book_count").to_list()) == {0}
    assert meta["moneyline"] == 0
    assert meta["spread"] == 0
    assert meta["total"] == 0
    assert meta["research_moneyline"] == 1
    assert meta["research_spread"] == 1
    assert meta["research_total"] == 1
    assert meta["complete_market_coverage"] == 0.0
    assert meta["research_market_coverage"] == 1.0

    executable, reason = validate_execution_row(frame.row(0, named=True))
    assert not executable
    assert "research-only" in reason


def test_research_fallback_never_enters_verified_line_history(tmp_path) -> None:
    markets = schedule_snapshot_markets(
        _targets(),
        captured_at=datetime(2026, 10, 1, 15, tzinfo=UTC),
    )
    path = tmp_path / "market_snapshots.csv"
    result = append_market_snapshots(markets, _targets(), path=path)

    assert result["captured_markets"] == 3
    assert result["eligible_verified_markets"] == 0
    assert result["skipped_research_only"] == 3
    assert result["appended_rows"] == 0
    assert load_market_snapshots(path).height == 0
