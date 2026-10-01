from __future__ import annotations

import polars as pl

from nfl import pipeline
from nfl.contracts import DataContractError


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
                "gameday": "2026-10-04",
            }
        ]
    )


def test_current_market_outage_returns_blocked_empty_state(monkeypatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
        raise DataContractError("no usable current NFL markets")

    monkeypatch.setattr(pipeline, "collect_current_markets", fail)
    markets, meta = pipeline._collect_current_markets_fail_closed(_targets(), week=4)

    assert markets == []
    assert meta["status"] == "BLOCKED"
    assert meta["books"] == 0
    assert meta["multi_book_coverage"] == 0.0
    assert meta["sources"] == []
    assert "no usable current NFL markets" in str(meta["reason"])
    assert meta["source_errors"] == [meta["reason"]]


def test_current_market_success_preserves_quotes_and_metadata(monkeypatch) -> None:
    quote = object()
    source_meta = {
        "status": "READY",
        "sources": ["ESPN"],
        "books": 1,
        "multi_book_coverage": 0.0,
        "source_errors": [],
    }

    def succeed(
        _targets: pl.DataFrame,
        *,
        week: int,
    ) -> tuple[list[object], dict[str, object]]:
        assert week == 4
        return [quote], source_meta

    monkeypatch.setattr(pipeline, "collect_current_markets", succeed)
    markets, meta = pipeline._collect_current_markets_fail_closed(_targets(), week=4)

    assert markets == [quote]
    assert meta == source_meta
